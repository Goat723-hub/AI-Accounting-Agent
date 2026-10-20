"""test_agent_logging.py —— Logger / run-summary 接进 langgraph 之后的集成测试。

跑法（在 AI-Accounting-Agent 目录下）：

    python -m unittest test_agent_logging

为什么和 test_logger.py 分开：

    那份测的是 logger 自己 —— 格式、handler、安全闸门，不依赖 langgraph，
    一百毫秒跑完。
    这份要**真的把图跑起来**：langgraph、openpyxl、临时 xlsx、
    还有一个真的起子进程的用例。慢得多，失败原因也完全不同。

    混在一起的话，一个环境问题会让格式测试跟着红，
    排查的人先被带偏半天。

两个用例类共用一个夹具（_GraphTestBase）：临时日志出口、临时规则库、
临时运行记录出口、一张编的工资表。它们要跑的是同一张图，
只是断言的东西不同 —— 一个看日志，一个看 summary。

全程用编造的数据（姓名、身份证号、金额都是假的），
不读任何真实工资表，也不会调用 LLM —— 用的表只含已知列名，
unknown_fields 是空的，那条路走不到 llm_rule_analysis。
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from openpyxl import Workbook

import llm_node
import logger
import memory
import payroll_agent
import run_summary
from test_logger import LINE_PATTERN, _drop_handlers, _lines_of, _log_text


PROJECT_DIR = Path(__file__).parent

# initial_state 的键，一个不多一个不少。
#
# 写死在这里是为了让「Logger 有没有偷偷往 State 里加字段」变成一条会红的测试。
# 以后真要给 State 加键，改这里 + 说清楚为什么 —— 那正是该停下来想一下的时刻。
EXPECTED_STATE_KEYS = [
    "calculation_result",
    "extracted_rows",
    "file_path",
    "halt_reason",
    "human_review",
    "llm_analysis",
    "memory_context",
    "new_memory_rules",
    "next_action",
    "parse_error",
    "problems",
    "raw_rows",
    "unknown_fields",
    "unresolved_fields",
    "validation_result",
]

# 子进程跑的那段代码。
#
# 为什么要起真进程：要验的是「异常退出之后日志还在不在文件里」，
# 而这件事在同进程里验不了 —— 同进程读文件，读到的是缓冲区之外的东西，
# 说明不了「进程没了它还在」。这里的子进程会带着一个未捕获的异常退出。
CHILD_CODE = """
import sys
from pathlib import Path

import logger
logger.LOG_FILE = Path(sys.argv[1])          # 换出口，别写进项目里的 logs/
logger.TRACE_FILE = Path(sys.argv[1]).with_name("trace.jsonl")   # 轨迹也是同一个目录

import run_summary
run_summary.SUMMARY_FILE = Path(sys.argv[2]) # 运行记录也一样，别写进项目里的 runs/

import memory
memory.MEMORY_PATH = Path(sys.argv[3])       # 别碰项目里那份真规则库

import payroll_agent


def boom(rows):
    raise ValueError("张三 的工资 12,000.00 元算不出来，身份证 110101199001011234")


payroll_agent.calculate_payroll = boom
payroll_agent.run(sys.argv[4])               # 异常一路抛出去，子进程非 0 退出
"""


def _matches(path: Path) -> list:
    """把日志文件按统一格式解析成一条条记录。对不上的行直接丢掉。"""
    return [
        matched
        for matched in (LINE_PATTERN.match(line) for line in _lines_of(path))
        if matched
    ]


def _events(matches: list) -> set:
    return {(m["node_name"], m["event"]) for m in matches}


def _trace_records() -> list:
    """把 logs/trace.jsonl 读成 dict 列表。文件不存在就是空列表。"""
    return [json.loads(line) for line in _lines_of(logger.TRACE_FILE)]


def _task_events(matches: list, event: str) -> list:
    return [m for m in matches if m["event"] == event]


@contextmanager
def _calculate_raises():
    """让 calculate 节点炸掉。

    补丁必须打在**函数真正被查找的那个模块**上。payroll_agent 里写的是
    `from calculation import calculate_payroll`，所以 calculate_node 调的是
    payroll_agent 模块里的那个全局名 —— 改 calculation.calculate_payroll
    它看不见。这是本项目第四次踩这个坑（前面三次见 memory.py 和
    payroll_agent 场景四的注释），规律是一样的：谁定义函数，就改谁模块里的全局名。

    异常消息故意带上姓名、金额和身份证号：这样「有没有泄露」是被真的验证过的，
    而不是只验证了「某个固定字符串没出现」。
    """
    original = payroll_agent.calculate_payroll

    def boom(rows):
        raise ValueError("张三 的工资 12,000.00 元算不出来，身份证 110101199001011234")

    payroll_agent.calculate_payroll = boom
    try:
        yield
    finally:
        # finally 里还原：用例中途失败也必须还回去，
        # 否则后面几条会一路用着这个炸掉的版本，红得莫名其妙
        payroll_agent.calculate_payroll = original


@contextmanager
def _fake_llm(input_tokens=1200, output_tokens=80, raises=None):
    """把 llm_node 的客户端换成假的，不真的联网。

    补丁打在 **llm_node** 模块上：谁定义 _build_client，就改谁模块里的全局名。
    这是本项目第四次还是第五次踩这条规矩了（见 _calculate_raises 的注释）。

    raises 传一个异常类型就模拟「调用失败」—— 那条路没有任何用量，
    正是最容易被记成「免费」的那种。
    """
    response = type(
        "FakeResponse",
        (),
        {
            "content": '{"candidates": [{"field": "季度激励", '
            '"suggested_mapping": "bonus", "confidence": 0.9}]}',
            "usage_metadata": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        },
    )()

    class _FakeClient:
        def invoke(self, messages):
            if raises is not None:
                raise raises("模拟的网络抖动")
            return response

    original = llm_node._build_client
    llm_node._build_client = lambda: _FakeClient()
    try:
        yield
    finally:
        llm_node._build_client = original


class _GraphTestBase(unittest.TestCase):
    """共用的夹具：四个出口全指向临时目录。

    四个都要换 —— 日志、节点轨迹、规则库、运行记录。任何一个漏了，
    用例就会往项目根目录写东西，跑第二遍时断言开始时灵时不灵。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._dir = Path(self._tmp.name)

        self._real_stdout = sys.stdout
        self._fake_stdout = io.StringIO()
        sys.stdout = self._fake_stdout

        self._real_log_file = logger.LOG_FILE
        logger.LOG_FILE = self._dir / "logs" / "agent.log"
        _drop_handlers()

        # 节点轨迹也要换。它和 agent.log 在同一层被写，
        # 但它是每个节点一行，量比日志大得多 ——
        # 漏了的话每跑一次测试就往项目里倒几十行
        self._real_trace_file = logger.TRACE_FILE
        logger.TRACE_FILE = self._dir / "logs" / "trace.jsonl"

        # 规则库也指向临时目录：这几条用例走的路不会写规则，
        # 但万一以后哪条路会写，也保证它写不到项目里那份真的
        self._real_memory_path = memory.MEMORY_PATH
        memory.MEMORY_PATH = self._dir / "memory.json"

        self._real_summary_file = run_summary.SUMMARY_FILE
        run_summary.SUMMARY_FILE = self._dir / "runs" / "summary.jsonl"

        self._xlsx = self._make_xlsx()

    def tearDown(self):
        _drop_handlers()
        logger.LOG_FILE = self._real_log_file
        logger.TRACE_FILE = self._real_trace_file
        memory.MEMORY_PATH = self._real_memory_path
        run_summary.SUMMARY_FILE = self._real_summary_file
        sys.stdout = self._real_stdout
        self._tmp.cleanup()

    def _make_xlsx(self, name: str = "payroll.xlsx", extra: list | None = None) -> str:
        """一张工资表。数据全是编的。

        extra 是额外加的列（[(列名, 值)]）。加一列没人认识的，
        流程才会走到 llm_rule_analysis —— 那是有 LLM 调用的那条路。
        默认不加，所以默认的用例跑不到 LLM。
        """
        header = ["工号", "姓名", "部门", "基本工资", "奖金", "扣款", "实发工资"]
        row = ["A001", "张伟", "研发部", 12000, 3000, 0, 11780]

        for column, value in extra or []:
            header.append(column)
            row.append(value)

        path = self._dir / name
        wb = Workbook()
        ws = wb.active
        ws.append(header)
        ws.append(row)
        wb.save(path)
        wb.close()
        return str(path)


class GraphLoggingTest(_GraphTestBase):

    # ---- 1. 每次运行一个独立的 task_id ----

    def test_每次运行生成不同的_task_id(self):
        payroll_agent.run(self._xlsx)
        payroll_agent.run(self._xlsx)

        started = [m["task_id"] for m in _task_events(_matches(logger.LOG_FILE), "task_started")]

        self.assertEqual(len(started), 2)
        self.assertNotEqual(started[0], started[1], "两次运行必须是两个不同的 task_id")
        self.assertNotIn("-", started, "任务级事件不该是占位符")

    # ---- 2. 同一次运行共用一个 task_id ----

    def test_同一次运行里所有日志共用一个_task_id(self):
        payroll_agent.run(self._xlsx)

        task_ids = {m["task_id"] for m in _matches(logger.LOG_FILE)}

        self.assertEqual(
            len(task_ids), 1, f"一次运行只该有一个 task_id，实际出现了：{task_ids}"
        )
        self.assertNotIn("-", task_ids, "节点日志没取到上下文里的 task_id")

    def test_两次运行的日志用_task_id_分得开(self):
        payroll_agent.run(self._xlsx)
        payroll_agent.run(self._xlsx)

        matches = _matches(logger.LOG_FILE)
        started = _task_events(matches, "task_started")
        first, second = started[0]["task_id"], started[1]["task_id"]

        # 一次运行该有 10 条节点日志（5 个节点 × 开始 + 结束）
        # 加 2 条任务日志（task_started / task_completed）= 12 条。
        # 两边条数相等，说明两轮的记录没有互相串进来、也没有漏掉。
        self.assertEqual(sum(1 for m in matches if m["task_id"] == first), 12)
        self.assertEqual(sum(1 for m in matches if m["task_id"] == second), 12)

    # ---- 3. 正常执行的节点有开始和结束 ----

    def test_节点正常执行有开始和结束事件(self):
        payroll_agent.run(self._xlsx)

        events = _events(_matches(logger.LOG_FILE))
        started = {name for name, event in events if event == "node_started"}
        completed = {name for name, event in events if event == "node_completed"}
        failed = {name for name, event in events if event == "node_failed"}

        # 正常表走的那条路
        self.assertEqual(
            started, {"parse", "extract", "memory_lookup", "calculate", "validate"}
        )
        self.assertEqual(started, completed, "有节点开始了却没结束")
        self.assertEqual(failed, set())

    # ---- 4 & 5. 节点抛异常 / 主流程抛异常 ----

    def test_节点抛异常记_node_failed_且异常不被吞掉(self):
        with _calculate_raises():
            with self.assertRaises(ValueError):
                payroll_agent.run(self._xlsx)

        matches = _matches(logger.LOG_FILE)
        failed = [m for m in matches if m["event"] == "node_failed"]

        self.assertEqual([m["node_name"] for m in failed], ["calculate"])
        self.assertEqual(failed[0]["status"], "failed")
        self.assertEqual(failed[0]["error"], "ValueError")
        self.assertNotIn(
            ("calculate", "node_completed"),
            _events(matches),
            "抛异常的那个节点不该同时有一条「成功」",
        )

    def test_未捕获异常记_task_failed_且不伪造成功状态(self):
        with _calculate_raises():
            with self.assertRaises(ValueError):
                payroll_agent.run(self._xlsx)

        matches = _matches(logger.LOG_FILE)

        self.assertEqual(len(_task_events(matches, "task_failed")), 1)
        self.assertEqual(
            _task_events(matches, "task_completed"), [], "异常路径不该有「完成」事件"
        )

        failed = _task_events(matches, "task_failed")[0]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"], "ValueError")
        self.assertNotIn("-", failed["task_id"])

    # ---- 6. 业务失败不是程序崩溃 ----

    def test_业务失败不被当成程序崩溃(self):
        """文件不存在：流程在 exception 节点收住，**正常返回**。

        它不是异常 —— run() 不该抛，也不该记 task_failed。
        该记的是 task_completed + status=failed（结束了，但没算成）。
        """
        missing = str(self._dir / "这张表不存在.xlsx")

        final = payroll_agent.run(missing)          # 不抛异常

        self.assertTrue(final["halt_reason"], "业务失败该有 halt_reason")

        matches = _matches(logger.LOG_FILE)
        self.assertEqual(_task_events(matches, "task_failed"), [])

        completed = _task_events(matches, "task_completed")
        self.assertEqual(len(completed), 1)
        self.assertEqual(
            completed[0]["status"], "failed", "业务上没算成，状态要如实记 failed"
        )
        self.assertIn("exception", {name for name, _ in _events(matches)})

    def test_正常跑完记_success(self):
        payroll_agent.run(self._xlsx)

        completed = _task_events(_matches(logger.LOG_FILE), "task_completed")
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["status"], "success")

    # ---- 9. 敏感内容 ----

    def test_异常里的姓名金额身份证不写进日志和控制台(self):
        with _calculate_raises():
            with self.assertRaises(ValueError):
                payroll_agent.run(self._xlsx)

        # 第三个出口也一起验。它是这四处里**唯一收整个 State 的**，
        # 所以它是泄露风险最大的那一个 —— 漏了它就等于没验。
        output = (
            _log_text()
            + self._fake_stdout.getvalue()
            + logger.TRACE_FILE.read_text(encoding="utf-8")
        )

        # 12,000.00 带逗号和小数点，不可能与 task_id 的十六进制撞上 ——
        # 用纯数字做断言会偶发地红（task_id 里恰好出现那几个数字）
        for secret in ("张三", "12,000.00", "110101199001011234"):
            self.assertNotIn(secret, output, f"「{secret}」不该出现在日志、控制台或轨迹里")

        # 只删干净是能过这条的，但那样轨迹就不回答任何问题了。
        # 所以顺带要求：该在的东西在 —— 失败那个节点的形状和错误类型都有
        failed = [record for record in _trace_records() if record["status"] == "failed"]
        self.assertEqual([record["node"] for record in failed], ["calculate"])
        self.assertEqual(failed[0]["error_type"], "ValueError")
        self.assertEqual(failed[0]["input"]["extracted_rows"], "list(1)")

        # 哪一步炸的：那个被换掉的 calculate_payroll（用例里打在 payroll_agent 上）
        # ——这是**代码坐标**，和异常消息里那些姓名金额无关
        self.assertTrue(
            failed[0]["error_at"].endswith(" boom"), failed[0]["error_at"]
        )

    # ---- 11. 节点轨迹落盘 ----

    def test_不传_sink_也照样落盘(self):
        """落盘和内存副本是**两个开关**，这条把这件事钉死。

        节点轨迹要能回答「上周三那次为什么卡在 calculate」，
        而那次没人会记得要先传个列表进来。
        """
        payroll_agent.run(self._xlsx)     # 故意不传 trace

        records = _trace_records()
        self.assertEqual(
            [record["node"] for record in records],
            ["parse", "extract", "memory_lookup", "calculate", "validate"],
        )
        self.assertEqual([record["status"] for record in records], ["success"] * 5)

    def test_轨迹和日志能按_task_id_对上(self):
        """两个文件结合着看，靠的就是这一格。对不上就只是两份各自孤立的记录。"""
        payroll_agent.run(self._xlsx)

        logged = {m["task_id"] for m in _matches(logger.LOG_FILE)}
        traced_ids = {record["task_id"] for record in _trace_records()}

        self.assertEqual(len(traced_ids), 1, "一次运行只该有一个 task_id")
        self.assertTrue(traced_ids <= logged, "轨迹里的 task 必须在日志里找得到")

    def test_一个节点都没跑成时_轨迹只记跑到哪儿(self):
        """文件读不进来那次：parse 就停了。

        轨迹要如实反映「只跑到这儿」—— 后面八个节点**不该有行**。
        补几行空记录的话，看轨迹的人会以为它们跑过。
        """
        final = payroll_agent.run(self._dir / "并不存在.xlsx")

        self.assertTrue(final["halt_reason"])
        self.assertEqual(
            [record["node"] for record in _trace_records()], ["parse", "exception"]
        )

    # ---- 10. 没有改变原有行为 ----

    def test_trace_和_State_没被改动(self):
        trace = []
        final = payroll_agent.run(self._xlsx, trace=trace)

        # trace 还在，走的还是那条路
        self.assertEqual(
            [entry["node"] for entry in trace],
            ["parse", "extract", "memory_lookup", "calculate", "validate"],
        )

        # 一条轨迹有哪几个键，钉死。
        #
        # 这里原来是四个键（node / input / changed / next_action）。
        # 加进来的四个各自有用，缺一个都查不动：
        #
        #   time        和 agent.log 对齐时间轴时要用。轨迹和日志是两个
        #               文件，对不上时间就只能靠 task_id + 顺序猜
        #   task_id     两个文件**唯一**的关联键。没有它，结合着看无从谈起
        #   status      失败那一行是整份文件里最值钱的，得有东西能认出它
        #   error_type  失败的话是什么错。只记类型名，和 logger 一个口径
        #   error_at    哪一步炸的（文件:行号 函数名）。
        #               只有类型名的话，「KeyError」这种到处都有的错误
        #               还是得靠重跑一遍才知道是哪儿
        #
        # 之所以要钉死，是因为多一个键意味着多一份可能落盘的东西 ——
        # 以后谁想往里加「顺便把 State 也存一份」，这条会红，
        # 而那一刻正是该停下来想清楚的时候。
        self.assertEqual(
            sorted(trace[0].keys()),
            [
                "changed",
                "error_at",
                "error_type",
                "input",
                "next_action",
                "node",
                "status",
                "task_id",
                "time",
            ],
        )

        # State 的键一个都没多、没少
        self.assertEqual(
            sorted(payroll_agent.initial_state("x.xlsx").keys()),
            sorted(EXPECTED_STATE_KEYS),
        )
        self.assertNotIn("task_id", payroll_agent.PayrollState.__annotations__)

        # 业务结果还是老样子
        self.assertEqual(len(final["calculation_result"]["people"]), 1)
        self.assertIsNone(final["halt_reason"])
        self.assertIn("passed", final["validation_result"])

    def test_绕过_run_直接跑图时_task_id_是占位符不崩(self):
        """已知边界，写成测试免得以后被当成 bug 查半天。

        task_id 是 run() 发的。直接 build_graph().invoke() 就没有它 ——
        Node 日志照样记，只是 task 那一格是占位符。
        """
        graph = payroll_agent.build_graph()
        final = graph.invoke(payroll_agent.initial_state(self._xlsx))

        self.assertIsNone(final["halt_reason"])

        matches = _matches(logger.LOG_FILE)
        self.assertEqual({m["task_id"] for m in matches}, {"-"})
        self.assertEqual(_task_events(matches, "task_started"), [])

    # ---- 11. 真进程：异常退出后日志还在 ----

    def test_进程异常退出后日志仍然留在文件里(self):
        log_path = self._dir / "子进程" / "agent.log"
        summary_path = self._dir / "子进程" / "summary.jsonl"
        memory_path = self._dir / "子进程_memory.json"

        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                CHILD_CODE,
                str(log_path),
                str(summary_path),
                str(memory_path),
                self._xlsx,
            ],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        self.assertNotEqual(proc.returncode, 0, "子进程本该带着未捕获的异常退出")
        self.assertIn("ValueError", proc.stderr, "异常没有被吞掉，traceback 应该还在")

        text = log_path.read_text(encoding="utf-8")
        self.assertIn("event=task_failed", text)
        self.assertIn("event=node_failed", text)

        # 运行记录也落在**临时目录**里 —— 这条断言看着多余，其实是钉住
        # 一个真出过的事：子进程那段代码一开始漏换了 run_summary.SUMMARY_FILE，
        # 于是每跑一次测试就往项目根的 runs/summary.jsonl 里塞一行。
        # 用例全绿，只有那个文件在悄悄长大 —— 和「日志写到错的地方」是同一类错。
        summary_text = summary_path.read_text(encoding="utf-8")
        self.assertIn('"failed_node": "calculate"', summary_text)

        # 只对**日志文件**断言。日志不该有这些。
        for secret in ("张三", "12,000.00", "110101199001011234"):
            self.assertNotIn(secret, text, f"「{secret}」不该进日志")

        # 反过来，stderr 里**应该**有 —— 这条断言是在把边界写清楚，不是漏了。
        #
        # 那句 traceback 是 Python 解释器自己打的，属于「异常照原样往上抛」
        # 这个原有语义的一部分（也是本阶段明确要保留的：不吞异常）。
        # 想让 stderr 也不出现姓名，只有两个办法：把异常吞掉，或者改掉
        # 抛异常的人写的消息 —— 两个都超出 Logger 该管的事。
        #
        # 所以「日志不泄露」 != 「进程 stderr 不泄露」。
        # 真正该在日志里守住的，是下一条用例断言的那些。
        self.assertIn("张三", proc.stderr, "traceback 原文由解释器打，不属于日志出口")


class RunSummaryTest(_GraphTestBase):
    """run-summary 的集成测试。

    它记的是「一次运行」，所以断言之前必须真的跑一次运行 ——
    路径和图那边完全一样，只是断言的东西从 agent.log 换成了
    runs/summary.jsonl。
    """

    def _records(self) -> list:
        return run_summary.read_all()

    def _only(self) -> dict:
        records = self._records()
        self.assertEqual(len(records), 1, f"该正好一行，实际 {len(records)} 行")
        return records[0]

    # ---- 1. 正常跑完 ----

    def test_正常跑完记_success(self):
        payroll_agent.run(self._xlsx)

        record = self._only()
        self.assertEqual(record["status"], "success")
        self.assertIsNone(record["failed_node"], "没失败过就不该有失败节点")
        self.assertIsNone(record["error_type"])
        self.assertTrue(record["task_id"])
        self.assertLessEqual(record["start_time"], record["end_time"])

    # ---- 2. 业务失败和节点失败是两件事 ----

    def test_业务失败记_failed_但没有失败节点(self):
        """文件不存在：流程在 exception 节点收住，**没有任何节点抛异常**。

        所以 status 是 failed（确实没算成），failed_node 却是 null。
        这两个必须分得开 —— 业务失败要去找业务方确认口径，
        节点失败要去看 traceback，处理的人不一样。
        """
        payroll_agent.run(str(self._dir / "这张表不存在.xlsx"))

        record = self._only()
        self.assertEqual(record["status"], "failed")
        self.assertIsNone(record["failed_node"])
        self.assertIsNone(record["error_type"])

    # ---- 3. 节点抛异常 ----

    def test_节点抛异常时记下是哪个节点和什么类型(self):
        with _calculate_raises():
            with self.assertRaises(ValueError):
                payroll_agent.run(self._xlsx)

        record = self._only()
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failed_node"], "calculate")
        self.assertEqual(record["error_type"], "ValueError")

    # ---- 4. 和日志用同一个 task_id ----

    def test_task_id_和日志里的是同一个(self):
        payroll_agent.run(self._xlsx)

        logged = {m["task_id"] for m in _matches(logger.LOG_FILE)}
        self.assertEqual({self._only()["task_id"]}, logged)

    # ---- 5. 字段表是钉死的 ----

    def test_只记元数据_字段一个不多一个不少(self):
        """这条是给以后的人看的：想往 summary 里加字段，先读这里的理由。

        summary 是长期留存、会被人翻的东西。往里加一个 raw_rows，
        等于给工资数据开了一条**新的**落盘路径 —— 而且是绕过 logger
        那三道闸门开的。

        ---- V2 加的三个（fail_category / review / schema_version）----

        它们过闸门的方式和 raw_rows 不一样，逐条交代：

        fail_category
            一个**枚举字符串**（input_error / business_halt / llm_error /
            agent_error），不是自由文本。枚举不可能装下工资数据 ——
            这是它敢进来的原因。

        review
            装的是**字段名**（客户表里的列名：餐补、交通补贴），
            外加一个 reviewer 和几个计数。**不含金额、不含人员信息。**
            「只记键名不记值」这条口径没变，和 logger.describe() 是同一个路子。

        schema_version
            一个整数。

        换句话说：这三个都是**分类标签和键名**，没有一个是客户填进来的值。
        """
        payroll_agent.run(self._xlsx)

        self.assertEqual(
            sorted(self._only().keys()),
            [
                "end_time",
                "error_type",
                "fail_category",
                "failed_node",
                "input_tokens",
                "llm_calls",
                "output_tokens",
                "review",
                "schema_version",
                "start_time",
                "status",
                "task_id",
            ],
        )
        # 钉死的字段表和模块里那份必须一致 —— 改了 write() 忘了改常量，
        # normalize() 就会漏补新字段，而那种错在读老行时才出现。
        self.assertEqual(sorted(self._only().keys()), sorted(run_summary.RECORD_FIELDS))

    # ---- 6. 敏感数据一个字都没有 ----

    def test_工资数据不会进运行记录(self):
        with _calculate_raises():
            with self.assertRaises(ValueError):
                payroll_agent.run(self._xlsx)

        text = run_summary.SUMMARY_FILE.read_text(encoding="utf-8")

        for secret in ("张三", "12,000.00", "110101199001011234", "算不出来"):
            self.assertNotIn(secret, text, f"「{secret}」不该进运行记录")

        # 类型名要留下 —— 定位靠它，而且它不含数据
        self.assertIn("ValueError", text)

    # ---- 7. 两次运行不串 ----

    def test_上一次失败的节点不会漏到下一次(self):
        """failed_node 存在执行流上，而执行流是**复用**的。

        task_scope 退出时如果不把它清掉，第二次运行明明成功了，
        summary 里却还写着 calculate —— 而且带着 failed 状态。
        这是最难查的一类错：日志看着全对，只有 summary 在说谎。
        """
        with _calculate_raises():
            with self.assertRaises(ValueError):
                payroll_agent.run(self._xlsx)

        payroll_agent.run(self._xlsx)

        second = self._records()[1]
        self.assertEqual(second["status"], "success")
        self.assertIsNone(second["failed_node"], "上一次的失败节点漏到这一次了")

    # ---- 8. 记录功能坏掉不能把流程带崩 ----

    def test_运行记录写不进去时不崩且不顶掉原异常(self):
        """这条是 summary 和 logger 最不一样的地方。

        它被调用的位置在 run() 的 except 分支里，下一行就是 raise。
        要是它自己抛 OSError，抛出来的就变成 OSError 了 ——
        **原来那个 ValueError 被顶掉**，等于「为了记一次失败，
        把失败的原因弄丢了」。
        """
        blocker = self._dir / "占位文件"
        blocker.write_text("这是文件不是目录", encoding="utf-8")
        run_summary.SUMMARY_FILE = blocker / "summary.jsonl"

        with _calculate_raises():
            with self.assertRaises(ValueError):
                payroll_agent.run(self._xlsx)      # 抛的必须是 ValueError

        self.assertFalse(run_summary.SUMMARY_FILE.exists())
        self.assertIn("summary_write_failed", _log_text())

    # ---- 9. LLM 用量 ----

    def test_LLM_用量进运行记录(self):
        """一条未知列才会走到 llm_rule_analysis —— 那是有 LLM 调用的那条路。"""
        xlsx = self._make_xlsx("有未知列.xlsx", extra=[("季度激励", 3000)])

        with _fake_llm(input_tokens=1200, output_tokens=80):
            payroll_agent.run(xlsx)

        record = self._only()
        self.assertEqual(record["llm_calls"], 1)
        self.assertEqual(record["input_tokens"], 1200)
        self.assertEqual(record["output_tokens"], 80)

    def test_没走_LLM_的时候三个数都是_0(self):
        """列全认识，走不到 llm 节点。这时候 0 是**真的** 0，不是缺失。"""
        payroll_agent.run(self._xlsx)

        record = self._only()
        self.assertEqual(record["llm_calls"], 0)
        self.assertEqual(record["input_tokens"], 0)
        self.assertEqual(record["output_tokens"], 0)

    def test_LLM_调用失败时计数但不记_token(self):
        """这条是三个数里 llm_calls 存在的全部理由。

        只看 token 的话，「没调用」和「调了但失败」都是 0/0 ——
        失败看起来是免费的，而它恰恰是最该被看见的那种。
        """
        xlsx = self._make_xlsx("有未知列.xlsx", extra=[("季度激励", 3000)])

        with _fake_llm(raises=TimeoutError):
            payroll_agent.run(xlsx)

        record = self._only()
        self.assertEqual(record["llm_calls"], 1, "失败的调用也是调用")
        self.assertEqual(record["input_tokens"], 0, "没拿到用量就记 0，不是编一个数")
        self.assertEqual(record["output_tokens"], 0)

    def test_跑两次不会把用量累到一起(self):
        """用量和执行流上的 marks 一样，每次运行从零开始。

        第二轮要换一份空的规则库才跑得到 LLM：第一轮已经把「季度激励」
        的候选存成 pending 了，不换的话 memory_lookup 会认出这个字段，
        直接去找人 —— 压根不问 LLM，用量当然是 0。

        那个行为本身是对的（问过一遍就不再问），这里只是别让它把
        「用量有没有串」这件事搅浑。
        """
        xlsx = self._make_xlsx("有未知列.xlsx", extra=[("季度激励", 3000)])

        with _fake_llm(input_tokens=100, output_tokens=10):
            payroll_agent.run(xlsx)

        memory.MEMORY_PATH = self._dir / "第二轮_memory.json"

        with _fake_llm(input_tokens=200, output_tokens=20):
            payroll_agent.run(xlsx)

        first, second = self._records()
        self.assertEqual(first["input_tokens"], 100)
        self.assertEqual(
            second["input_tokens"], 200, "第二次的用量里混进了第一次的"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
