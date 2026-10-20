"""test_logger.py —— logger.py 的最小验证。

跑法（在 AI-Accounting-Agent 目录下）：

    python -m unittest test_logger

或者：

    python test_logger.py

用标准库 unittest，不引 pytest：这个项目到现在没有测试框架，
Logger 也用不着为了几条断言装一个第三方包。
以后真引了 pytest，这份文件照样能被它跑 —— pytest 认得 unittest 的 TestCase。

每个用例都把出口换到临时目录 / 内存里。
理由和 memory.py 演示里改 MEMORY_PATH 是同一条：
logger 默认往 logs/agent.log 追加写，拿项目根目录那个文件做断言，
跑第二遍时里面混着上一遍的日志，断言就时灵时不灵了。
节点轨迹（logs/trace.jsonl）是同一条规矩 —— 它是后加的出口，
一样不能拿项目里那个文件做断言。

用例里的姓名、身份证号、银行卡号、工资全是编的，不用任何真实员工信息。
"""

import importlib
import io
import json
import logging
import re
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import logger


# 统一格式的「反向解析器」。
#
# 这里按**约定的格式**另写一份正则，而不是复用 logger.LOG_FORMAT —— 拿同一个
# 常量去验它自己，永远是绿的。写死一份之后，谁改了 LOG_FORMAT，这条才会红，
# 而它红的时候正是该有人想清楚「格式变了，下游有没有人在按老格式解析」的时候。
LINE_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r" \| task=(?P<task_id>[^|]*)"
    r" \| node=(?P<node_name>[^|]*)"
    r" \| event=(?P<event>[^|]*)"
    r" \| status=(?P<status>[^|]*)"
    r" \| error=(?P<error>[^|]*)"
    r" \| (?P<message>.*)$"
)


def _drop_handlers() -> None:
    """把 logger 上挂着的 handler 全摘掉。

    每个用例开头都要来一次：handler 挂在 logging 的 logger 对象上，
    那个对象比用例活得久。不清的话，第二个用例还在往第一个用例的
    临时目录里写 —— 日志看起来正常，只是写到了错的地方。
    """
    log = logging.getLogger(logger.LOGGER_NAME)
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()


def _log_text() -> str:
    """日志文件的全部内容。文件还不存在（比如出口降级了）就是空串。"""
    if not logger.LOG_FILE.exists():
        return ""
    return logger.LOG_FILE.read_text(encoding="utf-8")


def _lines_of(path: Path) -> list:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _trace_records() -> list:
    """把 trace.jsonl 读成 dict 列表。文件不存在就是空列表。"""
    return [json.loads(line) for line in _lines_of(logger.TRACE_FILE)]


def _boom_with_business_data():
    """抛一个**消息里带业务数据**的异常，用来验它不会落盘。

    写成模块级的函数（不写成 lambda）是为了让 error_where 记下来的
    函数名是一个能对得上的名字。
    """
    raise ValueError("张三 的工资 12,000.00 元算不出来，卡号 6222021234567890123")


@contextmanager
def _tiny_limit(bytes_: int = 10):
    """把「多大算该滚了」改小，好让用例不用真写 5 MB。

    改的是 logger 模块里的全局名 —— 这个项目踩过四五次的那条规矩：
    谁定义常量，就改谁模块里的全局名。
    """
    original = logger.MAX_FILE_BYTES
    logger.MAX_FILE_BYTES = bytes_
    try:
        yield
    finally:
        logger.MAX_FILE_BYTES = original


class LoggerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

        # 控制台出口也收到内存里：一来断言得了，二来跑测试时不会往终端刷日志
        self._real_stdout = sys.stdout
        self._fake_stdout = io.StringIO()
        sys.stdout = self._fake_stdout

        # 换出口。必须在第一次 get_logger() 之前改 ——
        # handler 是那一刻按 LOG_FILE / sys.stdout 建出来的
        self._real_log_file = logger.LOG_FILE
        logger.LOG_FILE = Path(self._tmp.name) / "logs" / "agent.log"

        self._real_trace_file = logger.TRACE_FILE
        logger.TRACE_FILE = Path(self._tmp.name) / "logs" / "trace.jsonl"

        _drop_handlers()

    def tearDown(self):
        _drop_handlers()
        logger.LOG_FILE = self._real_log_file
        logger.TRACE_FILE = self._real_trace_file
        sys.stdout = self._real_stdout
        self._tmp.cleanup()

    # ---- 1. 控制台 ----

    def test_日志输出到控制台(self):
        logger.task_started("t-001", message="开跑")

        printed = self._fake_stdout.getvalue()
        self.assertIn("task=t-001", printed)
        self.assertIn("event=task_started", printed)
        self.assertIn("开跑", printed)

    # ---- 2. 文件 ----

    def test_日志写入文件并自动创建_logs_目录(self):
        self.assertFalse(
            logger.LOG_FILE.parent.exists(), "临时目录一开始不该有 logs/"
        )

        logger.node_started("parse", task_id="t-001")

        self.assertTrue(logger.LOG_FILE.exists(), "logs/ 和 agent.log 应该被自动创建")
        self.assertIn("node=parse", logger.LOG_FILE.read_text(encoding="utf-8"))

    # ---- 3. 格式 ----

    def test_字段格式统一(self):
        logger.node_failed(
            "parse",
            task_id="t-001",
            message="文件读不进来",
            error=FileNotFoundError("没这个文件"),
        )

        lines = _lines_of(logger.LOG_FILE)
        self.assertEqual(len(lines), 1)

        matched = LINE_PATTERN.match(lines[0])
        self.assertIsNotNone(matched, f"格式对不上：{lines[0]}")

        self.assertEqual(matched["task_id"], "t-001")
        self.assertEqual(matched["node_name"], "parse")
        self.assertEqual(matched["event"], "node_failed")
        self.assertEqual(matched["status"], "failed")
        # 只有类型名，异常消息（这里是「没这个文件」）不记 —— 见 error_type
        self.assertEqual(matched["error"], "FileNotFoundError")
        self.assertEqual(matched["message"], "文件读不进来")
        self.assertEqual(
            matched["timestamp"][:10], date.today().isoformat(), "时间戳应该是今天"
        )

    # ---- 4. 可选字段缺失 ----

    def test_可选字段缺失不会崩(self):
        # 只给一个事件名，别的全不给
        logger.log_event(logger.NODE_STARTED)

        # 失败但没带异常
        logger.node_failed("validate")

        # error 传了字符串（不是异常对象）：类型报出来，内容不记 ——
        # 见 error_type。要写人看的一句话请用 message。
        logger.node_failed("calculate", error="计算返回了空结果")

        lines = _lines_of(logger.LOG_FILE)
        self.assertEqual(len(lines), 3, "三条都该落盘，一条都不能丢")

        first = LINE_PATTERN.match(lines[0])
        self.assertIsNotNone(first, f"格式对不上：{lines[0]}")
        self.assertEqual(first["event"], "node_started")
        for key in ("task_id", "node_name", "status", "error"):
            self.assertEqual(first[key], "-", f"{key} 该用占位符补上")
        self.assertEqual(first["message"], "")

        third = LINE_PATTERN.match(lines[2])
        self.assertEqual(third["error"], "<str>", "字符串内容不该被记进 error")

    # ---- 5. 重复初始化 ----

    def test_重复初始化不会重复输出(self):
        logger.task_started("t-002", message="第一次")

        # 模拟「模块被多次导入 / get_logger 被反复调用」
        logger.get_logger()
        logger.get_logger()

        log = logging.getLogger(logger.LOGGER_NAME)
        self.assertEqual(len(log.handlers), 2, "handler 被装重了")

        # 模拟模块被重新加载：模块顶层的常量会归零，
        # 但 handler 挂在 logger 对象上，不受 reload 影响 ——
        # 这正是「判断放在 logger.handlers 上」而不是放在模块变量上的原因。
        # 路径要先记下来：reload 之后 logger.LOG_FILE 会变回项目里的那个。
        log_path = logger.LOG_FILE
        importlib.reload(logger)
        logger.task_started("t-002", message="第二次")

        text = log_path.read_text(encoding="utf-8")
        self.assertEqual(text.count("第一次"), 1)
        self.assertEqual(text.count("第二次"), 1)
        self.assertEqual(len(_lines_of(log_path)), 2, "一共两条，不是四条")

    # ---- 6. 不泄露业务数据 ----

    def test_不会泄露工资数据或完整_state(self):
        # 编的假表：姓名、身份证号、银行卡号、工资全部是假的
        state = {
            "file_path": "data/payroll_demo.xlsx",
            "raw_rows": [
                {
                    "姓名": "张三",
                    "身份证号": "110101199001011234",
                    "银行卡号": "6222021234567890123",
                    "基本工资": 12000,
                }
            ],
            "human_review": {"reviewer": "张三"},
            "problems": [],
        }

        logger.node_completed(
            "parse", task_id="t-003", message=f"进 {logger.describe(state)}"
        )

        output = logger.LOG_FILE.read_text(encoding="utf-8") + self._fake_stdout.getvalue()

        for secret in ("张三", "110101199001011234", "6222021234567890123", "12000"):
            self.assertNotIn(secret, output, f"「{secret}」不该出现在日志里")

        # 键名和形状要留下 —— 安全不等于没用，这条日志得让排查的人看懂
        self.assertIn("raw_rows", output)
        self.assertIn("human_review", output)
        self.assertIn("list(", output)

    def test_异常消息里的姓名和金额不会进日志(self):
        """这条以前是反过来的 —— 它断言姓名**会**出现在日志里。

        那等于把「异常消息原文落盘」当成设计记下来了。可日志这一层
        没有任何办法判断哪条异常消息安全：写 raise 的人写什么就是什么，
        截断也不是脱敏。所以这一版只记类型名，这条测试跟着翻过来。
        """
        logger.node_failed(
            "calculate",
            message="节点抛出异常",
            error=ValueError("张三 的工资 12,000.00 元算不出来，身份证 110101199001011234"),
        )

        line = _lines_of(logger.LOG_FILE)[0]
        self.assertIn("error=ValueError", line)   # 类型名要留下，定位靠它
        for secret in ("张三", "12,000.00", "110101199001011234"):
            self.assertNotIn(secret, line, f"「{secret}」不该出现在日志里")

    def test_超长和换行消息被压成一行(self):
        logger.log_event(logger.NODE_STARTED, "第一行\n第二行" + "长" * 900)

        lines = _lines_of(logger.LOG_FILE)
        self.assertEqual(len(lines), 1, "换行没被压掉，一条日志占了好几行")
        self.assertIn("第一行 第二行", lines[0])
        self.assertIn("已截断", lines[0])

    # ---- 7. 安全闸门（补）----

    def test_任意业务对象不会被序列化进日志(self):
        """不只验证「某个固定姓名被藏起来」，而是验证任何对象都不被展开。

        传 dict 就记 "<dict>"，传 list 就记 "<list>" ——
        看得出「有人传错了」，但一个字的数据都没出去。
        """
        fake_state = {
            "raw_rows": [
                {"姓名": "张三", "身份证号": "110101199001011234", "基本工资": 12000}
            ],
            "file_path": "data/张三所在公司_工资表.xlsx",
        }

        logger.node_started("parse", message=fake_state)          # 手滑传了整个 State
        logger.node_failed("parse", error=fake_state)             # 手滑传进 error
        logger.log_event(                                          # 每个字段都传错
            logger.NODE_STARTED,
            fake_state,
            task_id=fake_state,
            node_name=fake_state,
        )

        text = _log_text() + self._fake_stdout.getvalue()

        for secret in ("张三", "110101199001011234", "12000", "张三所在公司"):
            self.assertNotIn(secret, text, f"「{secret}」不该出现在日志里")

        # 键名也不该出去 —— 只报类型，不报里面装了什么
        for key in ("raw_rows", "file_path", "基本工资", "姓名"):
            self.assertNotIn(key, text)

        self.assertIn("<dict>", text, "要看得见「有人传错了」")

    # ---- 8. task_id 的传递 ----

    def test_task_id_没传时从执行流上取(self):
        logger.node_started("parse")                    # 外面：没有，占位符

        with logger.task_scope("t-ctx"):
            logger.node_started("parse")                # 里面：自动带上

        logger.node_started("parse")                    # 出来：回到占位符

        tasks = [
            LINE_PATTERN.match(line)["task_id"] for line in _lines_of(logger.LOG_FILE)
        ]
        self.assertEqual(tasks, ["-", "t-ctx", "-"])

    def test_task_scope_退出时回到上一层而不是清空(self):
        """嵌套时 reset(token) 该回到外层，不是把外层也抹掉。"""
        with logger.task_scope("outer"):
            logger.node_started("parse")
            with logger.task_scope("inner"):
                logger.node_started("parse")
            logger.node_started("parse")                # 该是 outer，不是 "-"

        tasks = [
            LINE_PATTERN.match(line)["task_id"] for line in _lines_of(logger.LOG_FILE)
        ]
        self.assertEqual(tasks, ["outer", "inner", "outer"])

    def test_每次生成的_task_id都不一样(self):
        self.assertEqual(len({logger.new_task_id() for _ in range(200)}), 200)

    def test_失败节点能穿过拷贝出来的_context_传回外层(self):
        """这条钉的是一个**实测出来的**坑，不是设计出来的。

        contextvars 只向下传播。LangGraph 把每个节点放在拷贝出来的
        context 里跑（等价于 contextvars.copy_context().run(...)），所以：

            节点里读 run() 设的 task_id      可以
            节点里 set() 的值，run() 读到    不行

        task_id 是「往下发、节点只读」，正好没事。
        failed_node 是「节点写、run() 读」，正好踩上 —— 第一版就是
        ContextVar[str]，跑起来 failed_node 永远是 None，而且一声不吭。

        所以 failed_node 装在可变对象里（拷的是引用，不是内容）。
        copy_context 在这里是模拟 LangGraph 的行为，不用真起一个图。
        """
        import contextvars

        with logger.task_scope("t-1"):
            # 这一段就是「节点在拷贝的 context 里跑」
            contextvars.copy_context().run(
                lambda: logger.node_failed("calculate", message="节点抛出异常")
            )

            self.assertEqual(
                logger.last_failed_node(),
                "calculate",
                "节点里记的失败没有冒到外层 —— contextvars 的老问题又回来了",
            )

    def test_下一次运行的失败节点不会被上一次污染(self):
        with logger.task_scope("t-1"):
            logger.node_failed("calculate")
            self.assertEqual(logger.last_failed_node(), "calculate")

        with logger.task_scope("t-2"):
            self.assertIsNone(
                logger.last_failed_node(), "上一次的失败节点漏到这一次了"
            )

    # ---- LLM 用量 ----

    def test_llm_用量累加且能穿过拷贝的_context(self):
        """和 failed_node 同一个坑：写在节点里、读在 run() 里。

        区别是它是**累加**而不是覆盖 —— marks.llm_calls += 1，
        所以不能靠「最后一次 set 赢了」蒙混过去。
        """
        import contextvars

        with logger.task_scope("t-1"):
            contextvars.copy_context().run(
                lambda: logger.note_llm_call(input_tokens=100, output_tokens=10)
            )
            contextvars.copy_context().run(
                lambda: logger.note_llm_call(input_tokens=200, output_tokens=20)
            )

            self.assertEqual(
                logger.llm_usage(),
                {"llm_calls": 2, "input_tokens": 300, "output_tokens": 30},
            )

    def test_没拿到用量的调用只计数不记_token(self):
        """「没拿到」必须和「拿到了 0」分开 —— 否则失败看起来是免费的。"""
        with logger.task_scope("t-1"):
            logger.note_llm_call()                       # 失败：没有用量

            self.assertEqual(
                logger.llm_usage(),
                {"llm_calls": 1, "input_tokens": 0, "output_tokens": 0},
            )

    def test_不在运行里调_llm_不会崩(self):
        """有人单独调 llm_node 的函数时不该炸 —— 丢掉这次计数就行。"""
        logger.note_llm_call(input_tokens=999)

        self.assertEqual(logger.llm_usage()["llm_calls"], 0)

    def test_简写的可选参数是_keyword_only(self):
        """这条是真出过事才写的。

        node_xxx 的第二个位置参数是 task_id，task_xxx 的第二个是 message。
        两个都是字符串，写错不报错 —— 只是 task_id 那一格变成了一句话：

            logger.node_completed(name, "交给 extract")
            -> task=交给 extract | node=extract | ...   真正的 task_id 没了

        V2 接进 payroll_agent 的第一版就是这么写的，跑起来一点毛病没有，
        是集成测试把 {'交给 validate', '交给 route', ...} 当成 task_id
        读出来才发现的。加 * 之后这种调用当场 TypeError。
        """
        with self.assertRaises(TypeError):
            logger.node_completed("parse", "交给 extract")

        with self.assertRaises(TypeError):
            logger.task_completed("t-001", "跑完了")

    # ---- 9. 出口降级 ----

    def test_文件出口建不出来时只剩控制台也不崩(self):
        """日志模块坏掉，不该把工资流程一起带崩 —— 但也不该悄悄假装没事。"""
        blocker = Path(self._tmp.name) / "占位文件"
        blocker.write_text("这是一个文件，不是目录", encoding="utf-8")
        logger.LOG_FILE = blocker / "agent.log"   # 父目录是个文件，建不出来
        _drop_handlers()

        logger.node_started("parse", task_id="t-005")     # 不抛异常

        log = logging.getLogger(logger.LOGGER_NAME)
        self.assertEqual(len(log.handlers), 1, "只剩控制台那一个")
        self.assertIn("node=parse", self._fake_stdout.getvalue())

    def test_文件按_utf8_写_中文能原样读回来(self):
        logger.task_started("t-006", message="中文消息：文件读不进来")

        text = logger.LOG_FILE.read_text(encoding="utf-8")
        self.assertIn("中文消息：文件读不进来", text)

    # ---- 补充：级别和 describe ----

    def test_失败用_ERROR_级别_其余用_INFO(self):
        # 先把两个正式出口装好，再加自己的捕获 handler。
        # 顺序反过来的话，get_logger() 会看到「已经有 handler」直接返回，
        # 那两个正式出口就装不上了 —— 这是 get_logger 那条 if 的代价。
        logger.get_logger()
        captured = io.StringIO()
        handler = logging.StreamHandler(captured)
        handler.setFormatter(logging.Formatter("%(levelname)s %(event)s"))
        logging.getLogger(logger.LOGGER_NAME).addHandler(handler)

        logger.task_started("t-004")
        logger.node_completed("parse", task_id="t-004")
        logger.node_failed("parse", task_id="t-004")
        logger.task_failed("t-004")

        self.assertEqual(
            [line for line in captured.getvalue().splitlines() if line.strip()],
            [
                "INFO task_started",
                "INFO node_completed",
                "ERROR node_failed",
                "ERROR task_failed",
            ],
        )

    def test_describe_只报形状不报内容(self):
        self.assertEqual(logger.describe(None), "None")
        self.assertEqual(logger.describe([]), "list(0)")
        self.assertEqual(logger.describe({}), "{} 空")
        self.assertEqual(logger.describe({"a": []}), "{a: list(0)}")
        self.assertEqual(logger.describe([{"x": 1}]), "list(dict(1键) x1)")
        self.assertEqual(logger.describe("张三"), "str(2)")

    # ---- 节点轨迹（第二个出口 logs/trace.jsonl）----
    #
    # 这一组是**能落盘的东西**里最危险的一条路：它收的是一个完整的 State，
    # 而 State 里装着姓名、身份证号、工资。所以这里断言的重点不是
    # 「格式对不对」，是「有没有一个值漏出去」。

    def test_shape_of_只报形状不报内容(self):
        self.assertEqual(
            logger.shape_of(
                {"raw_rows": [{"姓名": "张三"}], "file_path": "data/工资表.xlsx"}
            ),
            {"raw_rows": "list(1)", "file_path": "str(13)"},
        )
        self.assertEqual(logger.shape_of(None), None)
        self.assertEqual(logger.shape_of(12000), "number")
        self.assertEqual(logger.shape_of("张三"), "str(2)")
        self.assertEqual(logger.shape_of(b"bytes"), "bytes")

        # bool 挡在数字前面：parse_error 是 True 还是 1，是这个字段的全部信息
        self.assertEqual(logger.shape_of(True), "bool")

        # 字典往下钻一层，和 describe 同一个尺度。
        # 「规则库里有几条规则」必须看得出来 —— 只看 dict(2键) 的话，
        # 分不出「这次没规则可用」和「有规则但全是历史记录」
        self.assertEqual(
            logger.shape_of({"memory_context": {"approved_rules": [], "historical_rules": [{}]}}),
            {"memory_context": {"approved_rules": "list(0)", "historical_rules": "list(1)"}},
        )

    def test_轨迹一行一个节点(self):
        logger.node_trace("parse", {"raw_rows": [1, 2, 3]}, changed=["raw_rows"], next_action="extract")

        records = _trace_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["node"], "parse")
        self.assertEqual(records[0]["status"], logger.SUCCESS)
        self.assertEqual(records[0]["input"], {"raw_rows": "list(3)"})
        self.assertEqual(records[0]["changed"], ["raw_rows"])
        self.assertEqual(records[0]["next_action"], "extract")
        # 成功就没有错误类型。留个 None，不留个 ""，否则读的人分不清
        # 「这次没出错」和「出了错但类型忘了填」
        self.assertIsNone(records[0]["error_type"])

    def test_轨迹里一个业务值都没有(self):
        state = {
            "file_path": "data/XX公司_工资表.xlsx",
            "raw_rows": [{"姓名": "张三", "身份证号": "110101199001011234", "实发工资": 12000}],
            "halt_reason": "张三 的银行卡 6222021234567890123 对不上",
            "parse_error": False,
            "empty_thing": [],
            "nothing": None,
        }
        logger.node_trace("parse", state)

        text = logger.TRACE_FILE.read_text(encoding="utf-8")

        # 值一个字都不能出现
        for secret in ("张三", "XX公司", "110101199001011234", "12000", "6222021234567890123"):
            self.assertNotIn(secret, text, f"「{secret}」不该出现在轨迹里")

        # 而键名和形状必须都在 —— 不然这个文件就没用了。
        # 「只删干净」是能过的，但那样它就不回答任何问题
        record = _trace_records()[0]
        self.assertEqual(
            sorted(record["input"].keys()),
            ["file_path", "halt_reason", "parse_error", "raw_rows"],
        )
        self.assertEqual(record["input"]["raw_rows"], "list(1)")
        self.assertEqual(record["input"]["parse_error"], "bool")

        # 空的那两格**不报**。十几个键里大部分是空的，
        # 全报出来是一面墙，真正有信息的那几个被淹掉
        self.assertNotIn("empty_thing", text)
        self.assertNotIn("nothing", text)

    def test_失败那一行_changed_是_None_不是空列表(self):
        logger.node_trace(
            "calculate",
            {"extracted_rows": [1]},
            status=logger.FAILED,
            error=ValueError("张三 的工资 12,000.00 元算不出来"),
        )

        record = _trace_records()[0]
        self.assertEqual(record["status"], logger.FAILED)
        # 只记类型名。异常消息里那句「张三 的工资 12,000.00」一个字都不进文件
        self.assertEqual(record["error_type"], "ValueError")
        self.assertNotIn("张三", logger.TRACE_FILE.read_text(encoding="utf-8"))
        self.assertNotIn("12,000.00", logger.TRACE_FILE.read_text(encoding="utf-8"))

        # **节点没跑完**（没有局部更新）和**跑完了一个键都没改**是两回事。
        # 都写成 [] 的话，看轨迹的人分不出「这个节点干了件事但没改 State」
        # 和「这个节点炸了」—— 而后者恰恰是他要找的那一行
        self.assertIsNone(record["changed"])
        self.assertIsNone(record["next_action"])

    def test_轨迹的_task_id_从执行流上取(self):
        with logger.task_scope("t-trace"):
            logger.node_trace("parse", {})
        logger.node_trace("extract", {})   # 不在任何运行里

        first, second = _trace_records()
        self.assertEqual(first["task_id"], "t-trace")
        self.assertEqual(second["task_id"], logger.EMPTY)

    def test_error_where_只记代码坐标不记消息(self):
        try:
            _boom_with_business_data()
        except ValueError as exc:
            where = logger.error_where(exc)

        # 形式是「文件:行号 函数名」。行号不写死（改上面的代码就变了），
        # 但**文件名和函数名要能对上** —— 靠这两个就能直接跳到出事那行
        self.assertTrue(where.startswith("test_logger.py:"), where)
        self.assertTrue(where.endswith(" _boom_with_business_data"), where)

        # 这一格的**全部意义**在这里：异常消息里有姓名、金额、卡号，
        # 而 error_where 一个字都不带出来
        for secret in ("张三", "12,000.00", "6222021234567890123"):
            self.assertNotIn(secret, where)

        # 凭空造的异常没有 traceback。给 None，不给一个假的坐标
        self.assertIsNone(logger.error_where(ValueError("没被 raise 过")))
        self.assertIsNone(logger.error_where(None))

    def test_失败的轨迹行里有哪一步炸的(self):
        try:
            _boom_with_business_data()
        except ValueError as exc:
            logger.node_trace("calculate", {"extracted_rows": [1]}, status=logger.FAILED, error=exc)

        record = _trace_records()[0]
        self.assertEqual(record["error_type"], "ValueError")
        self.assertTrue(record["error_at"].endswith(" _boom_with_business_data"))

    # ---- 文件滚不滚 ----

    def test_没超上限就不滚(self):
        logger.node_started("parse", task_id="t-roll")
        size = logger.LOG_FILE.stat().st_size

        with _tiny_limit(size + 1):     # 上限比现有内容大一点点
            logger.rotate_if_needed()

        self.assertTrue(logger.LOG_FILE.exists(), "没超上限，原文件该原封不动")
        self.assertFalse(
            logger.LOG_FILE.with_name(logger.LOG_FILE.name + logger.ROTATED_SUFFIX).exists()
        )

    def test_超上限就滚一代_而且只留一代(self):
        logger.node_started("parse", task_id="t-roll")
        first = logger.LOG_FILE.read_text(encoding="utf-8")
        rotated = logger.LOG_FILE.with_name(logger.LOG_FILE.name + logger.ROTATED_SUFFIX)

        with _tiny_limit():
            logger.rotate_if_needed()      # 第一次：滚出 .1
            logger.node_started("extract", task_id="t-roll")
            _drop_handlers()               # 让文件出口重新打开（见 get_logger）
            logger.node_started("calculate", task_id="t-roll")

            logger.rotate_if_needed()      # 第二次：覆盖 .1

        # 只留一代 —— .2 这种东西不该出现。
        # 留下它等于「换个地方删旧文件」，而磁盘占用就不再是恒定的了
        self.assertFalse(rotated.with_name(rotated.name + ".1").exists())
        self.assertIn("node=calculate", rotated.read_text(encoding="utf-8"))
        self.assertNotIn("node=parse", rotated.read_text(encoding="utf-8"))
        self.assertNotEqual(rotated.read_text(encoding="utf-8"), first)

    def test_滚过之后日志还得继续写进新文件(self):
        """这条盯的是滚日志最容易出的那个 bug。

        get_logger() 看到 handlers 非空就直接返回。滚完要是忘了重装，
        出口就一直挂在一个**已经不存在的文件名**上 ——
        现象是：滚了一次之后，agent.log 静悄悄地再也没长过，
        而且没有任何报错。
        """
        logger.node_started("parse", task_id="t-roll")

        with _tiny_limit():
            logger.rotate_if_needed()
            logger.node_started("extract", task_id="t-roll")

        self.assertTrue(logger.LOG_FILE.exists(), "滚完该有新文件")
        self.assertIn("node=extract", logger.LOG_FILE.read_text(encoding="utf-8"))

    def test_两个文件一起滚(self):
        """两个都要滚，而且**一次调用就都滚掉** —— 调用方只管在运行前喊一声。

        这里顺带钉住一个新文件 handler 的副作用：agent.log 滚完之后
        那个路径**立刻又有文件了**（空文件）—— handler 被重装时把它建了出来。
        轨迹文件没有 handler，所以它要等到下一次 node_trace 才重新出现。
        两种表现都在这里写清楚，免得以后有人看见空的 agent.log 以为滚失败了。
        """
        logger.node_started("parse", task_id="t-roll")
        logger.node_trace("parse", {"raw_rows": [1]})

        with _tiny_limit():
            logger.rotate_if_needed()

        for path in (logger.LOG_FILE, logger.TRACE_FILE):
            rotated = path.with_name(path.name + logger.ROTATED_SUFFIX)
            self.assertTrue(rotated.exists(), f"{path.name} 该滚出 .1")
            self.assertIn("parse", rotated.read_text(encoding="utf-8"))

        self.assertEqual(logger.LOG_FILE.read_text(encoding="utf-8"), "", "新的一代从空开始")
        self.assertFalse(logger.TRACE_FILE.exists(), "轨迹要等下一次 node_trace 才重建")

    def test_滚不动时不崩_只记一条_WARNING(self):
        """滚不动就照原样往老文件里写 —— 不崩，也不装作没发生。"""
        logger.node_started("parse", task_id="t-roll")

        # 把 .1 的位置先占成一个**目录**，os.replace 就一定会失败
        rotated = logger.LOG_FILE.with_name(logger.LOG_FILE.name + logger.ROTATED_SUFFIX)
        rotated.mkdir(parents=True)
        _drop_handlers()

        with _tiny_limit():
            logger.rotate_if_needed()      # 不抛异常

        log_text = _log_text()
        self.assertIn(f"event={logger.ROTATE_FAILED}", log_text)
        self.assertIn("agent.log 滚不动", log_text)
        # message 里是文件名，不是完整路径 —— 路径里有这台机器的目录结构
        self.assertNotIn(str(self._tmp.name), log_text)

        # 老文件还在，新的这句写进去了
        self.assertTrue(logger.LOG_FILE.exists())
        self.assertIn("node=parse", logger.LOG_FILE.read_text(encoding="utf-8"))

    def test_轨迹写不进去时不崩_只记一条_WARNING(self):
        """这条比日志那个降级用例更要紧。

        node_trace 是在**节点跑的路径上**调用的。它要是抛异常，
        traced 那个 except 会抓住，记成「节点抛出异常」，然后整个流程停住 ——
        也就是说：磁盘满了会让一次工资计算**失败**。
        为了记一行「这个节点长什么样」而让这个节点真的失败，是拿反了。
        """
        blocker = Path(self._tmp.name) / "占位文件"
        blocker.write_text("这是一个文件，不是目录", encoding="utf-8")
        logger.TRACE_FILE = blocker / "trace.jsonl"   # 父目录是个文件，建不出来

        record = logger.node_trace("parse", {"raw_rows": [1]})   # 不抛异常

        # 不崩，但也不装作没发生 —— 往 agent.log 记一条：
        # 事件名是 trace_write_failed，错误只记类型名（和别处一个口径，
        # OSError 的消息里带着那个路径，一样不落盘）。
        # 不记的话，轨迹文件静悄悄地一行都不长，而没有任何人会去看它
        log_text = _log_text()
        self.assertIn(f"event={logger.TRACE_WRITE_FAILED}", log_text)
        # 具体的类型名（FileExistsError 是 OSError 的子类），不是笼统的 OSError。
        # 这里要的正是「哪个类型」，因为**消息一个字都不记**
        self.assertIn("error=FileExistsError", log_text)
        self.assertNotIn("占位文件", log_text, "错误消息里的路径不该落盘")

        # 写不进去也照样把这一行还给调用方：内存里那份轨迹不受磁盘影响。
        # 不然「磁盘坏了」会变成「调试信息也一起没了」—— 而这两件事无关
        self.assertEqual(record["node"], "parse")
        self.assertEqual(record["input"], {"raw_rows": "list(1)"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
