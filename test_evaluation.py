"""test_evaluation.py —— evaluation.py 的最小验证。

跑法（在 AI-Accounting-Agent 目录下）：

    python -m unittest test_evaluation

或者：

    python test_evaluation.py

用标准库 unittest，不引 pytest —— 和 test_logger.py 同一条理由。

============================================================
这份文件里最重要的是哪几条
============================================================

大部分用例在验「算得对不对」，那是基础。真正要守住的是另外两条：

    1. test_没有数据时不能装作没问题
       evaluation 最容易骗自己的地方，就是「没有数据」和「没有问题」
       在报告上长得一模一样。这条用例把这个坑钉死。

    2. test_老行缺的字段按不知道算_不按零算
       老行没有 llm_calls。要是补成 0，一条「当年还没记用量」的记录
       会被算成「这次没调 LLM」—— 报告上看着一切正常，
       而那个结论是凭空造出来的。

用例里的字段名、金额、任务 ID 全是编的，不用任何真实员工信息。
"""

import json
import tempfile
import unittest
from pathlib import Path

import evaluation
import run_summary


def _record(**overrides) -> dict:
    """造一条 v2 的运行记录，只覆盖你关心的那几个字段。

    默认值挑的是「一切正常」的样子：成功了、没问模型、没走到复核。
    每条用例只改它要测的那一格 —— 这样一条用例红的时候，
    原因只有一个，不用在一堆默认值里找。
    """
    record = {
        "schema_version": run_summary.SCHEMA_VERSION,
        "task_id": "abcdef123456",
        "start_time": "2026-01-01 10:00:00",
        "end_time": "2026-01-01 10:00:01",
        "status": "success",
        "failed_node": None,
        "error_type": None,
        "fail_category": None,
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "review": None,
    }
    record.update(overrides)
    return record


def _failed(**overrides) -> dict:
    """一条典型的失败记录。"""
    base = {"status": "failed"}
    base.update(overrides)
    return _record(**base)


def _review(approved=(), rejected=(), undecided=()) -> dict:
    return run_summary.review_record(
        approved=list(approved),
        rejected=list(rejected),
        undecided=list(undecided),
        reviewer="张三",
    )


def _ids(findings: list) -> set:
    return {finding["id"] for finding in findings}


# ============================================================
# 置信度
# ============================================================


class ConfidenceTest(unittest.TestCase):
    """置信度是个**样本量**指标，不是概率。

    这里把它的三个性质钉死：样本太少就压低、封顶不给 1.0、能手算。
    谁哪天把它改成「跑个统计检验」，这几条会红 —— 那时候该先想清楚
    「20 条数据算出来的 p 值，值不值得让人相信」。
    """

    def test_样本太少时一律压低(self):
        for total in range(0, evaluation.MIN_SAMPLE):
            self.assertEqual(
                evaluation._confidence(total, total),
                0.3,
                f"{total} 条就该给 0.3，不该因为「全都中招」就往上涨",
            )

    def test_样本够了才按占比算(self):
        # 5 条里 1 条 -> 0.4 + 0.5 * 0.2 = 0.5
        self.assertEqual(evaluation._confidence(1, 5), 0.5)
        # 10 条里 5 条 -> 0.4 + 0.5 * 0.5 = 0.65
        self.assertEqual(evaluation._confidence(5, 10), 0.65)

    def test_封顶不给满分(self):
        """样本推不出「一定」。给 1.0 等于说「我已经确认了」——
        这不是从记录里能得出的结论。"""
        self.assertLess(evaluation._confidence(100, 100), 1.0)
        self.assertEqual(evaluation._confidence(100, 100), 0.9)


# ============================================================
# 读窗口
# ============================================================


class ReadRunsTest(unittest.TestCase):
    def setUp(self):
        self._dir = Path(tempfile.mkdtemp())
        self._path = self._dir / "summary.jsonl"

    def test_只读最近几条_而且是最新的那几条(self):
        lines = [
            json.dumps({"task_id": f"t{index}", "status": "success"})
            for index in range(10)
        ]
        self._path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        records = run_summary.read_runs(limit=3, path=self._path)

        # 从旧到新，而且是最后三条 —— 不是前三条
        self.assertEqual([r["task_id"] for r in records], ["t7", "t8", "t9"])

    def test_老行缺的字段按不知道算_不按零算(self):
        """这一条是整份文件里最要紧的一条。

        v1 的行没有 llm_calls。要是补成 0，
        「当年还没记用量」就会被读成「这次压根没调模型」——
        后者是一个**结论**，而它是凭空造出来的。

        补成 None 之后，「没调」和「不知道调没调」分得开。
        """
        old = {"task_id": "old", "status": "success"}  # v1 的行，没几个字段
        self._path.write_text(json.dumps(old) + "\n", encoding="utf-8")

        record = run_summary.read_runs(path=self._path)[0]

        self.assertIsNone(record["llm_calls"], "缺的字段该补 None")
        self.assertIsNot(record["llm_calls"], 0, "补成 0 就是把「不知道」写成了结论")
        self.assertIsNone(record["schema_version"], "老行没有版本号，拿到 None 才知道它是 v1")
        # 记录里原本就有的键不能被抹掉
        self.assertEqual(record["task_id"], "old")

    def test_坏行跳过_但会留下一条日志(self):
        """半截 JSON（进程被掐在写一半的时候）不该让整个文件读不出来。

        但也不能**静默**跳过 —— 少一个样本，统计就少一分，
        而读报告的人看不出来，只会以为样本本来就那么多。
        """
        good = json.dumps({"task_id": "good", "status": "success"})
        self._path.write_text(
            good + "\n" + '{"task_id": "half' + "\n" + good + "\n", encoding="utf-8"
        )

        records = run_summary.read_runs(path=self._path)

        self.assertEqual(len(records), 2, "两条好的该读出来")
        self.assertEqual({r["task_id"] for r in records}, {"good"})

    def test_文件不存在就是空的(self):
        self.assertEqual(run_summary.read_runs(path=self._dir / "没有这个文件"), [])


# ============================================================
# 失败归类
# ============================================================


class ClassifyTest(unittest.TestCase):
    def test_成功了就没有分类(self):
        self.assertIsNone(run_summary.classify())

    def test_文件打不开时归输入_不归业务停(self):
        """两种信号同时成立时，选哪个是有讲究的。

        broken-01（文件不存在）里，parse_error 和 halt_reason **都有值**。
        归成 business_halt 的话，看报告的人会去找业务方问口径 ——
        而其实该做的是去问客户「你这张表为什么打不开」。
        """
        category = run_summary.classify(
            parse_error="文件不存在", halt_reason="流程在 exception 节点终止"
        )
        self.assertEqual(category, run_summary.CAT_INPUT)

    def test_人工否决停下归业务停(self):
        category = run_summary.classify(halt_reason="人工否决了 餐补")
        self.assertEqual(category, run_summary.CAT_BUSINESS_HALT)

    def test_模型那一步炸了归外部(self):
        category = run_summary.classify(failed_node=run_summary.LLM_NODE)
        self.assertEqual(category, run_summary.CAT_LLM_ERROR)

    def test_其余归我们自己(self):
        category = run_summary.classify(failed_node="calculate")
        self.assertEqual(category, run_summary.CAT_AGENT_ERROR)

    def test_分类值都在约定的集合里(self):
        """分类是给程序按值匹配的。多出一个没人认识的值，
        报告里就会多出一堆「分类是新的，还没有对应的处置办法」。"""
        produced = {
            run_summary.classify(parse_error="x"),
            run_summary.classify(halt_reason="x"),
            run_summary.classify(failed_node=run_summary.LLM_NODE),
            run_summary.classify(failed_node="calculate"),
        }
        self.assertTrue(produced.issubset(set(run_summary.FAIL_CATEGORIES)))
        # 每个分类都得有人知道该怎么办，不然报告只能写「不知道找谁」
        self.assertEqual(set(run_summary.FAIL_CATEGORIES), set(evaluation._ACTION))


# ============================================================
# 复核记录
# ============================================================


class ReviewRecordTest(unittest.TestCase):
    def test_候选数是三个名单加起来(self):
        record = _review(approved=["餐补"], rejected=["神秘补贴"], undecided=["宿舍费"])
        self.assertEqual(record["candidates"], 3)

    def test_一个候选都没有也算走到过复核(self):
        record = _review()
        self.assertEqual(record["candidates"], 0)
        self.assertEqual(record["approved"], [])

    def test_只收字段名(self):
        """review 里不能出现金额、姓名这些东西 —— 和 logger 那三道闸门一个口径。"""
        record = _review(approved=["餐补"])
        self.assertEqual(
            sorted(record.keys()),
            ["approved", "candidates", "rejected", "reviewer", "undecided"],
        )


# ============================================================
# A. 失败分析
# ============================================================


class FailureAnalysisTest(unittest.TestCase):
    def test_没有失败就没有发现(self):
        records = [_record() for _ in range(10)]
        self.assertEqual(evaluation.analyze_failures(records), [])

    def test_同类失败扎堆时报出来(self):
        records = [_record() for _ in range(7)]
        records += [_failed(fail_category=run_summary.CAT_AGENT_ERROR) for _ in range(3)]

        findings = evaluation.analyze_failures(records)

        self.assertIn("failure.dominant_category", _ids(findings))
        top = [f for f in findings if f["id"] == "failure.dominant_category"][0]
        self.assertEqual(top["affected_task_count"], 3)
        self.assertIn(run_summary.CAT_AGENT_ERROR, top["detected_problem"])

    def test_孤零零一次失败不报_一两条例不出规律(self):
        records = [_record() for _ in range(9)]
        records.append(_failed(fail_category=run_summary.CAT_AGENT_ERROR))

        self.assertEqual(evaluation.analyze_failures(records), [])

    def test_归不了类时要明说_不能读成没问题(self):
        """老行没有 fail_category。不说的话，报告上就是「没有失败发现」，
        读的人会得出「失败没有规律」—— 而那是一个**没有数据支撑的结论**。
        """
        records = [_record() for _ in range(5)]
        records += [_failed() for _ in range(3)]  # 三条老行，没有 fail_category

        findings = evaluation.analyze_failures(records)

        self.assertIn("failure.unclassified", _ids(findings))

    def test_失败扎堆在同一个节点也要报(self):
        records = [_record() for _ in range(7)]
        records += [
            _failed(fail_category=run_summary.CAT_AGENT_ERROR, failed_node="calculate")
            for _ in range(3)
        ]

        self.assertIn("failure.failed_node_hotspot", _ids(evaluation.analyze_failures(records)))


# ============================================================
# B. LLM 用量
# ============================================================


class LlmUsageTest(unittest.TestCase):
    def test_调了却没拿到用量要报(self):
        """calls=1 而 tokens=0 —— 这可能是调用失败，也可能是服务端没返回用量。

        两种情况现在长得一模一样，分不开就没法知道
        是不是在为一个坏掉的接口付费。
        """
        records = [
            _record(llm_calls=1, input_tokens=0, output_tokens=0) for _ in range(3)
        ]
        records += [_record() for _ in range(5)]

        self.assertIn("llm_usage.usage_missing", _ids(evaluation.analyze_llm_usage(records)))

    def test_压根没调模型不算没拿到用量(self):
        """calls=0 是「表全认识，不用问」，是好事，不该报成问题。

        它的 token 也是 0，和「调了但失败了」在数字上分不开 ——
        分得开靠的是 llm_calls，不是 token。
        """
        records = [_record(llm_calls=0) for _ in range(8)]
        self.assertEqual(evaluation.analyze_llm_usage(records), [])

    def test_输入token偏高要报(self):
        records = [
            _record(llm_calls=1, input_tokens=9000, output_tokens=60) for _ in range(4)
        ]
        records += [_record() for _ in range(4)]

        self.assertIn("llm_usage.input_tokens_high", _ids(evaluation.analyze_llm_usage(records)))

    def test_大小正常就不报(self):
        records = [
            _record(llm_calls=1, input_tokens=800, output_tokens=60) for _ in range(6)
        ]
        records += [_record() for _ in range(2)]

        self.assertEqual(evaluation.analyze_llm_usage(records), [])

    def test_每次都问模型说明记忆没起作用(self):
        """memory 存在的前提是「同一个字段第二次不用再问」。
        一次都没省下来，要么规则库是空的，要么一直没命中。
        """
        records = [
            _record(llm_calls=1, input_tokens=800, output_tokens=60) for _ in range(6)
        ]

        self.assertIn(
            "llm_usage.memory_never_hit", _ids(evaluation.analyze_llm_usage(records))
        )

    def test_全是老行时一个字都不说(self):
        """老行没有 llm_calls。全都是老行 = 没有数据，
        不是「每次都调了模型」也不是「一次都没调」。"""
        records = [{"task_id": f"old{index}", "status": "success"} for index in range(10)]
        normalized = [run_summary.normalize(r) for r in records]

        self.assertEqual(evaluation.analyze_llm_usage(normalized), [])


# ============================================================
# C. 记忆建议
# ============================================================


class MemorySignalTest(unittest.TestCase):
    def test_没有数据时不能装作没问题(self):
        """这一条是最要紧的。

        没有 review 数据时，「被反复否决的字段」那个循环一次都不会进，
        报告上就是干干净净的一张白纸 —— 看起来像「记忆没问题」。

        但真相是「**没数据**」。两句话指向的动作完全相反：
            没问题 -> 什么都不用做
            没数据 -> 去把数据补上，再看
        """
        records = [_record() for _ in range(10)]  # 全是 review=None

        findings = evaluation.analyze_memory_signals(records)

        self.assertIn("memory.no_review_data", _ids(findings))
        self.assertTrue(
            any("根本没有数据" in f["detected_problem"] for f in findings),
            "不能只说「没问题」，要说清楚是「没数据」",
        )

    def test_反复被否的字段要报(self):
        records = [_record(review=_review(rejected=["神秘补贴"])) for _ in range(3)]
        records += [_record() for _ in range(5)]

        findings = evaluation.analyze_memory_signals(records)

        ids = _ids(findings)
        self.assertTrue(any("repeated_rejection" in i for i in ids))
        # 建议里必须提到「补进 FIELD_MAP 就等于不再过人工确认」这层代价
        target = [f for f in findings if "repeated_rejection" in f["id"]][0]
        self.assertIn("人工确认", target["suggested_action"])

    def test_只否过一次不算规律(self):
        records = [_record(review=_review(rejected=["神秘补贴"]))]
        records += [_record() for _ in range(9)]

        self.assertEqual(evaluation.analyze_memory_signals(records), [])

    def test_一直没人表态的字段要报(self):
        """这类比「被否决」更危险：被否决至少有人看过并拍了板，
        这类是压根没人看 —— 而当前实现里人不表态流程照走，
        用到这个字段的人会被静默跳过。没报错，工资少算了。
        """
        records = [_record(review=_review(undecided=["餐补"])) for _ in range(2)]
        records += [_record() for _ in range(6)]

        self.assertIn(
            "memory.stuck_undecided::餐补", _ids(evaluation.analyze_memory_signals(records))
        )


# ============================================================
# 报告整体
# ============================================================


class ReportTest(unittest.TestCase):
    def test_发现按影响面从大到小排(self):
        """排序在合并之后做，不是各能力自己做 ——
        各自排序再合并，顺序又乱了，等于没排。"""
        records = [_record() for _ in range(20)]
        records += [_failed(fail_category=run_summary.CAT_AGENT_ERROR) for _ in range(5)]
        records += [_record(review=_review(rejected=["神秘补贴"])) for _ in range(2)]

        findings = evaluation.build_report(records)["findings"]
        counts = [f["affected_task_count"] for f in findings]

        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_每条发现都长成约定的样子(self):
        records = [_failed(fail_category=run_summary.CAT_AGENT_ERROR) for _ in range(6)]
        report = evaluation.build_report(records)

        for finding in report["findings"]:
            self.assertEqual(
                sorted(finding.keys()),
                [
                    "affected_task_count",
                    "capability",
                    "confidence",
                    "detected_problem",
                    "id",
                    "suggested_action",
                ],
            )
            self.assertLessEqual(finding["confidence"], 0.9)

    def test_窗口里记了有多少数据能用(self):
        """结论的可信度由窗口决定 —— 读报告的人应该先看这里。"""
        records = [_record() for _ in range(8)]
        records += [_failed(fail_category=run_summary.CAT_INPUT)]
        records.append({"task_id": "old", "status": "success"})  # 一条老行

        window = evaluation.build_report([run_summary.normalize(r) for r in records])["window"]

        self.assertEqual(window["analyzed"], 10)
        self.assertEqual(window["success"], 9)
        self.assertEqual(window["failed"], 1)
        self.assertEqual(window["legacy_rows"], 1)
        self.assertEqual(window["with_fail_category"], 1)
        self.assertEqual(window["with_review"], 0)

    def test_样本不够时报告自己要说出来(self):
        records = [_record() for _ in range(2)]
        report = evaluation.build_report(records)

        self.assertTrue(
            any("只是线索" in note for note in report["notes"]),
            "样本这么少，报告必须自己承认",
        )

    def test_报告能直接json序列化(self):
        """报告要落成文件、要被人转发。datetime 之类的东西漏进去就写不出来。"""
        records = [_failed(fail_category=run_summary.CAT_LLM_ERROR, failed_node="llm_rule_analysis")]
        text = json.dumps(evaluation.build_report(records), ensure_ascii=False)
        self.assertIn("analysis_time", text)


# ============================================================
# 三条不能越的线
# ============================================================


def _imported_modules(module) -> set:
    """一个模块真正 import 了谁。

    用 ast 读 import 语句，**不在源码文本里搜** —— 那是第一版写法，
    它匹配到了文档字符串里「本模块不 import calculation」这句话本身，
    于是「声明自己不 import」反而让用例红了。

    文本搜不出「代码」和「关于代码的话」的区别，语法树分得清。
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)

    return names


class BoundariesTest(unittest.TestCase):
    def test_不import_那三个会改数据的模块(self):
        """evaluation 只读。

        它要是能顺手把规则改了，Human-in-the-loop 那道闸门就白设了 ——
        「发现问题」和「拍板」合成一个角色，正是这套设计一开始要拆开的。
        """
        forbidden = {"calculation", "validation", "memory", "payroll_agent"}

        self.assertEqual(
            _imported_modules(evaluation) & forbidden,
            set(),
            "evaluation 是只读的，不该 import 这几个会改数据的模块",
        )

    def test_它确实import了_summary_和_logger(self):
        """上一条只说了「不许 import 谁」，这条说「必须 import 谁」。

        少了它，把 evaluation 改成不 import 任何东西（上一条照样绿）
        也看不出来。
        """
        imports = _imported_modules(evaluation)
        self.assertIn("run_summary", imports)
        self.assertIn("logger", imports)

    def test_建议是句子不是命令(self):
        """suggested_action 是给人看的，不是待执行的。

        报告里写着「建议给某个字段加别名」，这件事仍然要人去做。
        自动应用等于把「发现问题的」和「拍板的」合成一个角色。
        """
        records = [_failed(fail_category=run_summary.CAT_AGENT_ERROR) for _ in range(6)]
        findings = evaluation.build_report(records)["findings"]

        self.assertTrue(findings)
        for finding in findings:
            action = finding["suggested_action"]
            self.assertIsInstance(action, str)
            self.assertTrue(action.strip(), "建议不能是空的")


if __name__ == "__main__":
    unittest.main(verbosity=2)
