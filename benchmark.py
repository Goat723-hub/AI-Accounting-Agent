"""benchmark.py —— 跑评测集，出量化指标。

跑法（在 AI-Accounting-Agent 目录下）：

    python benchmark.py                    # 只出总表
    python benchmark.py -v                 # 每条用例都打一行
    python benchmark.py --json out.json    # 顺手把指标落一份 json

    # 把 27 次运行的记录留下来，喂给 evaluation：
    python benchmark.py --keep-records /tmp/demo
    python evaluation.py --file /tmp/demo/summary.jsonl

它和 evaluation.py 是两件事，别混：

    benchmark.py      **主动跑**一遍评测集，回答「这套东西有多准」
                      输入是 eval_cases.json（带标准答案），
                      产出是准确率/成功率 —— 这是能写进简历的那部分。

    evaluation.py     **被动读**已经攒下来的运行记录，回答「哪里有毛病」
                      输入是 runs/summary.jsonl（没有标准答案），
                      产出是 finding 报告。

    一个是考试，一个是体检。共用同一批运行数据，但问的问题不一样。

============================================================
关于「期望值」这件事，有一条必须守住的线
============================================================

表里该写多少「实发工资」，这里**自己算一遍**，绝不 import calculation.py。

    拿被测代码去生成标准答案，等于让考生自己出卷子 —— 它算错了，
    期望值跟着错，两边一起错，测试全绿，而结果是错的。

    所以下面这份 _expected_net_pay 是照着业务口径**重新写的一份**。
    它和 calculation.py 的公式必须一致，但它是独立的一份 ——
    哪天有人在 calculation.py 里把社保比例从 22.5% 改成 20%，
    这里不会跟着改，于是 happy-01 立刻变红。**这正是它存在的意义。**

    代价：口径真变了的时候要改两处。这个代价是故意付的。

============================================================
关于 LLM
============================================================

new_field 那三条会走到 llm_rule_analysis，也就是真的会去调模型。
benchmark 里换成了固定桩（跟 test_agent_logging.py 一个路子）：
**桩测的是「LLM 提议 -> 存成 pending -> 问人 -> 批准后用上」这条链路的正确性，
不是模型质量。** 报告里会把这件事写明，不混进准确率里充数。
"""

import contextlib
import io
import json
import logging
import sys
import tempfile
from pathlib import Path

from openpyxl import Workbook

import extraction
import llm_node
import logger
import memory
import payroll_agent
import run_summary


PROJECT_DIR = Path(__file__).parent
CASES_FILE = PROJECT_DIR / "eval_cases.json"

# 表里代表「实发工资」的那一列，可能是这几种写法
NET_PAY_ALIASES = ("实发工资", "实发", "到手工资")

# 故意写错时往上加多少 —— 加 1000 块，保证一定对不上
WRONG_NET_PAY_OFFSET = 1000


# ============================================================
# 独立实现的计算口径（oracle）
# ============================================================

SOCIAL_RATE = 0.225
TAX_THRESHOLD = 5000.0
TAX_BRACKETS = (
    (3000.0, 0.03, 0.0),
    (12000.0, 0.10, 210.0),
    (25000.0, 0.20, 1410.0),
    (35000.0, 0.25, 2660.0),
    (55000.0, 0.30, 4410.0),
    (80000.0, 0.35, 7160.0),
    (None, 0.45, 15160.0),
)


def expected_net_pay(base_salary: float, bonus: float, deduction: float) -> float:
    """照业务口径重算一遍实发工资。**不是 import 来的，是重写的。**

    应发       = 基本 + 奖金 - 扣款
    社保       = 基本 x 22.5%
    应纳税所得 = max(0, 应发 - 5000 - 社保)
    个税       = 七级超额累进
    实发       = 应发 - 社保 - 个税
    """
    gross = base_salary + bonus - deduction
    social = round(base_salary * SOCIAL_RATE, 2)
    taxable = max(0.0, gross - TAX_THRESHOLD - social)

    for upper, rate, quick in TAX_BRACKETS:
        if upper is None or taxable <= upper:
            tax = round(taxable * rate - quick, 2)
            break

    return round(gross - social - tax, 2)


def _to_money(value) -> float:
    """把单元格的值洗成数字。空值当 0，洗不出来也当 0。

    和 extraction._to_number 口径一致 —— 但这是评测集的**输入处理**，
    不是被测逻辑：它决定的只是「我往这张表里填了多少」。
    """
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value)
    for junk in ("￥", "¥", ",", "，", "元", " ", "　"):
        text = text.replace(junk, "")

    try:
        return float(text)
    except ValueError:
        return 0.0


# ============================================================
# 用例 -> 一张真的 xlsx
# ============================================================


def _field_of(column: str, case: dict):
    """列名对应的业务字段。

    认得出的走 extraction.FIELD_MAP 的别名；
    认不出的（餐补、季度激励这种）只能靠用例自己声明的 extra_as ——
    因为「这列在业务上算什么」正是评测集要给出的标准答案，
    系统不知道，这里也不该替它猜。
    """
    return extraction.LOOKUP.get(column) or case.get("extra_as", {}).get(column)


def _fill_net_pay(case: dict, row: list) -> list:
    """把行里那一格 null / "wrong" 换成数字。

    填的依据是「这张表在业务上真实应得多少钱」——
    也就是所有列都算进去，包括系统还没认出来的那些。
    所以人和系统意见不一致时（比如规则没批），系统算出来的会对不上表 ——
    **那是对的，validation 就该报。**
    """
    columns = case["columns"]

    try:
        index = next(i for i, c in enumerate(columns) if c in NET_PAY_ALIASES)
    except StopIteration:
        return list(row)

    marker = row[index] if index < len(row) else None
    if marker not in (None, "wrong"):
        return list(row)  # 用例显式给了数字，不覆盖

    totals = {"base_salary": 0.0, "bonus": 0.0, "deduction": 0.0}
    for i, column in enumerate(columns):
        field = _field_of(column, case)
        if field in totals:
            totals[field] += _to_money(row[i] if i < len(row) else None)

    net_pay = expected_net_pay(**totals)
    if marker == "wrong":
        net_pay += WRONG_NET_PAY_OFFSET

    filled = list(row)
    while len(filled) <= index:
        filled.append(None)
    filled[index] = net_pay
    return filled


def _write_xlsx(case: dict, path: Path) -> None:
    """按 file_kind 造出这条用例要的那张表。"""
    kind = case.get("file_kind", "xlsx")

    if kind == "missing":
        return  # 故意不建文件

    workbook = Workbook()
    sheet = workbook.active

    if kind == "empty":
        pass  # 一个格子都不写 —— 工作表是空的

    elif kind == "no_header":
        # 有几列数据，但没有一列是眼熟的表头 -> parser 该拒绝，不该硬猜
        for row in case["rows"]:
            sheet.append(row)

    else:
        sheet.append(case["columns"])
        for row in case["rows"]:
            sheet.append(_fill_net_pay(case, row))

    workbook.save(path)
    workbook.close()


# ============================================================
# 规则库
# ============================================================


def _write_memory(case: dict, path: Path) -> None:
    """把用例预置的规则写进一份临时的 memory.json。

    只写用例声明的那几条 —— 不读项目里那份真的，也不受它影响。
    """
    rules = {}

    for field, rule in (case.get("memory") or {}).items():
        rules[field] = {
            "field": field,
            "mapping": rule["mapping"],
            "status": rule["status"],
            "confidence": rule.get("confidence", 0.9),
            "source": "llm",
            "created_at": "2026-01-01T00:00:00",
            "decided_at": "2026-01-01T00:00:00" if rule["status"] != "pending" else None,
            "notes": "",
        }

    if rules:
        path.write_text(
            json.dumps({"version": 1, "rules": rules}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ============================================================
# 沙箱：把三个出口全指向临时目录
# ============================================================


def _drop_handlers() -> None:
    """摘掉 logger 上挂着的 handler。

    handler 挂在 logging 的 logger 对象上，那个对象比一次调用活得久。
    不摘的话，出口换了、写的地方还是老的。
    """
    log = logging.getLogger(logger.LOGGER_NAME)
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()


@contextlib.contextmanager
def _sandbox(workdir: Path, summary_file: Path | None = None):
    """日志、节点轨迹、运行记录、规则库**四个**出口全挪进临时目录。

    少换一个，benchmark 就会往项目根目录里写东西 ——
    runs/summary.jsonl 会混进 27 条假运行记录，
    下次 evaluation.py 读到它们，报告的每个数字都是错的。

    logs/trace.jsonl 是后加的那个，而且它量最大：27 条用例 × 九个节点，
    一次 benchmark 就往项目里倒两百多行没人要的轨迹。

    summary_file 单独一个参数（不跟着 workdir 走），是为了
    --keep-records：27 条用例共用**一份**运行记录，
    但**各用各的**规则库和日志。

        规则库必须分开 —— 前一条批准的规则漏到后一条，
        会让某条用例的结果悄悄变好，而且不报错。

        运行记录可以共用 —— 它是 append-only 的，
        27 次运行往一个文件追加 27 行，互不干扰。
        而这 27 行正好就是 evaluation.py 想读的东西。
    """
    real_log_file = logger.LOG_FILE
    real_trace_file = logger.TRACE_FILE
    real_memory_path = memory.MEMORY_PATH
    real_summary_file = run_summary.SUMMARY_FILE
    real_stdout = sys.stdout

    logger.LOG_FILE = workdir / "logs" / "agent.log"
    logger.TRACE_FILE = workdir / "logs" / "trace.jsonl"
    memory.MEMORY_PATH = workdir / "memory.json"
    run_summary.SUMMARY_FILE = summary_file or (workdir / "runs" / "summary.jsonl")

    # logger 往控制台也写一份。这里收进内存 ——
    # 一来不刷屏，二来它可以当成「这条用例跑了哪些节点」的证据
    console = io.StringIO()
    sys.stdout = console
    _drop_handlers()

    try:
        yield console
    finally:
        _drop_handlers()
        sys.stdout = real_stdout
        logger.LOG_FILE = real_log_file
        logger.TRACE_FILE = real_trace_file
        memory.MEMORY_PATH = real_memory_path
        run_summary.SUMMARY_FILE = real_summary_file


@contextlib.contextmanager
def _stub_llm():
    """把 llm_node 的客户端换成固定桩，不联网。

    补丁打在 **llm_node** 模块上 —— 谁定义 _build_client，
    就改谁模块里的全局名。（这条规矩这个项目已经踩过四五次了。）

    桩永远回答「季度激励 -> bonus」。所以认不出的列必须叫「季度激励」，
    别的新列名它答不上来 —— 这是故意的，桩只负责把
    「提议 -> pending -> 问人 -> 批准后用上」这条链路跑通。
    """
    response = type(
        "StubResponse",
        (),
        {
            "content": '{"candidates": [{"field": "季度激励", '
            '"suggested_mapping": "bonus", "confidence": 0.9}]}',
            "usage_metadata": {"input_tokens": 1200, "output_tokens": 80},
        },
    )()

    class _StubClient:
        def invoke(self, messages):
            return response

    original = llm_node._build_client
    llm_node._build_client = lambda: _StubClient()
    try:
        yield
    finally:
        llm_node._build_client = original


# ============================================================
# 跑一条用例
# ============================================================


def run_case(case: dict, workdir: Path, summary_file: Path | None = None) -> dict:
    """跑一条，返回判决和证据。

    每条用例都用一个**独立的**临时目录：规则库和日志互不串味。
    共用一个目录的话，前一条批准的规则会漏到后一条 ——
    而那种错不会报错，只会让某条用例的结果悄悄变好或变坏。

    运行记录是唯一可以共用的东西，见 _sandbox。
    """
    case_dir = workdir / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)

    _write_memory(case, case_dir / "memory.json")

    xlsx = case_dir / "payroll.xlsx"
    if case.get("file_kind", "xlsx") != "missing":
        _write_xlsx(case, xlsx)

    expect = case["expect"]
    verdict = {"id": case["id"], "scenario": case["scenario"], "checks": {}}

    try:
        with _sandbox(case_dir, summary_file):
            final = payroll_agent.run(str(xlsx), case.get("decisions"))
    except Exception as exc:
        # 用例把图跑炸了 —— 这本身就是一个结果，不该让整个 benchmark 挂掉
        verdict["crashed"] = f"{type(exc).__name__}: {exc}"
        verdict["checks"] = {key: False for key in ("fields", "route", "calc", "valid", "problem")}
        return verdict

    calculation = final.get("calculation_result") or {}
    validation = final.get("validation_result") or {}

    checks = {
        # 该认出哪些列认不出 —— 集合完全相等才算对
        "fields": sorted(final.get("unknown_fields") or []) == sorted(expect["unknown_fields"]),
        # 该不该解析失败、该不该停下
        "route": bool(final.get("parse_error")) == expect["parse_error"]
        and bool(final.get("halt_reason")) == expect["halt"],
        # 算出来几个人、跳过几个人
        "calc": len(calculation.get("people") or []) == expect["people_count"]
        and len(calculation.get("skipped") or []) == expect["skipped_count"],
    }

    if expect["validation_passed"] is None:
        checks["valid"] = True  # 用例说「这一项不检查」（流程停在 exception，压根没跑校验）
    else:
        checks["valid"] = bool(validation.get("passed")) == expect["validation_passed"]

    if "has_problem" in expect:
        checks["problem"] = bool(final.get("problems")) == expect["has_problem"]
    else:
        checks["problem"] = True

    verdict["checks"] = checks
    verdict["evidence"] = {
        "unknown_fields": final.get("unknown_fields"),
        "parse_error": final.get("parse_error"),
        "halt_reason": (final.get("halt_reason") or "")[:60],
        "people": len(calculation.get("people") or []),
        "skipped": len(calculation.get("skipped") or []),
        "validation_passed": validation.get("passed"),
        "problems": len(final.get("problems") or []),
    }
    return verdict


# ============================================================
# 汇总
# ============================================================

# 指标名 -> 看哪几个检查项。顺序就是报告里的顺序。
METRICS = (
    ("字段识别", ("fields",)),
    ("流程路由", ("route",)),
    ("计算结果", ("calc",)),
    ("结果校验", ("valid",)),
    ("规则冲突", ("problem",)),
)

E2E_CHECKS = ("fields", "route", "calc", "valid", "problem")


def _ratio(hit: int, total: int) -> str:
    return f"{hit / total * 100:5.1f}%" if total else "  n/a"


def summarise(verdicts: list) -> dict:
    total = len(verdicts)

    metrics = {}
    for name, keys in METRICS:
        hit = sum(1 for v in verdicts if all(v["checks"].get(k) for k in keys))
        metrics[name] = {"hit": hit, "total": total}

    e2e = sum(1 for v in verdicts if all(v["checks"].get(k) for k in E2E_CHECKS))
    metrics["端到端"] = {"hit": e2e, "total": total}

    by_scenario = {}
    for v in verdicts:
        bucket = by_scenario.setdefault(v["scenario"], {"hit": 0, "total": 0})
        bucket["total"] += 1
        if all(v["checks"].get(k) for k in E2E_CHECKS):
            bucket["hit"] += 1

    return {"metrics": metrics, "by_scenario": by_scenario, "verdicts": verdicts}


def print_report(result: dict, verbose: bool) -> None:
    verdicts = result["verdicts"]
    total = len(verdicts)

    print()
    print("=" * 62)
    print(f"  评测集 {total} 条  (eval_cases.json)")
    print("=" * 62)

    for name, _ in METRICS:
        m = result["metrics"][name]
        print(f"  {name:<10} {m['hit']:>3} / {m['total']:<3}   {_ratio(m['hit'], m['total'])}")

    m = result["metrics"]["端到端"]
    print("  " + "-" * 46)
    print(f"  {'端到端':<10} {m['hit']:>3} / {m['total']:<3}   {_ratio(m['hit'], m['total'])}")

    print()
    print("  按场景:")
    for scenario, bucket in sorted(result["by_scenario"].items()):
        mark = " " if bucket["hit"] == bucket["total"] else "!"
        print(
            f"   {mark} {scenario:<18} {bucket['hit']}/{bucket['total']}"
        )

    failed = [v for v in verdicts if not all(v["checks"].get(k) for k in E2E_CHECKS)]
    if failed:
        print()
        print("  没过的用例:")
        for v in failed:
            bad = [k for k in E2E_CHECKS if not v["checks"].get(k)]
            print(f"   - {v['id']:<20} 挂在这几项: {', '.join(bad)}")
            if verbose and v.get("evidence"):
                print(f"       实际: {json.dumps(v['evidence'], ensure_ascii=False)}")

    if verbose:
        print()
        print("  逐条:")
        for v in verdicts:
            mark = "ok  " if all(v["checks"].get(k) for k in E2E_CHECKS) else "FAIL"
            print(f"   {mark} {v['id']}")
            if v.get("crashed"):
                print(f"       炸了: {v['crashed']}")

    print()
    print("  注: new_field 三条用的 LLM 是固定桩，测的是链路不是模型质量。")
    print("  注: 想分析这批运行本身，用 --keep-records 留一份，再喂给 evaluation.py。")
    print()


# ============================================================
# 入口
# ============================================================


def main(argv: list) -> int:
    verbose = "-v" in argv or "--verbose" in argv

    json_out = None
    if "--json" in argv:
        index = argv.index("--json")
        json_out = Path(argv[index + 1]) if index + 1 < len(argv) else PROJECT_DIR / "runs" / "benchmark.json"

    # --keep-records DIR：跑完把 27 次运行的记录留一份在 DIR/summary.jsonl。
    #
    # 这是**连接 benchmark 和 evaluation 的那根线** ——
    # 跑完直接 `python evaluation.py --file DIR/summary.jsonl`，
    # evaluation 就能读到这批带 fail_category 和 review 的新记录。
    #
    # 必须显式给目录，没有默认值：不写就一行都不留。
    # 留了的话它会混进项目里的 runs/ —— 那是真运行记录的地方，
    # 混进 27 条评测运行，下次 evaluation 读到的每个数字都是错的。
    keep_dir = None
    if "--keep-records" in argv:
        index = argv.index("--keep-records")
        if index + 1 >= len(argv):
            print("--keep-records 后面要跟一个目录")
            return 2
        keep_dir = Path(argv[index + 1])

    document = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    cases = document["cases"]

    summary_file = None
    if keep_dir is not None and keep_dir.is_dir() and (keep_dir / "summary.jsonl").exists():
        # 追加会跟上一轮的记录混在一起，报告就串了两批数据。
        # 这里不静默覆盖也不静默追加 —— 停下来说清楚。
        print(f"{keep_dir / 'summary.jsonl'} 已经存在了。")
        print("要么换个目录，要么自己先把它删掉 —— 不替你决定覆盖。")
        return 2

    if keep_dir is not None:
        keep_dir.mkdir(parents=True, exist_ok=True)
        summary_file = keep_dir / "summary.jsonl"

    with tempfile.TemporaryDirectory() as workdir:
        root = Path(workdir)
        with _stub_llm():
            verdicts = [run_case(case, root, summary_file) for case in cases]

    result = summarise(verdicts)
    print_report(result, verbose)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "case_count": len(cases),
            "metrics": {k: v for k, v in result["metrics"].items()},
            "by_scenario": result["by_scenario"],
            "failures": [
                {"id": v["id"], "checks": v["checks"], "evidence": v.get("evidence")}
                for v in verdicts
                if not all(v["checks"].get(k) for k in E2E_CHECKS)
            ],
        }
        json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  指标已写入 {json_out}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
