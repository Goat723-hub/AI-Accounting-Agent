"""evaluation.py —— 体检员。

跑法（在 AI-Accounting-Agent 目录下）：

    python evaluation.py                  # 读最近 30 条，出报告
    python evaluation.py --window 50      # 看远一点
    python evaluation.py --file 别的.jsonl # 换个文件读

============================================================
它和 benchmark.py 是两件事
============================================================

    benchmark.py    **主动跑**一遍带标准答案的评测集 -> 出准确率
                    输入 eval_cases.json，回答「有多准」

    evaluation.py   **被动读**已经攒下来的运行记录 -> 出发现
                    输入 runs/summary.jsonl，回答「病在哪」

    一个是考试，一个是体检。

    benchmark 的题是**我出的**（27 条我想得到的场景）；
    evaluation 的题是**现实出的** —— 客户替你出的，你事先不知道。

    benchmark 永远发现不了的三件事，正是这里要回答的：
        频率    3 条 new_field 说明不了「这个行业列名就是乱」
        成本    平均每次调用烧多少 token、有没有白烧
        形状    客户真实的表长什么样，而不是我猜的那样

============================================================
三条不能越的线
============================================================

1. **只读，不写 Agent 的东西。**
   本模块不 import calculation / validation / memory，
   不碰 memory.json，不改 State，不重跑工作流。
   唯一会写的是 runs/eval/ 下自己那份报告 —— 那是它的产出，不是别人的。
   它要是能顺手把规则改了，Human-in-the-loop 那道闸门就白设了。

2. **不自动执行建议。**
   suggested_action 是**给人看的句子**，不是待执行的命令。
   报告里写着「建议给某个字段加别名」，这件事仍然要人去做、人去看。
   自动应用等于把「发现问题的」和「拍板的」合成一个角色。

3. **不猜。**
   数据不够支撑结论时，报告里写「不够」，不写一个好看的数。
   老行缺字段就按「不知道」算，不按 0 算（见 run_summary.read_runs）。

============================================================
输出
============================================================

    runs/eval/<时间戳>-v<格式版本>.json

每个存储版本一份报告，不覆盖 —— 版本之间本来就不该混着比。
（老报告留着还有个用处：下一次的同名 finding 可以直接跟自己比。）
"""

import datetime
import json
import sys
from pathlib import Path

import logger
import run_summary


PROJECT_DIR = Path(__file__).parent

# 报告落在 runs/eval/ 下面，和运行记录 runs/summary.jsonl 同一棵树下。
# 不另开一个顶层目录：报告说的就是这批记录，放一起，清理和备份不会漏掉一个。
EVAL_DIR = PROJECT_DIR / "runs" / "eval"


# 少于这个条数，「占比」这个词就不该用。
#
# 5 条里 1 条异常 = 20%，5 条里 2 条 = 40% —— 差一条就差一个量级，
# 那不是规律，是抖动。定 5 是因为它是「占比」这句话的最低门槛，
# 不是因为它统计上够用（统计上 30 条都嫌少）。
MIN_SAMPLE = 5


# 单次调用输入 token 超过这个数就提醒。
#
# 这是个**业务判断，不是技术判断** —— 和 validation 里那个
# DEDUCTION_WARNING_RATIO 是同一类东西：它不报错，只是说
# 「这个数看着不太对，你看一眼」。
#
# 2000 的依据：判断一列叫什么名字，需要的是**表头**加上少量样例行，
# 不是整张表。这个数是拍出来的，要调就调这里 ——
# 但调之前先看报告里那几个实际数，别凭感觉往下压。
LLM_INPUT_ALERT = 2000


# 该找谁。跟 run_summary.FAIL_CATEGORIES 一一对应 ——
# 那边定义分类，这边定义「分到这一类之后该怎么办」。
# 分开放是因为它们改动的理由不一样：
#   那边改 = 分类口径变了
#   这里改 = 处置办法变了
_ACTION = {
    run_summary.CAT_INPUT: (
        "不用改代码。把这张表退回给客户确认：文件是不是真的存在、"
        "表头是不是换了个写法。然后把这个写法补进 extraction.FIELD_MAP 的别名里。"
    ),
    run_summary.CAT_BUSINESS_HALT: (
        "不用改代码。去看 logs/agent.log 里那个 task_id，"
        "找到 human_review 那几笔，看清是哪个字段被谁否的 —— 去要一个业务口径。"
    ),
    run_summary.CAT_LLM_ERROR: (
        "给 llm_node 的调用加超时和重试。"
        "顺手看一眼是不是 prompt 太长被截断了。"
    ),
    run_summary.CAT_AGENT_ERROR: (
        "这是我们自己的问题。去 logs/agent.log 里翻那个 task_id，"
        "看 failed_node 那一步进出的形状 —— 异常类型在 summary 的 error_type 里。"
    ),
}


# ============================================================
# 工具
# ============================================================


def _confidence(affected: int, total: int) -> float:
    """这条结论有多可信。

    ---- 它回答的是「样本够不够」，不是「概率」----

    20 条数据算不出 p 值。硬报一个 0.87 是在装精确 ——
    那个小数点后两位背后没有任何东西撑着，只会让人以为它是算出来的。

    所以这里只回答一个问题：**这点样本撑不撑得住这句话**。
    规则简单到能心算：

        样本 < 5 条        一律 0.3 —— 少于 5 条，「占比」这个词就没意义
        否则               0.4 + 0.5 × (影响条数 / 总条数)

    于是：影响 1/10 -> 0.45；影响一半 -> 0.65；影响全部 -> 0.9。

    **封顶 0.9，不给 1.0** —— 从样本里推不出「一定」。
    """
    if total < MIN_SAMPLE:
        return 0.3

    return round(min(0.9, 0.4 + 0.5 * (affected / total)), 2)


def _finding(capability: str, code: str, affected: int, total: int, problem: str, action: str) -> dict:
    """造一条发现。

    字段名是照着一开始定的报告格式来的 ——
    换个字段名，读报告的人就要跟着改一遍。
    """
    return {
        "id": f"{capability}.{code}",
        "capability": capability,
        "affected_task_count": affected,
        "detected_problem": problem,
        "suggested_action": action,
        "confidence": _confidence(affected, total),
    }


def _count_field(records: list, extractor) -> dict:
    """按某个键分组计数。

    extractor 收一条记录、返回它落进哪个桶（返回 None 就不算）。
    用函数而不是固定字段名，是因为要数的东西里有列表（review.rejected）
    也有标量（fail_category），取法不一样。
    """
    counter = {}
    for record in records:
        key = extractor(record)
        if key is None:
            continue
        counter[key] = counter.get(key, 0) + 1
    return counter


# ============================================================
# A. 失败分析
# ============================================================


def analyze_failures(records: list) -> list:
    """A. 失败分析 —— 栽在哪、栽了几次、该找谁。

    分两刀切，因为这两刀切出来的东西不一样：

        fail_category    这是**谁的锅** -> 决定去找谁
        failed_node      这是**哪一步** -> 决定去翻哪段日志

    两个都有用，不合并。同一条失败在两边各算一次，
    因为它们回答的是两个不同的问题，不是同一个问题的两种说法。
    """
    total = len(records)
    failures = [r for r in records if r.get("status") == logger.FAILED]

    if not failures:
        return []

    findings = []

    # ---- 第一刀：按「该找谁」分 ----

    by_category = _count_field(failures, lambda r: r.get("fail_category"))
    unclassified = sum(1 for r in failures if r.get("fail_category") is None)

    if by_category:
        # 只报最大的一堆。剩下几堆各自一两条，报出来是列表不是结论。
        top_category, top_count = max(by_category.items(), key=lambda kv: kv[1])

        if top_count >= 2:
            findings.append(
                _finding(
                    "failure",
                    "dominant_category",
                    top_count,
                    total,
                    (
                        f"{len(failures)} 次失败里有 {top_count} 次是同一类：{top_category}。"
                        f"剩下的分布是 {by_category}。"
                    ),
                    _ACTION.get(top_category, "分类是新的，还没有对应的处置办法 —— 补进 _ACTION。"),
                )
            )

    # ---- 老行没这个字段，得说出来 ----
    #
    # 不说的话，报告会是「0 条失败发现」，读的人以为「失败没有规律」。
    # 而真相是「这批行太老，压根归不了类」—— 两句话指向完全不同的动作：
    # 前者是「不用管」，后者是「等数据攒够或重跑一遍」。
    if unclassified >= 2:
        findings.append(
            _finding(
                "failure",
                "unclassified",
                unclassified,
                total,
                (
                    f"{len(failures)} 次失败里有 {unclassified} 次归不了类 —— "
                    f"这些行是 v1 写的，那个时候 summary 还没有 fail_category 这个字段。"
                ),
                (
                    "这一块暂时没法分析。要么等 v2 的新数据攒够，"
                    "要么拿 benchmark.py 重跑一遍评测集，现造一批带分类的记录。"
                ),
            )
        )

    # ---- 第二刀：按「卡在哪个节点」分 ----

    by_node = _count_field(failures, lambda r: r.get("failed_node"))

    if by_node:
        top_node, top_node_count = max(by_node.items(), key=lambda kv: kv[1])

        if top_node_count >= 2:
            findings.append(
                _finding(
                    "failure",
                    "failed_node_hotspot",
                    top_node_count,
                    total,
                    (
                        f"{len(failures)} 次失败里有 {top_node_count} 次停在同一个节点：{top_node}。"
                        f"失败没有散在各处，是卡在一个地方。"
                    ),
                    (
                        f"去 logs/agent.log 里 grep 这个 task_id 附近的 {top_node}，"
                        f"看那几步进出的形状有什么共同点。"
                    ),
                )
            )

    return findings


# ============================================================
# B. LLM 用量分析
# ============================================================


def analyze_llm_usage(records: list) -> list:
    """B. LLM 用量分析 —— 花了多少、有没有白花。

    只看三个数：llm_calls / input_tokens / output_tokens。
    它们必须**一起看**，单看任何一个都会得出反的结论：

        calls=0,  tokens=0     压根没问模型（列全认识，省钱）
        calls=1,  tokens=0     问了，但没拿到用量（多半是失败了）
        calls=1,  tokens=1280  正常问了

    中间那种和第一种在 token 上完全一样。只看 token 的话，
    **失败看起来是免费的** —— 这正是 logger 当初加 llm_calls 的原因，
    这里不能又把它丢掉。
    """
    total = len(records)

    # llm_calls 是 None 说明这行是 v1 写的（那时还没记用量）——
    # 它既不是「调了」也不是「没调」，是「不知道」。单独拎出来。
    known = [r for r in records if r.get("llm_calls") is not None]
    if not known:
        return []

    findings = []

    called = [r for r in known if (r.get("llm_calls") or 0) > 0]
    never = [r for r in known if (r.get("llm_calls") or 0) == 0]

    # ---- 调了，但一个 token 都没记下来 ----

    blind = [r for r in called if (r.get("input_tokens") or 0) == 0]

    if len(blind) >= 2:
        findings.append(
            _finding(
                "llm_usage",
                "usage_missing",
                len(blind),
                total,
                (
                    f"{len(called)} 次问了模型的运行里，有 {len(blind)} 次一个 token 都没记下来。"
                    f"这可能是调用失败（压根没出网），也可能是出网了但服务端没返回用量 —— "
                    f"**现在这两种在记录里长得一模一样**。"
                ),
                (
                    "在 llm_node 里把这两种情况分开：调用失败的那次记一条 error_type，"
                    "拿到响应但没有 usage 字段的另记一种。分不开就没法知道是不是在为一个坏掉的接口付费。"
                ),
            )
        )

    # ---- 单次调用成本 ----

    priced = [r for r in called if (r.get("input_tokens") or 0) > 0]

    if len(priced) >= 2:
        avg_in = sum(r["input_tokens"] for r in priced) / len(priced)
        avg_out = sum(r.get("output_tokens") or 0 for r in priced) / len(priced)

        if avg_in > LLM_INPUT_ALERT:
            findings.append(
                _finding(
                    "llm_usage",
                    "input_tokens_high",
                    len(priced),
                    total,
                    (
                        f"问了模型的 {len(priced)} 次里，平均每次输入 {avg_in:.0f} token、"
                        f"输出 {avg_out:.0f} token。输入是输出的 {avg_in / max(avg_out, 1):.0f} 倍 —— "
                        f"判断一列叫什么名字，需要的是表头，不是整张表。"
                    ),
                    (
                        "看一下 llm_node 拼 prompt 的时候塞了多少东西进去。"
                        "只发表头和列名就够了，把整张表塞进去既贵又不会更准。"
                    ),
                )
            )

    # ---- 一次都没省下来 ----

    if len(never) == 0 and len(known) >= MIN_SAMPLE:
        findings.append(
            _finding(
                "llm_usage",
                "memory_never_hit",
                len(known),
                total,
                (
                    f"{len(known)} 次运行**每一次**都问了模型。"
                    f"memory 存在的前提是「同一个字段第二次不用再问」，"
                    f"一次都没省下来，说明规则库要么是空的，要么一直没命中。"
                ),
                (
                    "翻一下 memory.json 里有几条 approved 规则，再对一下客户表里的列名 —— "
                    "如果规则库是空的，那是从来没走到过「人批准」那一步。"
                ),
            )
        )

    return findings


# ============================================================
# C. 记忆改进建议
# ============================================================


def analyze_memory_signals(records: list) -> list:
    """C. 记忆改进建议 —— 哪些字段老在麻烦人。

    数据来源只有一处：summary 里的 review（人工复核的结论）。
    它只有 v2 的行才有，而且只有**走到过复核**的运行才有。

    ---- 为什么这一块特别容易「假装没问题」----

    没有 review 数据时，下面两个循环一次都不会进 ——
    报告上就是干干净净的「没发现问题」。

    但真相是「**没数据**」，不是「没问题」。这两句话指向的动作完全相反：
        没问题 -> 什么都不用做
        没数据 -> 去把数据补上，再看

    所以没有数据的时候必须**主动报一条**（见下面的 no_review_data）。
    这是本模块最容易骗自己的地方，单独拎出来写清楚。
    """
    total = len(records)
    reviews = [r for r in records if r.get("review")]

    if not reviews:
        if total < MIN_SAMPLE:
            return []
        return [
            _finding(
                "memory",
                "no_review_data",
                total,
                total,
                (
                    f"窗口里 {total} 条记录，一条都没走到人工复核，"
                    f"所以「哪个字段老在麻烦人」这件事**根本没有数据可查**。"
                ),
                (
                    "去 benchmark.py 跑一遍评测集，或者真实跑几张表。"
                    "判断依据：summary 里有多少行的 review 不是 null。"
                ),
            )
        ]

    findings = []

    def _fields(record, key):
        return (record["review"] or {}).get(key) or []

    # ---- 反复被否决的字段 ----
    #
    # 模型提一次错是正常的；同一个字段提两次都被否，说明它每次都这么猜 ——
    # 那是 prompt 的问题，不是运气问题。

    rejected = {}
    for record in reviews:
        for field in _fields(record, "rejected"):
            rejected[field] = rejected.get(field, 0) + 1

    for field, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        if count < 2:
            break

        findings.append(
            _finding(
                "memory",
                f"repeated_rejection::{field}",
                count,
                total,
                (
                    f"「{field}」这个字段被人工否决了 {count} 次。"
                    f"模型不是偶尔猜错，是每次都往同一个方向猜。"
                ),
                (
                    f"两条路，选一条：\n"
                    f"  1. 在 llm_node 的 prompt 里给「{field}」加一条否定示例，告诉它别再这么猜；\n"
                    f"  2. 直接把「{field}」写成 extraction.FIELD_MAP 里的一个别名。\n"
                    f"选 2 要清楚代价：一旦进了 FIELD_MAP，它就被自动认出来，"
                    f"**不再经过人工确认**。这个字段当初是被否过的，凭什么现在可以不问人 —— "
                    f"这个问题得有人回答，本模块不替你回答。"
                ),
            )
        )

    # ---- 一直挂着、没人管的字段 ----
    #
    # 这一类比上面那类更危险：上面至少有人看过并拍了板（否决也是结论），
    # 这一类是**压根没人看**。而当前实现里人不表态流程照走 ——
    # 用到这个字段的人会被静默跳过（见 payroll_agent._halt_reason_after_review
    # 里写明的那个边界）。所以它是「没报错，但工资少算了」。

    undecided = {}
    for record in reviews:
        for field in _fields(record, "undecided"):
            undecided[field] = undecided.get(field, 0) + 1

    for field, count in sorted(undecided.items(), key=lambda kv: -kv[1]):
        if count < 2:
            break

        findings.append(
            _finding(
                "memory",
                f"stuck_undecided::{field}",
                count,
                total,
                (
                    f"「{field}」被提了 {count} 次，每次都卡在「没有人表态」。"
                    f"流程照常往下算了，但用到这个字段的人被静默跳过 —— "
                    f"**没报错，工资少算了**。"
                ),
                (
                    f"两个动作，都得人做：\n"
                    f"  1. 定一个人负责拍板「{field}」到底是什么，别让候选一直挂着；\n"
                    f"  2. 如果这种情况很常见，考虑把 halt 条件从「有人否决才停」"
                    f"改成「候选里还有字段没拿到 approved 就停」—— "
                    f"那是业务取舍（人不回复会让整张表停摆），得业务方点头，代码这边不擅自改。"
                ),
            )
        )

    return findings


# ============================================================
# 报告
# ============================================================


def _window(records: list) -> dict:
    """这份报告看的是哪个窗口、里面有多少能用。

    单独一块，是因为**结论的可信度由它决定** ——
    读报告的人应该先看这里，再看 findings。
    """
    return {
        "analyzed": len(records),
        "success": sum(1 for r in records if r.get("status") == logger.SUCCESS),
        "failed": sum(1 for r in records if r.get("status") == logger.FAILED),
        # 有 review 的才谈得上「记忆改进」，有 fail_category 的才谈得上「失败归因」
        "with_review": sum(1 for r in records if r.get("review")),
        "with_fail_category": sum(1 for r in records if r.get("fail_category")),
        # v1 写的行。它们不是「数据不全」，是「那个时候还没有这个字段」
        "legacy_rows": sum(1 for r in records if r.get("schema_version") is None),
    }


def _notes(records: list, window: dict) -> list:
    """报告末尾的大白话提醒。

    放在 JSON 里而不是只打在屏幕上：报告会被存档、会被人转发，
    看见它的往往不是跑它的那个人 —— 而**局限必须跟着结论一起走**，
    不然结论单飞出去就变成了「XX 系统准确率 100%」。
    """
    notes = []

    if window["analyzed"] < MIN_SAMPLE:
        notes.append(
            f"样本只有 {window['analyzed']} 条（门槛是 {MIN_SAMPLE} 条）—— "
            f"下面所有结论都只是线索，不要当成结论。"
        )

    if window["legacy_rows"]:
        notes.append(
            f"窗口里有 {window['legacy_rows']} 条是 v1 写的，没有 fail_category 也没有 review。"
            f"失败归因和记忆建议这两块的样本量，比「窗口里有几条」要小。"
        )

    if window["failed"] and not window["with_fail_category"]:
        notes.append(
            "有失败，但没有一条能归类 —— 「失败没有规律」这句话，"
            "现在**没有数据支撑**，不要这么读。"
        )

    if not window["with_review"]:
        notes.append(
            "没有一条记录带人工复核结论 —— 这不代表「记忆没有问题」，"
            "只代表这件事还没被观察过。"
        )

    if not notes:
        notes.append("没有需要特别提醒的。")

    return notes


def build_report(records: list) -> dict:
    """把三条能力跑一遍，收成一份报告。

    发现按「影响面」从大到小排 —— 影响面一样时，置信度高的在前。
    排序在这里做而不是各能力自己做：三条能力各自排序，
    合并之后顺序又乱了，等于没排。
    """
    findings = []
    findings += analyze_failures(records)
    findings += analyze_llm_usage(records)
    findings += analyze_memory_signals(records)

    findings.sort(key=lambda f: (-f["affected_task_count"], -f["confidence"], f["id"]))

    window = _window(records)

    return {
        "analysis_time": datetime.datetime.now().strftime(logger.DATE_FORMAT),
        "schema_version": run_summary.SCHEMA_VERSION,
        "window": window,
        "findings": findings,
        "notes": _notes(records, window),
    }


def report_path(moment: datetime.datetime | None = None) -> Path:
    """报告存哪。

    文件名里带格式版本，是因为两个版本的数据本来就不该混着比 ——
    分开存，想对比的时候是显式地拿两份文件比，而不是在同一份里
    把两代数据当成同一代读。
    """
    stamp = (moment or datetime.datetime.now()).strftime("%Y%m%d-%H%M%S")
    return EVAL_DIR / f"{stamp}-v{run_summary.SCHEMA_VERSION}.json"


def print_report(report: dict) -> None:
    """打到屏幕上给人看的那一份。

    屏幕版和文件版**内容同源**（都是这一份 report），
    但排版不一样：文件是给程序读的，屏幕是给人扫的。
    屏幕版只打要点的摘要 —— 全文在文件里，别在终端里铺一屏 JSON。
    """
    window = report["window"]

    print()
    print("=" * 62)
    print(f"  Evaluation  {report['analysis_time']}")
    print("=" * 62)
    print()
    print(
        f"  窗口      最近 {window['analyzed']} 条"
        f"（成功 {window['success']} / 失败 {window['failed']}）"
    )
    print(f"  能用的    带复核结论 {window['with_review']} 条 ｜ 能归类失败 {window['with_fail_category']} 条")
    if window["legacy_rows"]:
        print(f"            其中 {window['legacy_rows']} 条是 v1 老行")
    print()

    findings = report["findings"]

    if not findings:
        print("  没有发现。")
    else:
        print(f"  发现 {len(findings)} 条：")
        print()

        for index, finding in enumerate(findings, start=1):
            print(f"  {index}. [{finding['capability']}] {finding['id'].split('.', 1)[1]}")
            print(
                f"     影响 {finding['affected_task_count']} 次运行"
                f"   ｜   置信度 {finding['confidence']}"
            )
            print(f"     问题   {finding['detected_problem']}")
            for line_index, line in enumerate(finding["suggested_action"].splitlines()):
                prefix = "     建议   " if line_index == 0 else "            "
                print(f"{prefix}{line}")
            print()

    print("  ---- 注意 ----")
    for note in report["notes"]:
        print(f"   - {note}")
    print()


# ============================================================
# 入口
# ============================================================


def _parse_args(argv: list) -> dict:
    """参数解析。手写的 —— 就两个参数，为此引 argparse 不值当。"""
    options = {"window": run_summary.DEFAULT_WINDOW, "path": None, "write": True}

    index = 0
    while index < len(argv):
        token = argv[index]

        if token == "--window" and index + 1 < len(argv):
            options["window"] = int(argv[index + 1])
            index += 2
            continue

        if token == "--file" and index + 1 < len(argv):
            options["path"] = Path(argv[index + 1])
            index += 2
            continue

        if token == "--no-write":
            options["write"] = False
            index += 1
            continue

        print(f"看不懂的参数：{token}")
        print("用法：python evaluation.py [--window N] [--file 路径] [--no-write]")
        raise SystemExit(2)

    return options


def main(argv: list) -> int:
    options = _parse_args(argv)

    records = run_summary.read_runs(limit=options["window"], path=options["path"])
    report = build_report(records)

    print_report(report)

    if options["write"]:
        target = report_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  报告已写入 {target}\n")

    return 0


if __name__ == "__main__":
    # Windows 控制台默认按 GBK 输出，中文会变成乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    sys.exit(main(sys.argv[1:]))
