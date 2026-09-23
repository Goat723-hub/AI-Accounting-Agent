"""payroll_agent.py —— 项目经理。

Agent 只做三件事：

    1. 管理任务状态（State）   现在走到哪了、手里有什么
    2. 调度已有工具（Node）    该谁上场就让谁上场
    3. 根据状态选下一步（Router）  看状态决定往哪条边走

Agent 不做的事：

    不修改业务计算规则、不替代 Validation、不重新处理数据。

    parser / extraction / calculation / validation 都是**被调用的模块**，
    一个都不改。Agent 只负责「什么时候叫谁」，不负责「叫来之后怎么干」。

已经加进来的：

    Memory（规则记忆）—— 见 memory.py。查「这个字段以前确认过映射到哪」。

    LLM（语义分析）—— 见 llm_node.py。
    只有一个节点用它，只干一件事：给没见过的列名提候选映射。

刻意还没做的：

    LLM 决策路由、RAG、Tool Calling。

    路由全是写死的 Python 判断 —— 流程不需要判断力，需要的是可预测。
    等流程稳了，再把某几个判断点换成 LLM，那时才知道换掉的是哪一个。

这一版最重要的一条边界，读代码时可以拿它对照：

    LLM 有提议权，人有批准权，机器只有执行权。

        LLM        只能产出 candidate，一条都进不了 memory.json
        人         只能通过 human_review 的 decisions 表态
        memory     只有 approve_rule 能把状态推到 approved

    三者在代码里是三个不同的模块、三组不同的函数，
    不是靠「约定好不要越界」。这样才查得出来是谁干的。

关于 Router 的一个事实：

    LangGraph 里没有「Router 节点」这种东西。
    节点是干活的，Router 是 `add_conditional_edges()` ——
    一个读 State、返回分支名的普通函数。
    所以下面看不到名叫 router 的节点，只有 route_after_xxx()。
"""

import sys
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from calculation import calculate_payroll
from extraction import extract_rows
from llm_node import llm_rule_analysis_node
from memory import (
    APPROVED,
    PENDING,
    REJECTED,
    apply_rules,
    approve_rule,
    query_rule,
    reject_rule,
    save_candidate_rule,
)
from parser import read_payroll_excel
from validation import validate_payroll


# ============================================================
# State —— Agent 做决定时看的全部事实
# ============================================================
#
# 只放「运行状态和结果」，不放函数、不放工具对象、不放模块引用。
#
# 为什么不能放函数：
#   LangGraph 会把 State 在节点之间传来传去，
#   放进不可序列化的东西，checkpoint / 并行 / 断点续跑全都会出问题。
#
# State 默认的合并方式是**覆盖**：
#   节点返回 {"a": 1}，State 里的 a 就变成 1，原来是什么不重要。
#   两个节点都想往同一个键写，后写的会把先写的挤掉。
#
# 想改成「累加」得用 reducer —— 见下面的 problems。


class PayrollState(TypedDict):
    # ---- 输入 ----
    file_path: str

    # ---- parser 的结果 ----
    raw_rows: list              # 原始行，键是表里原本的列名
    parse_error: str | None     # 解析失败的原因；成功时是 None

    # ---- extraction 的结果 ----
    extracted_rows: list        # 标准化后的行
    unknown_fields: list        # 没认出来的列名（去重）

    # ---- memory 的结果（一）----
    #
    # memory 查到了什么。两个列表，界限是「机器能不能用」：
    #
    #   approved_rules     人工批准过的，机器可以直接拿去改数据
    #   historical_rules   pending / rejected，机器**不能**用，
    #                      但 LLM 要看到（不然会把被否过的提议再提一遍），
    #                      人做决定时也要看到（上次为什么否的）
    #
    # 为什么不让 Router 自己拿 unknown_fields 去减：
    #   「谁批准了」「谁提过」是 memory 的知识，不是 Router 的。
    #   Router 只该读结论，不该重新推一遍 —— 两处各算一遍，迟早会不一致。
    memory_context: dict

    # ---- memory 的结果（二）----
    #
    # memory 里**完全没有记录**的字段，也就是只能去问 LLM 的那批。
    #
    # 为什么它不在 memory_context 里：
    #   它是 Router 的第一判据，单独摆出来读代码时不用先钻进嵌套字典。
    #
    #   代价说清楚：现在 memory 的结论被拆在两个 State 键上，
    #   写它们的是同样两个节点（memory_lookup_node、human_review_node），
    #   改一个就必须改另一个。改的时候别漏。
    #
    # 和 unknown_fields 的区别（这两个很容易混）：
    #   unknown_fields     extraction 说「这一列我不认识」——原始事实
    #   unresolved_fields  memory 说「这个不认识的列，我也没有记录」——结论
    #   前者是后者的输入，后者是前者的子集。
    unresolved_fields: list

    # ---- LLM 的分析结果 ----
    #
    # LLM 提的候选，每条 {field, mapping, confidence, reasoning}。
    #
    # 是**列表**不是单个对象：未知字段本来就可能有好几个，
    # 一次跑要把它们一起分析完。
    #
    # 候选在 llm_rule_analysis_node 里就已经存进 memory 了，状态是 pending ——
    # 不存的话下一轮它还是 unresolved，会再问一遍 LLM。
    #
    # 所以这里装的是「已经存过、但**还没被批准**」的候选。
    # 机器用不了它们（query_rule 默认只返回 approved），
    # 要等 Human Review 批准才变成正式规则。
    llm_analysis: list

    # ---- 人工复核 ----
    #
    # {
    #   "status":    "pending" / "completed"     这次审核做完了没有
    #   "decision":  人看得懂的一句話总结
    #   "reviewer":  谁做的决定
    #   "decisions": [                           逐字段的决定
    #       {"field": "季度激励", "decision": "approve",
    #        "mapping": "bonus", "reason": "..."},
    #   ]
    # }
    #
    # status 记的是「审完了没有」，不是「结论是什么」——
    # 结论在 decisions 里，逐条都有。把两个轴塞进一个字段的话，
    # 「审完了但只批了一半」这种情况就没地方放了。
    #
    # 关键：decisions 是**外部输入**，节点自己不产生它。
    #
    #   生产环境里它来自 interrupt() 恢复时喂进来的值；
    #   演示和测试里由调用方在 initial_state 里预置。
    #
    # 这样 human_review_node 就是纯函数：同样的 State 进，同样的结果出。
    # 如果把 input() 写在节点里，同一个 State 重跑会得到不同结果，
    # 出了问题没法复现，也没法写测试。
    human_review: dict

    # ---- 本次跑新增的规则 ----
    #
    # memory_context.approved_rules 是**全量**（以前批的 + 这次批的），
    # 这个是**增量**：只有这次跑新批准的那些。
    #
    # 为什么要单独记一份：
    #   复盘时最常问的问题是「这次跑改变了什么」，而不是「现在一共有什么」。
    #   全量列表回答不了前者 —— 得跟上一版 memory.json 对比才知道。
    #
    # 本次被**否掉**的规则不在这里 —— 它们还不是规则。
    # 要查否决记录，看 human_review.decisions 和 memory.json 里的 rejected。
    new_memory_rules: list

    # ---- calculation 的结果 ----
    calculation_result: dict    # {"people": [...], "skipped": [...]}

    # ---- validation 的结果 ----
    validation_result: dict     # {"passed":..., "errors": [...], "warnings": [...]}

    # ---- 上游两层的「数据质量」记录 ----
    #
    # parser  的： "第 7 行比表头少 2 列"
    # extraction 的："base_salary 的值 '面议' 不是数字，已按 0 处理"
    #
    # 这些不是错误，是「数据有问题但我照样往下走了」的记录。
    # 不拦流程，但绝不能悄悄消失 —— 否则表里写「面议」，
    # 一路算成 0、一路验证通过，没有任何人知道。
    problems: Annotated[list, add]

    # ---- Agent 自己的状态 ----
    next_action: str            # 每个节点干完活记一笔：我交给谁

    # 流程**为什么提前停了**。正常跑完是 None。
    #
    # 为什么不能只靠 next_action：
    #   next_action 说的是「我交给谁」，不是「为什么」。
    #   走到 exception 有三条完全不同的路 —— 文件读不了、
    #   人工否决了某个字段、上游有别的异常 —— 事后只看
    #   next_action="exception" 分不出是哪种，也不知道该找谁处理。
    #
    # 为什么不能只靠 problems：
    #   problems 是「数据有毛病但我照样往下走」的记录，
    #   它**不拦流程**。而这个字段存在的意义恰恰是「流程被拦下了」。
    #   两件事混在一个列表里，读的人分不清这张表到底算完了没有。
    #
    # 和 parse_error 的关系：
    #   parse_error 是这一个字段的特例（解析层失败），
    #   历史上先有它，后有的 halt_reason。留着不合并是刻意的 ——
    #   parse_error 有明确的归属（parser 的结果），
    #   halt_reason 是 Agent 层的结论，两者由不同的人负责排查。
    halt_reason: str | None


def initial_state(file_path: str, decisions: list | None = None) -> PayrollState:
    """开跑前的 State。

    每个键都给上默认值，节点之间就不用到处 `state.get(x, 默认)`。

    decisions 是人工决定的入口，格式是**列表**：

        [{"field": "季度激励", "decision": "approve", "reason": ""},
         {"field": "房补",     "decision": "reject",  "reason": "不该算进奖金"}]

    为什么是列表不是 {字段: 结论} 的字典：
    字典能表达的东西列表都能表达，反过来不成立 —— 字典塞不下 reason，
    也塞不下「同一个人对同一个字段改了主意」这种历史。
    列表是追加式的，最后一条意见生效，前面的留在那儿看得见。

    默认空列表 —— 也就是说**没人表态时什么都不批**。
    这个默认值是刻意的：默认批准等于「没人看过的规则自动生效」，
    而它会一直生效下去，且不留痕迹。
    """
    return {
        "file_path": file_path,
        "raw_rows": [],
        "parse_error": None,
        "extracted_rows": [],
        "unknown_fields": [],
        "unresolved_fields": [],
        "memory_context": {
            "approved_rules": [],
            "historical_rules": [],
        },
        "llm_analysis": [],
        "human_review": {
            "status": "pending",
            "decision": "",
            "reviewer": "",
            "decisions": list(decisions or []),
        },
        "new_memory_rules": [],
        "calculation_result": {},
        "validation_result": {},
        "problems": [],
        "next_action": "",
        "halt_reason": None,
    }


# ============================================================
# Node —— 每个节点只干一件事：调用一个已有模块，把结果写进 State
# ============================================================
#
# 节点函数的约定（LangGraph 的规矩）：
#
#     进来的是完整 State，回去的是一个**局部更新**。
#     LangGraph 把返回的字典合并进 State，不是替换。
#
#     所以 parse_node 只返回它动过的那三个键，
#     别的键它不碰，也不该碰。


def parse_node(state: PayrollState) -> dict:
    """调用 parser：把 Excel 读成原始行。

    这是整条链唯一的文件 IO。后面所有节点都只看 State，不碰磁盘。
    """
    result = read_payroll_excel(state["file_path"])

    if not result["ok"]:
        # 文件不存在 / 不是 xlsx / 找不到表头 —— 全都到这儿。
        # 记下原因就走异常分支，不往下传空数据。
        #
        # 注意这里 **不抛异常**：抛出去整张图就炸了，
        # 而「资料没给全」是业务流程里正常会发生的一件事，不是程序错误。
        return {
            "raw_rows": [],
            "parse_error": result["error"],
            "problems": result["problems"],
            "next_action": "exception",
        }

    return {
        "raw_rows": result["rows"],
        "parse_error": None,
        # parser 记下的格式问题（哪一行比表头短）跟着 State 往下走
        "problems": result["problems"],
        "next_action": "extract",
    }


def extract_node(state: PayrollState) -> dict:
    """调用 extraction：原始列名 -> 标准字段。"""
    result = extract_rows(state["raw_rows"])

    return {
        "extracted_rows": result["rows"],
        "unknown_fields": result["unknown_fields"],
        # 这一条**接在** parser 的 problems 后面，不是覆盖它 ——
        # 靠的是 State 里 problems 那一行写的 Annotated[list, add]。
        # 没那个 reducer，这里返回的列表会把 parser 记的整段挤掉。
        "problems": result["problems"],
        "next_action": "memory_lookup",
    }


def memory_lookup_node(state: PayrollState) -> dict:
    """调用 memory：把没认出来的字段分成三类。

    为什么只在 unknown_fields 上查：
        认出来的字段不需要规则，它们走的是 extraction 的 FIELD_MAP。
        拿已经认识的东西去查记忆，只会得到一堆噪音。

    为什么要分三类，而不是只查 approved：

        approved_rules     机器可以直接用的
        historical_rules   以前提过、被人工否过的 —— 机器**不能**用，
                           但 LLM 需要看到（不然会把被否过的提议再提一遍），
                           人做决定时也需要看到（上次为什么否的）
        unresolved_fields  完全没见过的，只能问 LLM

    一次查三个状态、再按状态归类，比查两遍更省事，也不会出现
    「approved 查一遍、pending 查一遍」导致的两份数据对不上。

    **这个节点只查，不用，也不问 LLM。**

    查到 approved 要不要应用、没见过的要不要问 LLM，都是 Router 的判断。
    合在一起的话，Memory 一命中就会直接动 extracted_rows，
    你没法在不改数据的前提下先看一眼查到了什么。
    """
    approved, historical, unresolved = [], [], []

    for field in state["unknown_fields"]:
        rule = query_rule(field, statuses=(APPROVED, PENDING, REJECTED))

        if not rule["found"]:
            # memory 里完全没有这个字段的记录 —— 是正常情况（第一次遇到），
            # 不是错误，所以不记 problems
            unresolved.append(field)
            continue

        record = {
            "field": field,
            "mapping": rule["mapping"],
            "status": rule["status"],
            "confidence": rule["confidence"],
        }

        if rule["status"] == APPROVED:
            approved.append(record)
        else:
            # pending / rejected 都只是「历史」。
            # 机器不能拿它们去改数据 —— 一个没批过的映射一旦生效，
            # 出错的方式是：工资多算了、少算了，而所有日志都显示流程正常。
            historical.append(record)

    return {
        "memory_context": {
            "approved_rules": approved,
            "historical_rules": historical,
        },
        # 单独一个键，不在 memory_context 里。两个键由**同一行代码**产出，
        # 就是为了让它们不会各写各的 —— 见 PayrollState 里的说明。
        "unresolved_fields": unresolved,
        "next_action": "route",
    }


def human_review_node(state: PayrollState) -> dict:
    """人工复核：决定每条候选规则要不要变成正式规则。

    审两批东西：

        llm_analysis                    LLM 刚提的（还没进 memory）
        historical_rules 里 pending 的  以前提过、一直没人拍板的

    决定从 state["human_review"]["decisions"] 读，格式 {字段名: "approve"}。
    **这个节点不产生决定，它只执行决定** —— 所以它是纯函数，
    同样输入永远同样输出，debug 时不用猜「上次是谁点的」。

    批准之后有一步**必须**做：把规则追加进 memory_context.approved_rules。

        不追加的话，下一步 apply 读到的还是批准之前那份快照，
        人工批了等于没批。这是这一版最容易漏、又最难发现的 bug ——
        流程一路跑通、validation 也可能过，只是那份工资少算了刚批的字段。

    为什么否决也要写进 memory：
        不写的话下次 LLM 还会提同样的建议，人还得再拒一次。
        写进去之后 save_candidate_rule 会挡住重复提议。
    """
    candidates = _collect_candidates(state)
    decisions = state["human_review"].get("decisions") or []
    reviewer = state["human_review"].get("reviewer") or "human"

    approved_now, rejected_now, undecided = [], [], []
    problems = []

    for candidate in candidates:
        entry = _decision_for(decisions, candidate["field"])
        verdict = str(entry.get("decision", "")).strip().lower()

        if verdict in ("approve", "approved"):
            if _apply_approval(candidate, problems):
                approved_now.append(candidate)

        elif verdict in ("reject", "rejected"):
            _apply_rejection(candidate, entry.get("reason", ""), problems)
            rejected_now.append(candidate["field"])

        else:
            # 没有决定 = 不批准，保持 pending。
            # 默认放行是这里最危险的一个选择：人没看过的规则生效之后
            # 会一直生效下去，而且下次跑流程时它已经是一条「approved」，
            # 没人再会去看它第二眼。
            undecided.append(candidate["field"])

    return {
        "new_memory_rules": [
            {
                "field": candidate["field"],
                "mapping": candidate["mapping"],
                "status": APPROVED,
                "confidence": candidate["confidence"],
            }
            for candidate in approved_now
        ],
        **_context_after_review(state, candidates, approved_now, rejected_now),
        "human_review": {
            # 审完了没有 —— 不是结论是什么。
            # 一个候选都没有，也算审完了（没活干），所以这里用 == 判断。
            "status": "pending" if undecided else "completed",
            "decision": _review_summary(approved_now, rejected_now, undecided, candidates),
            "reviewer": reviewer,
            "decisions": decisions,
        },
        "problems": problems,
        # 有人否决 -> 记下为什么停，后面的 Router 据此拐去 exception。
        # 节点只负责「记录事实」，不负责「决定去哪」—— 那是 Router 的事。
        "halt_reason": _halt_reason_after_review(rejected_now, approved_now),
        # "route" 而不是 "apply_memory_rules"：
        # 下一步去哪要看 halt_reason，这个节点自己不知道。
        # memory_lookup_node 结尾也是这么写的，两个节点的说法保持一致。
        "next_action": "route",
    }


def apply_memory_rules_node(state: PayrollState) -> dict:
    """把查到的规则用到数据上。

    规则说 餐补 -> bonus，就把每行 unmapped 里的「餐补」搬进 bonus。

    写回的是 extracted_rows —— 也就是说，**这一步之后的行，
    和 extraction 刚产出的行不是同一份了**。

    这是故意的：calculation 拿到的应该是「尽量完整」的数据，
    它不该知道某列是靠记忆认出来的。

    代价要说清楚：validation 里的 source_match 是拿 extracted_rows
    和 net_pay_in_file 对账的。如果一条规则映射错了（把「罚款」
    认成了奖金），这一步会安静地改变结果，而 validation 看不出来 ——
    它校验的是「算得对不对」，不是「字段认得对不对」。
    挡住这个的是 approve_rule 那道人工闸门。

    注意这里的规则是从 State 读的，不是重新查 memory：
        这样「应用了什么」在最终 State 里完全可见，事后复盘时
        不用去猜当时的 memory.json 长什么样 —— 它可能早就被改了。
    """
    result = apply_rules(
        state["extracted_rows"],
        state["memory_context"]["approved_rules"],
    )

    return {
        "extracted_rows": result["rows"],
        # 规则冲突（这行本来就有 bonus）不能静默略过
        "problems": result["problems"],
        "next_action": "calculate",
    }


def calculate_node(state: PayrollState) -> dict:
    """调用 calculation：按固定规则算工资。"""
    result = calculate_payroll(state["extracted_rows"])

    return {
        "calculation_result": result,
        "next_action": "validate",
    }


def validate_node(state: PayrollState) -> dict:
    """调用 validation：独立检查计算结果。

    注意参数怎么来的：
        rows   <- State 里的 extracted_rows
        people <- State 里 calculation_result 的 people

    validation 不知道 calculation 是怎么算的，它只是拿到两份数据对账。
    这里也不做任何「算不过就改一改让它过」的事 ——
    结果原样存进 State，过不过是 validation 说了算。
    """
    result = validate_payroll(
        state["extracted_rows"],
        state["calculation_result"]["people"],
    )

    return {
        "validation_result": result,
        "next_action": "end",
    }


def exception_node(state: PayrollState) -> dict:
    """出事了，停在这儿。

    它做两件事：把流程收住，以及**保证 halt_reason 一定有值**。

    走到这里的原因不止一个：

        文件读不进来      parse_error 里有，halt_reason 还是空的
        人工否决了字段    human_review_node 已经写了 halt_reason

    要是这里不兜底，看最终 State 的人会看到 next_action="exception"
    和一个空的 halt_reason —— 知道出事了，不知道出的什么事。
    所以最后一道口子由这个节点来收：**停下来了就一定有原因**。

    它**不负责**决定怎么办 —— 是补文件、找客户确认、还是人工算，
    那是人的事。它的价值在于：流程停在一个**看得见的地方**，
    而不是带着空数据一路跑到底、最后给出一份看起来正常的工资表。
    """
    reason = (
        state.get("halt_reason")
        or state.get("parse_error")
        or "流程在 exception 节点终止，但上游没有记下原因"
    )

    return {"next_action": "exception", "halt_reason": reason}


# ============================================================
# 人工复核用到的几个纯函数
#
# 抽出来是因为 human_review_node 本身已经够长，
# 再把分支判断混在里面，「谁在什么时候被批准了」这条线就读不清了。
# ============================================================


def _collect_candidates(state: PayrollState) -> list:
    """本轮要审的东西。

    三批合在一起：LLM 刚提的、以前提过没人拍板的、以前被否掉的。

    为什么被否掉的也要摆给人看：

        一个字段被否之后，memory 里就只剩一条 rejected 记录，
        它既不是 unresolved（有记录）也没有可用的映射。
        以前这种字段会走到 apply 分支、什么都不做、流程正常结束 ——
        那个人算不出工资，而**没有任何人被问过**。

        摆出来至少让人知道「张伟算不出来，是因为季度激励这个字段
        你上次否了」。人可以维持否决，也可以用 approve 把它翻回来。

    同一个字段只留一条（LLM 提的优先）：
        LLM 那条是这次基于最新数据算出来的，memory 里那条是旧的。
        两条都摆给人看，人只会更犹豫。

    status 字段是给 _review_summary 用的 —— 它要能分清
    「等你拍板」（pending）和「维持否决」（rejected），
    这两种「没决定」的含义完全不一样。
    """
    candidates = []
    seen = set()

    for item in state["llm_analysis"]:
        candidates.append(
            {
                "field": item["field"],
                "mapping": item["mapping"],
                "confidence": item["confidence"],
                "reasoning": item.get("reasoning", ""),
                "status": "proposed",
            }
        )
        seen.add(item["field"])

    for rule in state["memory_context"]["historical_rules"]:
        if rule["field"] in seen:
            continue
        candidates.append(
            {
                "field": rule["field"],
                "mapping": rule["mapping"],
                "confidence": rule["confidence"],
                "reasoning": "",
                "status": rule["status"],  # pending 或 rejected
            }
        )

    return candidates


def _apply_approval(candidate: dict, problems: list) -> bool:
    """把一条候选落成正式规则。返回有没有真的落地。

    顺序是「先存、再批」，而且 save 失败时**照样批**。这不是偷懒：

        memory 里这个字段是什么记录，取决于候选是从哪来的，
        而候选的来源只有三种（见 _collect_candidates）：

            没记录      -> save 建一条 pending     -> 存得进去
            pending     -> save 刷成新的          -> 存得进去
            rejected    -> save 会被拒（同映射不重复提）
                          但那正是「人工推翻自己上次的否决」，
                          本来就该放行，不能因为存不进去就不批

    所以 save 的返回值这里**故意不看**：它成不成功只决定
    「这次有没有记进 memory」，不决定「能不能批准」。
    批准作用的是这个字段已有的那条记录，映射和候选是一致的
    （pending/rejected 的候选就是从那条记录抄来的）。

    什么情况下才真的批不了：这个字段在 memory 里根本没记录，
    而且刚才那次 save 也没建成。那时候 approve_rule 会返回失败，
    下面那个 if 会把它记下来 —— 不静默吞掉。
    """
    field = candidate["field"]

    save_candidate_rule(field, candidate["mapping"], candidate["confidence"])

    result = approve_rule(field)

    if not result.get("approved"):
        problems.append(
            f"人工批准了 {field}，但 approve_rule 没成功：{result.get('reason')}"
        )
        return False

    return True


def _apply_rejection(candidate: dict, reason: str, problems: list) -> None:
    """把一条候选否掉。

    先 save 再 reject：memory 里要留下一条 rejected 记录，
    这样下次 LLM 提同样的映射会被 save_candidate_rule 挡掉，
    人不用再把同一个提议拒第二遍。

    save 的返回值同样不看，理由和 _apply_approval 一样。

    注意 reject_rule 对已经是 rejected 的记录也成立（changed=False），
    所以「人工维持否决」走这条路径不会出错，也不会重复写盘。
    —— 「同样映射的第二次否决不覆盖 notes」也是对的：
    第一次写下的理由才是当时真正的理由，第二次只是重申。

    reason 现在从决定里带过来了，不再写死「人工复核否决」。
    这不是措辞问题：rejected 记录会留在 memory.json 里被下一个人看到，
    「交通补贴不该算进奖金」和「人工复核否决」对读的人来说
    一个是依据、一个是废话。
    """
    field = candidate["field"]

    save_candidate_rule(field, candidate["mapping"], candidate["confidence"])

    result = reject_rule(field, reason=reason or "人工复核否决")
    if not result.get("rejected"):
        problems.append(
            f"人工否决了 {field}，但 reject_rule 没成功：{result.get('reason')}"
        )


def _decision_for(decisions: list, field: str) -> dict:
    """从决定列表里取这个字段的决定。没有就返回空字典。

    同一个字段出现多次时取**最后一条**。

    为什么是最后一条而不是第一条：
        决定列表是追加式的（见 initial_state 的说明）。
        人对同一个字段改主意时，新的一条接在后面 ——
        最新的意见才是他的意见。

    为什么返回空字典而不是 None：
        调用点要读 entry.get("decision") 和 entry.get("reason")，
        返回 {} 的话这两句直接就能用，不用每次先判空。
        「没决定」和「决定内容是空」在下游是同一件事 —— 都不批。
    """
    found = {}

    for item in decisions:
        if isinstance(item, dict) and item.get("field") == field:
            found = item

    return found


def _context_after_review(
    state: PayrollState, candidates: list, approved_now: list, rejected_now: list
) -> dict:
    """把人工的决定并回 memory_context 和 unresolved_fields。

    参数是**整个 state** 而不是 memory_context：
        要改的两个键分居 State 的两处，只传 memory_context 读不到
        unresolved_fields —— 这正是把它拆出去要付的代价，
        代价就落在这里：这个函数的入参变宽了。

    返回的是**两个键**，不是一个 —— State 里它们本来就是分开的。
    返回值直接 `**` 展开进节点的局部更新里。

    三件事：
        1. 新批准的规则追加进 approved_rules —— 不追加，apply 就漏掉它们
        2. **已定案的**字段从 historical_rules 里去掉
        3. **所有**候选字段从 unresolved_fields 里去掉

    2 和 3 的范围不一样，这不是笔误：

        historical_rules 里的每条都带一个 status，那个 status 是
        memory_lookup 那一刻抄下来的。人工定了案之后它就不新鲜了 ——
        「季度激励（pending）」这条要是留着，人看最终 State 时会以为
        它还挂着，其实刚被批准了。所以定过案的必须移走。

        但**没定案的不能移**：它还是 pending，status 依然是准的，
        移走反而会让最终 State 显示「没有待定规则」，与事实不符。

        unresolved_fields 则是所有候选都要移 —— 因为无论批没批，
        这一步之后 memory 里都有一份记录了，它们不再是「没见过」。

        （这正是 llm_rule_analysis_node 要把候选存成 pending 的连带好处：
          连没定案的字段也不会在下一轮被重复问一遍 LLM。）
    """
    context = state["memory_context"]
    approved_rules = list(context["approved_rules"])

    for candidate in approved_now:
        approved_rules.append(
            {
                "field": candidate["field"],
                "mapping": candidate["mapping"],
                "status": APPROVED,
                "confidence": candidate["confidence"],
            }
        )

    all_candidate_fields = {candidate["field"] for candidate in candidates}
    decided_fields = {c["field"] for c in approved_now} | set(rejected_now)

    return {
        "memory_context": {
            "approved_rules": approved_rules,
            "historical_rules": [
                r for r in context["historical_rules"] if r["field"] not in decided_fields
            ],
        },
        "unresolved_fields": [
            f for f in state["unresolved_fields"] if f not in all_candidate_fields
        ],
    }


def _review_summary(approved_now, rejected_now, undecided, candidates) -> str:
    """人看得懂的一句话结论。

    单独写一个函数，是因为「本轮发生了什么」要被记进 State、
    展示在 demo 里、以后可能还要写进审计日志 ——
    它是给人看的，所以它不该只存在于某个 if 分支的注释里。

    「没决定」要分成两种说法，因为后果完全不同：

        保持待定   -> 还有机会，下次接着问
        维持否决   -> 这个人这次算不出工资，得让人知道是这个字段害的

    不分的话，日志里只会写「没有人工决定」，看的人不知道
    到底是有事发生还是没事发生。
    """
    if not candidates:
        return "本轮没有需要人工复核的候选"

    if undecided:
        by_field = {candidate["field"]: candidate["status"] for candidate in candidates}
        waiting = [f for f in undecided if by_field.get(f) != REJECTED]
        kept_rejected = [f for f in undecided if by_field.get(f) == REJECTED]

        parts = []
        if waiting:
            parts.append(f"以下字段没有人工决定，保持待定：{'、'.join(waiting)}")
        if kept_rejected:
            parts.append(
                f"以下字段维持否决，用到它们的人算不出工资：{'、'.join(kept_rejected)}"
            )
        return "；".join(parts)

    parts = []
    for candidate in approved_now:
        parts.append(f"批准 {candidate['field']} -> {candidate['mapping']}")
    for field in rejected_now:
        parts.append(f"否决 {field}")

    return "；".join(parts)


def _halt_reason_after_review(rejected_now: list, approved_now: list) -> str | None:
    """人工否决之后，要不要把整条流程停下来。返回停下来的原因，不停就是 None。

    为什么否决一个字段就要停整个流程，而不是「跳过它、别人照算」：

        字段还在表里，用到它的人还在表上。跳过它的后果不是「少算一点」，
        是**那些人根本没有结果** —— calculation 把他们放进 skipped，
        不进 people。而 validation 是按人核对「表里写的实发 vs 我算的实发」的，
        这些人**不在 people 里，没有可对的行**，所以校验一路通过。

        最终输出：一份看起来正常、其实少发了几个人工资的表，
        而所有环节都显示成功。这是最坏的一种错 —— 安静而且不可发现。

        停下来至少是**响的**：halt_reason 写清楚是哪个字段、谁否的。
        「这次没算出来」比「算出来了但是错的」便宜得多。

    代价也说清楚：

        20 人的表里 1 个字段有争议，20 个人这次都拿不到工资。
        这是故意的 —— 工资表的完整性不该由「其他人能不能先拿到钱」来妥协。
        真要局部放行，那是业务流程上的决定，得由人来显式地说
        「这批先按老口径发」，而不是由 Agent 自己猜。

    为什么只否决才算停，不表态不算（这一条是**当前实现的边界**）：

        人没表态（undecided）时流程照常往下算。这留下一个和上面同类的洞：
        不表态的字段同样没有可用映射，用到它的人同样会被静默跳过。
        要堵的话，把这里的条件从 `rejected_now` 换成
        「候选里还有字段没拿到 approved」即可 —— 但那意味着
        「人没回复」也会让整张表停摆，那是另一个业务取舍，得你点头。
    """
    if not rejected_now:
        return None

    fields = "、".join(rejected_now)
    approved_hint = ""

    if approved_now:
        # 说清楚「不是全否了」—— 否则看日志的人会以为是规则全被推翻。
        approved_hint = (
            f"（同一次复核里批准的 {'、'.join(c['field'] for c in approved_now)} "
            f"已经写进 memory，下次跑会直接用上）"
        )

    return (
        f"人工否决了字段「{fields}」，但这些列还在工资表里。"
        f"它们算不进工资，这张表就是不全的 —— "
        f"与其算一份少发钱的工资，不如停在这里等人处理。{approved_hint}"
    )


# ============================================================
# Router —— 读 State，返回下一站的名字
# ============================================================
#
# Router 是纯函数：只看 State，不看外部变量，不改任何东西。
# 同样的 State 进去，永远同样的分支出来 —— 这是流程可复现的前提。


def route_after_parse(state: PayrollState) -> str:
    """解析成功了才往下走。"""
    return "exception" if state.get("parse_error") else "extract"


def route_after_memory_lookup(state: PayrollState) -> str:
    """看完 memory 之后，决定还缺什么、下一步该找谁。

    四个出口，按「缺的东西有多难补」排序：

        llm_analysis  有字段完全没见过   -> 只能问 LLM
        human_review  见过但没定案       -> 候选已经有了（pending）或者被人
                                          否过（rejected），都得找人
        apply         规则齐了，没别的活 -> 谁也不用找，直接应用
        calculate     连规则都没有       -> 没有可应用的，跳过 apply

    为什么 llm_analysis 排在 human_review 前面：

        一次跑要把所有没解决的字段都处理掉。如果先去做 review，
        那几个没见过的字段这一轮就没人管了 —— 流程照样跑完，
        用到它们的人被 calculation 跳过，而**没有任何节点碰过它们**。
        而 review 那边本来就是「等人工」，早一轮晚一轮区别不大。

    为什么 human_review 排在 apply 前面：

        同样的道理。一个 pending 字段机器用不了（query_rule 默认只给
        approved），apply 会直接跳过它，流程正常结束，那个人算不出工资。
        静默地少算一个人，比多问一次人工贵得多。

    为什么 apply 和 calculate 分成两个出口：

        「有规则可应用」和「一条规则都没有」是两件事。
        「没有规则」时 apply 确实是个空操作（apply_rules 收到空列表
        原样返回），把它并进 apply 在行为上完全等价 —— 分开是为了让
        Router 的四种情况在**这一段代码里**就完整可读，
        不用翻到 apply_memory_rules 才知道没规则时它干什么。

    关于「approved 优先，不能调 LLM」：

        这个函数判不了这件事，因为压根没有可判的东西 ——
        memory_lookup 已经把有 approved 记录的字段从 unresolved 里摘掉了，
        它们根本不会走到这里。第一句查 unresolved，查的就是
        「memory 里一条记录都没有」的字段。

        所以「固定规则优先」是在**分层那一刻**锁定的，不是在这里判的。
        这也是为什么这里不需要一个 `if approved: return "apply"` 的分支：
        真有那个分支，反而会让「同时有已批准规则和全新字段」的表
        跳过新字段 —— 见下面 rejected 那段，是同一类坑。

    rejected 为什么也算「没定案」：

        一开始我把它当成定案了 —— 被否掉的字段走到 apply、什么都不做、
        流程正常结束。结果是一个字段一旦被否就永远卡住：
        那个人算不出工资，而没有任何人被问过第二遍，
        memory.py 里「换个映射还能再提」那条规则也永远触发不了。

        现在它进 human_review：人能看到「这个字段你上次否了，所以这批人
        算不出来」，维持或者翻回来都行。

        代价是：只要那个字段还在表里，每跑一次就会问一次人工。
        这是故意的 —— 静默地少算一个人，比多问一次贵得多。

    关于那句显式的 status 判断：

        memory_lookup 只会把非 approved 的放进 historical_rules，
        所以 `if context["historical_rules"]:` 在今天的代码里是等价的。
        留着显式的状态比较，是为了让「哪些状态算没定案」写在这一行上 ——
        以后 memory 多出一个状态（比如过期），改动点在这儿，一眼能看到。
        它确实和 memory_lookup 的分类逻辑重复了，这是明知故犯：
        重复的是一句 if，不是一份数据。
    """
    context = state["memory_context"]

    if state["unresolved_fields"]:
        return "llm_analysis"

    if any(
        rule["status"] in (PENDING, REJECTED) for rule in context["historical_rules"]
    ):
        return "human_review"

    if context["approved_rules"]:
        return "apply"

    # 什么规则都没有：没有可以应用的，也没有要问的。
    # 直接去算 —— 这时候的计算结果就是「只用了 extraction 认出来的列」。
    return "calculate"


def route_after_human_review(state: PayrollState) -> str:
    """人做完决定之后：能算就走，算不全就停。

    判据只有一条 —— halt_reason 有没有值。为什么是读 State 而不是在这里
    重新判一遍「有没有人否决」：

        否决这件事发生在 human_review_node 里，它比 Router 更清楚
        发生了什么（谁否的、同一次里还批了什么）。让它把结论写进 State，
        Router 只读结论 —— 两处各判一遍的话，改了一处另一处不会跟着改，
        而且不会有任何报错，只会某天开始悄悄走错分支。

    这也是 Router 的通用原则：**只读，不推**。

    为什么不在 human_review_node 里直接 add_edge 到 exception：

        节点的返回值决定**State 变成什么样**，边的走向决定**流程去哪**。
        混在一起的话，一个节点就有了两个职责，而且「什么情况下会停」
        这件事会散落在节点内部，读图的人看不出来 ——
        他只会看到 human_review 后面有一堆边，不知道哪条什么时候走。
    """
    return "exception" if state.get("halt_reason") else "apply"


# ============================================================
# 节点日志
# ============================================================
#
# 出了问题时想知道的三件事：
#
#     哪个节点跑了          node
#     它进来时手上有什么    input
#     它把 State 改成了什么 changed
#
# 第三件其实 .stream() 也能给（它吐的就是每个节点的局部更新），
# 缺的是第一件和第二件 —— 尤其是**第二件**：
# 一个节点判断错了，通常不是因为它逻辑写错，是因为它看到的 State
# 不是你以为的那个样子。把「它进来时看到了什么」记下来，
# 就不用靠重跑加 print 去猜。


def traced(name: str, node, sink: list | None):
    """把一个节点包一层，记录它的进出。sink 是 None 就什么都不记。

    为什么写成包装器，而不是在每个节点函数里插一句记录：

        节点就不知道自己在被记录 —— 它的签名还是「State 进，局部更新出」，
        单独 import 出来测一个节点时不会打印一堆东西，
        也不会因为没人给它 sink 就崩。

        记录是**图层的事**：谁被记录、记到哪去，由 build_graph 决定。

    为什么 sink 是参数不是全局变量：

        全局的话跑两次之间要记得清空，忘了就会看到上一次的记录混进来，
        而且看的人会以为是这一次跑的 —— 调试工具本身制造 bug 是最糟的。

    为什么 sink 是 None 时还要走一遍这个包装：

        为了让「有日志」和「没日志」走的是**同一条代码路径**。
        如果日志只在不传 sink 时装上，那上线跑的图和你本地调试的图
        就是两个不同的图，调通了不代表线上没问题。
    """
    def wrapped(state: PayrollState) -> dict:
        update = node(state)

        if sink is not None:
            sink.append(
                {
                    "node": name,
                    "input": _snapshot(state),
                    "changed": sorted(update.keys()),
                    "next_action": update.get("next_action"),
                }
            )

        return update

    return wrapped


def _snapshot(state: dict) -> dict:
    """State 的快照：只报有内容的键，值只报「有多大」。

    为什么要滤掉空的：
        默认值是 None / [] / "" 的键到处都是，全报出来是一面墙，
        真正有信息的那几个被淹掉。空的不报，报的都是这次跑真的产生了什么的。

    为什么不展开内容：
        extracted_rows 可能有几千行。日志打全了就没法看，
        要看内容去看最终 State —— 日志回答的是「形状对不对」，
        不是「值对不对」。
    """
    return {key: _size_of(value) for key, value in state.items() if _has_content(value)}


def _has_content(value) -> bool:
    """这个值值不值得报。空列表、空字典、空串、None 都不值得。"""
    return value is not None and value != "" and value != [] and value != {}


def _size_of(value):
    """把值压成「有多大」。

    字典往下钻一层：memory_context 报成 {'approved_rules': 1, 'historical_rules': 0}
    比报成「2 个键」有用得多 —— 后者看不出机器到底能不能用这些规则。
    再往下就不钻了，收益递减而噪音翻倍。
    """
    if isinstance(value, list):
        return len(value)

    if isinstance(value, dict):
        return {
            key: (len(item) if isinstance(item, (list, dict)) else item)
            for key, item in value.items()
        }

    return value


# ============================================================
# 图
# ============================================================


def build_graph(trace: list | None = None):
    """把节点和边接起来。

    加边的两种方式，区别就一条：

        add_edge                A 之后必然走 B，没得选
        add_conditional_edges   A 之后走哪，看 route_xxx 返回什么

    返回值的映射 {分支名: 节点名} 不能省 ——
    不写的话 LangGraph 会拿分支名当节点名去找，名字对不上就报错。

    trace 传一个列表进来就会记录每个节点的进出，不传就没有额外开销。
    日志**不放进 State**：它是诊断信息不是业务结果，塞进 State 会让
    checkpoint 跟着膨胀，而且没有哪个 Router 会去读它。
    """
    graph = StateGraph(PayrollState)

    graph.add_node("parse", traced("parse", parse_node, trace))
    graph.add_node("extract", traced("extract", extract_node, trace))
    graph.add_node("memory_lookup", traced("memory_lookup", memory_lookup_node, trace))
    graph.add_node(
        "llm_rule_analysis", traced("llm_rule_analysis", llm_rule_analysis_node, trace)
    )
    graph.add_node("human_review", traced("human_review", human_review_node, trace))
    graph.add_node(
        "apply_memory_rules", traced("apply_memory_rules", apply_memory_rules_node, trace)
    )
    graph.add_node("calculate", traced("calculate", calculate_node, trace))
    graph.add_node("validate", traced("validate", validate_node, trace))
    graph.add_node("exception", traced("exception", exception_node, trace))

    graph.add_edge(START, "parse")

    graph.add_conditional_edges(
        "parse",
        route_after_parse,
        {"extract": "extract", "exception": "exception"},
    )

    # extract 之后**还是普通边**，尽管后面接的是记忆环节。
    #
    # 认不认得出字段，是 extraction 自己的事 —— 认不出它会把列名
    # 放进 unknown_fields 记着，但不拦流程（客户表多一列「餐补」很正常）。
    # 所以这里没有分支可走。
    graph.add_edge("extract", "memory_lookup")

    # 这里才是分支：缺什么就去找谁。
    #
    # 四个分支名和节点名**故意不一样**（llm_analysis / apply），
    # 因为它们说的不是同一件事：
    #
    #   分支名说的是「现在缺什么」   llm_analysis、human_review、apply、calculate
    #   节点名说的是「谁来干这件事」 llm_rule_analysis、human_review、apply_memory_rules
    #
    # 改成分支名就是节点名当然能跑（LangGraph 允许省略映射表），
    # 但那样「Router 在决定什么」就被节点名盖住了 ——
    # 想给一个分支换节点时，得连 Router 的返回值一起改。
    #
    # human_review 现在有两个入口：
    #   从 llm_rule_analysis 来（LLM 刚提了候选，一定要过人）
    #   从 memory_lookup 直接来（历史上有 pending/rejected，没人可问 LLM 的）
    # 两处都指向同一个节点，因为它要做的事是同一件。
    # calculate 同理，多了一个来自 memory_lookup 的入口（没规则可应用时）。
    graph.add_conditional_edges(
        "memory_lookup",
        route_after_memory_lookup,
        {
            "llm_analysis": "llm_rule_analysis",
            "human_review": "human_review",
            "apply": "apply_memory_rules",
            "calculate": "calculate",
        },
    )

    # LLM 提完候选，**一定**要经过人。没有从 llm_rule_analysis 直通 apply 的边。
    #
    # 这不是「暂时还没加」，是设计上就不该有：
    # 那条边一旦存在，某天有人图省事把它接上，机器就能自己给自己定规则了，
    # 而且不会有任何报错 —— 流程照样跑通，只是规则开始悄悄长出来。
    graph.add_edge("llm_rule_analysis", "human_review")

    # 人做完决定，能算就走 apply，算不全就停。**不再是无条件边。**
    #
    # 人工批准的新规则是在 human_review_node 里追加进 memory_context 的，
    # 所以走 apply 时读到的是**批准之后**的完整列表，包括刚批的那条。
    # 这也是 apply 只有一处、没有环的原因 ——
    # 如果 apply 放在 human_review 之前，新批的规则就得再 apply 一遍，
    # 图里会出现 apply -> human_review -> apply 的环。
    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {"apply": "apply_memory_rules", "exception": "exception"},
    )

    graph.add_edge("apply_memory_rules", "calculate")

    # 这两条也没得选
    graph.add_edge("calculate", "validate")
    graph.add_edge("validate", END)

    graph.add_edge("exception", END)

    return graph.compile()


# ============================================================
# 跑一遍看看
# ============================================================


def run(file_path: str, decisions: list | None = None, trace: list | None = None) -> dict:
    """跑完整流程，返回最终 State。

    decisions 是人工决定的入口，格式同 initial_state —— 列表。
    不传就是「没人表态」—— 那 LLM 提的候选会全部保持 pending，不会被应用。

    trace 传一个列表进来，每个节点跑完会往里面追一条进出记录。
    """
    return build_graph(trace).invoke(initial_state(file_path, decisions))


if __name__ == "__main__":
    # 验证用：python payroll_agent.py [xlsx路径]
    import json

    # Windows 控制台默认按 GBK 输出，中文会变成乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    def show(title: str, file_path: str, decisions=None) -> None:
        print(f"\n{'=' * 60}")
        print(f"{title}")
        print(f"{'=' * 60}")

        # trace 记的是每个节点的进出，看它就知道流程走了哪条路、
        # 每个节点当时手上有什么 —— 不用重跑加 print 去猜。
        trace = []
        final = run(file_path, decisions, trace=trace)

        print("  ---- 节点轨迹 ----")
        for entry in trace:
            print(f"  [{entry['node']:18}] -> {str(entry['next_action']):20}")
            print(f"  {'':18}    进 {entry['input']}")
            print(f"  {'':18}    出 {entry['changed']}")

        print("\n  ---- 最终 State 摘要 ----")
        print(f"  parse_error      : {final.get('parse_error')}")
        print(f"  raw_rows         : {len(final.get('raw_rows', []))} 行")
        print(f"  extracted_rows   : {len(final.get('extracted_rows', []))} 行")
        print(f"  unknown_fields   : {final.get('unknown_fields')}")

        context = final.get("memory_context") or {}
        print(f"  memory_context   :")
        print(f"      approved_rules    : {context.get('approved_rules')}")
        print(f"      historical_rules  : {context.get('historical_rules')}")
        print(f"  unresolved_fields: {final.get('unresolved_fields')}")

        print(f"  llm_analysis     : {final.get('llm_analysis')}")
        print(f"  human_review     : {final.get('human_review')}")
        print(f"  new_memory_rules : {final.get('new_memory_rules')}")
        print(f"  halt_reason      : {final.get('halt_reason')}")
        print(f"  problems         : {final.get('problems')}")

        # 停了就说停了。不区分的话，「流程半路停了」和「算完了但一个人
        # 都没算出来」在输出上长得一模一样 —— 都是 0 人。
        # 这两种情况要处理的事完全不同，输出上就必须能分开。
        if final.get("halt_reason"):
            print("  calculation      : 没跑 —— 流程已经停了，见上面的 halt_reason")
            print("  validation       : 没跑")
            return

        calc = final.get("calculation_result") or {}
        print(f"  calculation      : {len(calc.get('people', []))} 人算出来了")

        print("  validation       :")
        print(
            json.dumps(
                final.get("validation_result"),
                ensure_ascii=False,
                indent=4,
            ).replace("\n", "\n  ")
        )

    target = sys.argv[1] if len(sys.argv) > 1 else "data/payroll_demo.xlsx"

    # 场景一：正常流程，走到 validation
    show("场景一：正常文件", target)

    # 场景二：文件不存在，在 parse 就拐去 exception
    show("场景二：文件不存在", "data/这张表不存在.xlsx")

    # 场景三：脏数据 —— problems 一路带到 State 末尾。
    #
    # 金额写成「面议」，extraction 把它当 0 算过去，同时记一条 problems。
    # 工资照样算得出来，validation 照样报「跟表里写的对不上」。
    # 这一场要看的是：那条 problems 到了终点还在，没被任何环节吞掉。
    #
    # 注意：这一场**验证不了** problems 的 reducer。
    # parser 也有一条「行比表头少 N 列」的 problem，但它走不到 ——
    # openpyxl 的 iter_rows 会按整表宽度把短行补齐 None，
    # parser 永远看不见「短行」长什么样。
    # reducer 是拿两个最小图单独做的隔离实验（见对话），不在这儿。
    import shutil
    import tempfile
    from pathlib import Path

    from openpyxl import Workbook

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_file = tmp_dir / "messy.xlsx"

    wb = Workbook()
    ws = wb.active
    ws.append(["工号", "姓名", "部门", "基本工资", "奖金", "扣款", "实发工资"])
    ws.append(["A001", "张伟", "研发部", "面议", 3000, 0, 11780])  # 金额不是数字
    ws.append(["A002", "李娜", "销售部", 8000, 5000])              # 比表头少 2 列
    wb.save(tmp_file)
    wb.close()

    try:
        show("场景三：脏数据 —— problems 一路带到终点", str(tmp_file))
    finally:
        # 临时文件用完就删，不留在 data/ 里污染项目
        tmp_file.unlink(missing_ok=True)
        tmp_dir.rmdir()

    # ---- 场景四：遇到没见过的列名 -> LLM 提候选 -> 人工批准 -> 应用 ----
    #
    # 这一场会**真的调一次 DeepSeek**（花几分钱），
    # 而且会把规则写进 memory.json —— 所以指向临时目录，不碰项目里那份。
    #
    # 同一个坑见 memory.py：python payroll_agent.py 跑起来时，
    # memory 模块是从磁盘另外导入的一份，改本文件的全局变量没用。
    import memory

    scenario4_dir = Path(tempfile.mkdtemp())
    memory.MEMORY_PATH = scenario4_dir / "memory.json"

    # 表里只有「季度激励」，没有「奖金」列。
    # 规则没建起来之前，bonus 缺失，calculation 会跳过这个人。
    s4_file = scenario4_dir / "quarterly.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["工号", "姓名", "部门", "基本工资", "扣款", "实发工资", "季度激励"])
    ws.append(["A001", "张伟", "研发部", 12000, 0, 11780, 3000])
    wb.save(s4_file)
    wb.close()

    print(f"\n\n{'#' * 60}")
    print("# 场景四：LLM 提议 + 人工批准")
    print("#" * 60)

    # 数「这一轮到底有没有再调 LLM」。
    #
    # 光看输出看不出来：第二次跑照样能算出工资，至于是命中了规则
    # 还是又问了一遍模型，两种结果的形状一模一样。所以要打桩数调用次数。
    #
    # 补丁必须打在**函数真正所在的那个模块**上。
    #
    # llm_rule_analysis_node 现在住在 llm_node.py 里，它调用
    # analyze_unknown_fields 时查的是 llm_node 模块的全局名 ——
    # 改 payroll_agent 里的同名变量它根本看不见（payroll_agent 里
    # 现在压根没有这个名字了）。
    #
    # 这是同一个坑第三次出现（memory.py 演示、上一版这里），
    # 规律记牢：**谁定义函数，就改谁模块里的全局名。**
    import llm_node

    calls = {"n": 0}

    def counting_run(file_path, decisions=None):
        original = llm_node.analyze_unknown_fields

        def wrapper(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        llm_node.analyze_unknown_fields = wrapper
        try:
            return run(file_path, decisions)
        finally:
            # finally 里还原：中途抛异常也必须还回去，
            # 否则后面几轮会一路数下去，数字全错。
            llm_node.analyze_unknown_fields = original

    # 第一遍：LLM 提候选并存成 pending，人工不表态 —— 什么都不会被批准
    print("\n>>> 4a. 人工不表态（decisions 空）")
    result_a = counting_run(str(s4_file))
    print(f"    调了几次 LLM : {calls['n']}")
    print(f"    LLM 提的候选 : {result_a['llm_analysis']}")
    print(f"    人工复核结论 : {result_a['human_review']['status']} / {result_a['human_review']['decision']}")
    print(f"    approved     : {result_a['memory_context']['approved_rules']}")
    print(f"    算出几个人   : {len(result_a['calculation_result']['people'])}")
    print("    -> 候选没被批准，季度激励没进 bonus，这个人算不出来")

    # 第二遍：人工批准。
    # memory 里已经有 pending 了，所以这一遍应该**一次 LLM 都不调** ——
    # 这正是 pending 这一态存在的意义。
    print("\n>>> 4b. 人工批准（decisions = [{季度激励: approve}]）")
    before = calls["n"]
    result_b = counting_run(
        str(s4_file),
        decisions=[{"field": "季度激励", "decision": "approve", "reason": "属于奖金类收入"}],
    )
    people = result_b["calculation_result"]["people"]
    print(f"    调了几次 LLM : {calls['n'] - before}   <- 应该 0")
    print(f"    approved     : {result_b['memory_context']['approved_rules']}")
    print(f"    算出几个人   : {len(people)}")
    for p in people:
        print(f"        {p['emp_id']} {p['name']}  bonus={p['bonus']}  net_pay={p['net_pay']}")
    print(f"    validation   : passed={result_b['validation_result']['passed']}")

    # 第三遍：规则已 approved，直接命中 apply 分支，同样不该调 LLM。
    print("\n>>> 4c. 再跑一遍（规则已入库）")
    before = calls["n"]
    result_c = counting_run(str(s4_file))
    print(f"    调了几次 LLM : {calls['n'] - before}   <- 应该 0")
    print(f"    approved     : {result_c['memory_context']['approved_rules']}")
    print(f"    unresolved   : {result_c['unresolved_fields']}")
    print(f"    new_rules    : {result_c['new_memory_rules']}   <- 空的是对的，这次没新批任何规则")
    print(f"    算出几个人   : {len(result_c['calculation_result']['people'])}")

    print("\n  ---- memory.json 里留下了什么 ----")
    print((scenario4_dir / "memory.json").read_text(encoding="utf-8"))

    shutil.rmtree(scenario4_dir, ignore_errors=True)

    # ---- 场景五：人工否决 -> 停下来，不算半张表 ----
    #
    # 这一场**不调 LLM**：memory 里先预置一条 rejected 记录，
    # memory_lookup 直接把它归进 historical_rules，
    # Router 走 human_review 分支，然后人明确再说一次「不要」。
    #
    # 要看的只有一件事：流程**没有**走到 calculate。
    # 表里有一个字段没有可用映射，那就一个人都不算 —— 这是设计，不是 bug。
    scenario5_dir = Path(tempfile.mkdtemp())
    memory.MEMORY_PATH = scenario5_dir / "memory.json"

    memory.save_candidate_rule("季度激励", "bonus", 0.95)
    memory.reject_rule("季度激励", "口径没和业务方确认，先别算")

    s5_file = scenario5_dir / "rejected.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["工号", "姓名", "部门", "基本工资", "扣款", "实发工资", "季度激励"])
    ws.append(["A001", "张伟", "研发部", 12000, 0, 11780, 3000])
    wb.save(s5_file)
    wb.close()

    show(
        "场景五：人工否决 -> 停下来（不算半张表）",
        str(s5_file),
        decisions=[
            {"field": "季度激励", "decision": "reject", "reason": "口径未确认"}
        ],
    )

    print("\n  ---- memory.json 里留下了什么 ----")
    print((scenario5_dir / "memory.json").read_text(encoding="utf-8"))

    shutil.rmtree(scenario5_dir, ignore_errors=True)
