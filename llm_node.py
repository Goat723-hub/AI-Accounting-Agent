"""llm_node.py —— 语义分析员，整条链上唯一用 LLM 的节点。

一个文件干完五件事：

    1. 从 State 里取「要分析哪些字段」和判断用的上下文
    2. 拼 prompt
    3. 调 DeepSeek
    4. 解析 + 校验模型的返回
    5. 返回要写回 State 的局部更新

为什么全放一个文件，不拆成 service / node 两层：

    拆开之后「LLM 在哪」就有两个答案（服务层在哪定义、节点在哪调用），
    读代码的人要先跳一次。这条链上只有这一个节点用 LLM，
    收在一个文件里，「整条链只有这一处碰模型」是打开文件就看得见的事实，
    不是要靠翻 import 才能确认的结论。

    文件内部仍然分段：节点在最上面，纯函数在下面。
    纯函数（_parse_candidates 那一段）不依赖 State，可以单独 import 出来测 ——
    `python llm_node.py` 跑的就是那一层，不需要网络、不花钱。

这个节点**只有提议权**：

    不 approve、不 reject、不改 extracted_rows、不碰 calculation / validation、
    不决定下一步去哪（那是 Router 的事）。

    唯一的出口是一个列表，交给 Human Review 决定要不要变成规则。

关于「不修改 Memory」这条约束，有一处要说清楚：

    这个节点会调 save_candidate_rule()，把候选**记成 pending**。

    这不是越界，因为 pending 不是规则：机器用不了它
    （query_rule 默认只返回 approved，apply_rules 拿到的是 approved 列表），
    它对工资的计算结果没有任何影响。真正能改变结果的是 approve_rule，
    那一个仍然只有人能调。

    存它的理由只有一条：不存的话，下一次跑同一个字段还是 unresolved，
    Router 还是会把它送到这里 —— 那个字段**每跑一次就烧一次钱**，
    而且每次 confidence 可能不一样，人工审的时候看到的是哪一版全凭运气。

    所以这里做的是「把 LLM 说过的话记下来」，不是「让 LLM 的话生效」。

为什么整条链只有这里用 LLM：

    认列名是**语义**问题 —— 「季度激励」算不算奖金，靠规则表写不出来，
    靠人一条条加又太慢。这是 LLM 真正擅长、且别的办法做不到的地方。

    而「认出来之后算不算数」是**责任**问题，不是能力问题。
    所以映射一旦定下来，后面的计算、校验全程不用 LLM ——
    同样的输入必须永远得到同样的工资，这一条不能被模型的随机性破坏。

为什么输出要过一遍白名单：

    提示词里会写「suggested_mapping 必须是这 7 个之一」，
    但**提示词不是安全边界**。它是一段建议，不是一道约束。
    模型完全可能返回一个不存在的字段名，而那个值会一路存进 memory.json，
    之后每张表都按它算，而且没人知道是什么时候错的。

    所以 _parse_candidates 那一层不是格式转换，是**闸门**。
"""

import json
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from extraction import FIELD_MAP, UNMAPPED_KEY
import logger
from memory import save_candidate_rule


if TYPE_CHECKING:
    # 只为类型标注导入，运行时**不执行**。
    #
    # 写成真 import 的话就是循环引用：payroll_agent 要 import 本模块的节点，
    # 本模块又要 import payroll_agent 的 PayrollState —— 两个模块在
    # 各自的顶部互相等对方先加载完，Python 会直接报 ImportError。
    #
    # TYPE_CHECKING 里的东西只在做类型检查（IDE、mypy）时存在，
    # 所以既能给 state 标上类型，又不会在运行时绕回来。
    from payroll_agent import PayrollState


# DeepSeek 的 OpenAI 兼容端点。换模型只改这三行。
#
# 带 /v1 是 OpenAI SDK 的惯例路径，DeepSeek 两种都收。
# 不动它是因为它现在跑得通 —— 换 URL 属于「没出问题就别碰」。
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

# 每次最多取几个样本值给 LLM 看。5 个足够看出这一列是钱、是编号还是文本。
MAX_SAMPLES_PER_FIELD = 5

# 调用超时（秒）。LLM 卡住不能把整条工资流程一起卡住。
REQUEST_TIMEOUT = 30


SYSTEM_PROMPT = """你是工资表字段映射助手。

标准字段只有这 7 个：

  emp_id           工号 / 员工编号 / 员工号
  name             姓名 / 员工姓名
  department       部门 / 所属部门
  base_salary      基本工资 / 固定薪资 / 底薪
  bonus            奖金类**收入**：绩效、津贴、补贴、提成、激励
  deduction        扣款类**支出**：罚款、事假扣款、代扣
  net_pay_in_file  表里直接写死的「实发工资」那一列

你的任务：给每个待识别字段选一个标准字段，填进 suggested_mapping。

注意这个字段叫 **suggested_**mapping —— 字段名里的 suggested 是认真的。
你给的是**建议**，不是决定。这份建议会交给人工复核，
由人决定要不要采纳。所以：

  - 不要因为自己很有把握就跳过人工
  - 不要把 confidence 写成 1.0 来表示「这个肯定对」
  - 判断不了就干脆不要给这条候选，不要"给一个试试"

判断原则：

1. 看字段名的同时**必须看样本值**。同一个词在不同公司含义可能相反，
   但值通常不会骗人 —— 都是正数、量级和大额补贴相当，多半是收入。

2. 判断不了就**不要给这条候选**。宁可不提，也不要猜。
   猜错的代价不是「这一列没算」，而是整张表按错的规则算工资，
   而且校验也发现不了 —— 校验对的是「算得对不对」，不是「字段认得对不对」。

3. 已经有历史记录的字段（标了 pending 或 rejected 的），
   说明以前在这个字段上做过判断。不要给出和 rejected 完全一样的结果。

你不负责的事（做了也没用，返回里没有对应的位置）：

  - 不要执行任何工资计算
  - 不要修改规则，也不要判断规则该不该生效
  - confidence 高不代表自动批准

只输出 JSON，不要 markdown 代码块，不要任何解释文字。"""


# ============================================================
# 节点 —— LangGraph 调用的入口
# ============================================================


def llm_rule_analysis_node(state: "PayrollState") -> dict:
    """让 LLM 给没见过的字段提候选映射。返回 State 的局部更新。

    输入从哪来（全部来自 State，不读磁盘、不读全局变量）：

        待分析字段  <- state["unresolved_fields"]
                       memory 说「这些字段我一条记录都没有」
        样本值      <- state["extracted_rows"] 里每行的 unmapped
        历史规则    <- state["memory_context"]["historical_rules"]
                       以前提过什么、被否过什么

    为什么字段要用 unresolved_fields 而不是 unknown_fields：

        unknown_fields 是 extraction 说「这一列我不认识」——**原始事实**。
        unread 的字段里，有一部分 memory 已经记过了（pending 或 rejected），
        那些不该再问一遍 LLM，该直接找人。

        unresolved_fields 是 memory 筛过一轮之后的**结论**：
        「不认识的列里，我也没有记录的」。这才是要问模型的。

    为什么要把历史规则喂给 LLM：

        人工否过「交通费 -> bonus」，LLM 不知道的话下一轮还会再提一次。
        这不是让它「学习」，是把它上次没看到的信息补给它 ——
        规则本身仍然只存在 memory.json 里，LLM 没有自己的记忆。

    **这个节点不改 extracted_rows。**

        extracted_rows 只有 apply_memory_rules 能动。

    为什么 LLM 调用失败不抛异常：

        见 analyze_unknown_fields —— 它永远返回结构，失败信息进 problems。
        网络抖动不该让整张工资表算不出来，没认出来的字段继续留在
        unresolved_fields 里，下次再说。
    """
    field_names = state["unresolved_fields"]

    result = analyze_unknown_fields(
        unknown_fields=field_names,
        samples=_collect_samples(state["extracted_rows"], field_names),
        historical_rules=state["memory_context"]["historical_rules"],
    )

    problems = list(result["problems"])

    for candidate in result["candidates"]:
        saved = save_candidate_rule(
            candidate["field"], candidate["mapping"], candidate["confidence"]
        )
        if not saved.get("saved"):
            # 存不下通常是因为这个字段已经有一条 approved 规则了。
            # 那不拦着 —— 候选还是交给人工看，只是下次可能还要再问一次 LLM。
            #
            # 这里**不看返回值也不重试**：save 成不成功只决定
            # 「这次有没有记进 memory」，不决定「候选能不能给人看」。
            # 记不进去的原因要让人看见，所以就记一条 problem。
            problems.append(
                f"{candidate['field']} 的候选存进 memory 失败："
                f"{saved.get('reason')}（本次仍交给人工复核）"
            )

    return {
        "llm_analysis": result["candidates"],
        # LLM 调用失败、返回乱码、编了个不存在的标准字段 —— 全进 problems。
        # 它没给出结果时绝不替它编一条，它给了结果时也绝不无条件采信。
        "problems": problems,
        # 走到哪由 Graph 的边决定（llm_rule_analysis -> human_review 是无条件边）。
        # 这里写下来只是为了日志好看，不是路由依据。
        "next_action": "human_review",
    }


# ============================================================
# 纯函数 —— 不依赖 State，可以单独测
# ============================================================


def analyze_unknown_fields(
    unknown_fields: list,
    samples: dict,
    historical_rules: list,
) -> dict:
    """给没认出来的字段提候选映射。

    输入：
        unknown_fields:    ["季度激励", "房补"]
        samples:           {"季度激励": [5000, 5000, 0], "房补": [800]}
        historical_rules:  [{"field": ..., "mapping": ..., "status": ...}, ...]

    输出：
        {
            "candidates": [
                {"field": "季度激励", "mapping": "bonus",
                 "confidence": 0.96, "reasoning": "..."},
            ],
            "problems": ["..."],
        }

    注意 candidates 里的键是 **mapping**（不是 suggested_mapping）——
    模型用的是 suggested_mapping，这里是翻译之后的结果。翻译就发生在
    _parse_candidates 那道闸门里，因为它顺带要检查这个值合不合法，
    校验和翻译放在一起才不会两边对不上。

    这个函数**不抛异常**，和 parser.read_payroll_excel 一样：
    调用 LLM 失败是「这次没分析出来」，不是程序错误。
    抛出去会把整张图炸掉，而出去了的字段一个也回不来。
    """
    if not unknown_fields:
        return {"candidates": [], "problems": []}

    client = _build_client()

    if client is None:
        # 没有 key 就明确报告，**绝不**退化成「按字段名猜一个」。
        # 一个来源不明的兜底规则比没有规则危险得多：
        # 没有规则时人还能看见 unknown_fields，有规则就什么都看不见了。
        return {
            "candidates": [],
            "problems": ["LLM 不可用（没读到 DEEPSEEK_API_KEY），本次跳过语义分析"],
        }

    try:
        response = client.invoke(
            _build_prompt(unknown_fields, samples, historical_rules)
        )
        raw = response.content
    except Exception as e:
        # 网络断了、超时、key 没权限、服务端 5xx —— 全都到这儿。
        #
        # 记类型名不只记 message：DeepSeek 返回的 401 和网络超时
        # 都是「调用失败」，但处理方式完全不同（一个去查 key，一个等会儿重试）。
        #
        # 这也算「调用过一次」：不带用量地记一笔。否则「没调用」和
        # 「调了但失败」在费用上都表现为 0，失败就看起来是免费的了。
        logger.note_llm_call()
        return {"candidates": [], "problems": [f"LLM 调用失败：{type(e).__name__}: {e}"]}

    logger.note_llm_call(**_usage_of(response))
    return _parse_candidates(raw, unknown_fields)


def _usage_of(response) -> dict:
    """从响应里取 token 用量，取不到就返回空 dict。

    返回空 dict 而不是 {"input_tokens": 0}：note_llm_call 收到 None 记的是
    「没拿到」，收到 0 记的是「拿到了，是 0」—— 这两种在这个项目里必须分开。

    为什么用 getattr 而不是直接 response.usage_metadata：
    换个 provider、或者 langchain 改了字段名，这里会 AttributeError ——
    而少记一个数不该让整张工资表算不出来。取不到就当没拿到。
    """
    usage = getattr(response, "usage_metadata", None) or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }


def _build_client():
    """建 DeepSeek 客户端。拿不到 key 就返回 None。

    每次调用都 load_dotenv() 而不是在模块顶部做一次：
    模块导入时机和「用户配好 key」的时机不一定一致，
    在导入时读，key 后配的话要重启进程才生效，很难排查。

    key 从环境变量读，绝不写进代码 —— 写进代码的 key 会跟着
    这个文件进版本库、进截图、进简历附件，而且换 key 要改代码。
    """
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return None

    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        # 这是映射任务不是创作任务。temperature=0 让它尽量可复现，
        # 但**不要**因此认为它确定性 —— 模型输出永远要过白名单。
        temperature=0,
        timeout=REQUEST_TIMEOUT,
        max_retries=1,
    )


def _build_prompt(unknown_fields: list, samples: dict, historical_rules: list) -> list:
    """拼 prompt。

    分成三段：待识别字段（带样本值）、历史记录、输出格式。
    样本值和历史记录都是**证据**，不给证据就等于让模型编。

    为什么不把整个 State 或者源码丢给它：

        喂进去的东西越多，模型越容易顺着无关信息编。
        判断「这一列是钱还是编号」需要的全部信息就是
        列名 + 几个真实的值 + 允许的标准字段表 —— 别的都是噪音。

        而且发出去的数据越少，出问题时越清楚是什么泄露了。
    """
    parts = ["## 待识别字段\n\n"]

    for field in unknown_fields:
        values = samples.get(field) or []
        if values:
            shown = "、".join(repr(v) for v in values[:MAX_SAMPLES_PER_FIELD])
        else:
            # 说清楚「没取到值」而不是留空 ——
            # 留空的话模型会把「没有值」当成「值是空的」，给出错误推断。
            shown = "（这一列全是空的，没有任何值）"
        parts.append(f"- {field}\n  样本值：{shown}\n")

    if historical_rules:
        parts.append("\n## 历史记录\n\n")
        parts.append("这些字段以前有人看过，括号里是当时的结论：\n\n")
        for rule in historical_rules:
            parts.append(
                f"- {rule['field']} -> {rule['mapping']}（{rule['status']}）\n"
            )

    parts.append("\n## 输出格式\n\n")
    parts.append("严格按这个结构输出，只输出 JSON：\n\n")
    parts.append(
        '{"candidates": [{"field": "字段名", "suggested_mapping": "标准字段名", '
        '"confidence": 0.0, "reasoning": "一句话理由"}]}\n'
    )
    parts.append("\n判断不了的字段直接不要放进 candidates，不要为它编一条。\n")

    return [("system", SYSTEM_PROMPT), ("human", "".join(parts))]


def _parse_candidates(raw, unknown_fields: list) -> dict:
    """解析 + 校验 LLM 的输出。

    这是安全边界。下面每一条 if 都对应一种「模型会干但绝不能让它过去」的情况：

        带 markdown 代码块             -> _strip_code_fence
        不是合法 JSON                  -> 丢弃，报 problem
        顶层不是对象 / 没有 candidates -> 丢弃，报 problem
        提了一个根本没问它的字段       -> 丢弃
        编了一个不存在的标准字段       -> 丢弃
        缺 suggested_mapping           -> 丢弃
        confidence 不是 0~1 的数       -> 丢弃

    全部丢弃而不是「修一修」：这里没有能力判断哪条该修，
    猜错一次就是一条错误规则进了 memory.json，而它会一直生效下去。

    注意丢弃是**可追踪**的 —— 每条都写进 problems，
    所以「LLM 提了但被挡掉」这件事看得见，不是静默消失。
    """
    text = _strip_code_fence(raw)

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        # 模型返回空串、返回一段中文、返回半截 JSON —— 都到这儿。
        # 空串单独说，因为它的成因通常是「模型拒答」或者调用被截断，
        # 和「返回了乱码」不是一回事，排查方向不同。
        if not str(text).strip():
            return {"candidates": [], "problems": ["LLM 返回了空内容"]}
        return {"candidates": [], "problems": [f"LLM 返回的不是合法 JSON：{e}"]}

    if not isinstance(data, dict):
        return {"candidates": [], "problems": ["LLM 返回的 JSON 顶层不是对象"]}

    raw_candidates = data.get("candidates")
    if not isinstance(raw_candidates, list):
        return {"candidates": [], "problems": ["LLM 返回里没有 candidates 列表"]}

    allowed_fields = set(unknown_fields)
    candidates, problems = [], []

    for item in raw_candidates:
        if not isinstance(item, dict):
            problems.append(f"丢弃一条候选：不是对象（{item!r}）")
            continue

        field = str(item.get("field", "")).strip()

        if field not in allowed_fields:
            # 模型偶尔会「顺手」把已经识别出来的字段也报一遍，
            # 或者拼错字段名。两种都不能要。
            problems.append(f"丢弃候选：{field!r} 不在本次待识别字段里")
            continue

        mapping = str(item.get("suggested_mapping", "")).strip()

        if not mapping:
            # 键名写错（比如写回 mapping）也会落到这儿。
            # 严格要求键名不是吹毛求疵：两个键名并存的话，
            # 下一个读这段代码的人得先想清楚哪个是哪个，
            # 而想错一次就是一个字段被静默漏掉。
            problems.append(f"丢弃候选 {field}：没有 suggested_mapping")
            continue

        if mapping not in FIELD_MAP:
            problems.append(f"丢弃候选 {field} -> {mapping!r}：不是标准字段")
            continue

        confidence = _as_confidence(item.get("confidence"))
        if confidence is None:
            problems.append(
                f"丢弃候选 {field}：confidence {item.get('confidence')!r} 不是 0~1 的数"
            )
            continue

        candidates.append(
            {
                "field": field,
                # 模型说 suggested_mapping，下游一律叫 mapping ——
                # 名字对齐 memory.save_candidate_rule 的参数，
                # 翻译只在这一处发生。
                "mapping": mapping,
                "confidence": confidence,
                "reasoning": str(item.get("reasoning", "")).strip(),
            }
        )

    return {"candidates": candidates, "problems": problems}


def _strip_code_fence(text) -> str:
    """剥掉 ```json ... ``` 外壳。

    提示词里明确说了不要 markdown，模型还是经常加。
    这不是「容忍错误」，是知道对方有这个固定习惯、提前处理掉 ——
    为一个可预期的格式习惯让整条流程失败，不划算。
    """
    text = ("" if text is None else str(text)).strip()

    if not text.startswith("```"):
        return text

    lines = text.splitlines()[1:]  # 去掉 ```json 那一行
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def _as_confidence(value):
    """confidence 必须是 0~1 之间的数，否则返回 None。

    bool 是 int 的子类，不挡的话 True 会被当成 1.0 混过去。

    字符串 "0.96" 也一律拒绝 —— 严格一点只会丢一条候选（而且记进了
    problems，人看得见），放宽一点就可能让一个口径不一致的值
    混进 memory.json，之后再没人分得清哪条是可靠的。
    """
    if isinstance(value, bool):
        return None

    if not isinstance(value, (int, float)):
        return None

    if not 0.0 <= float(value) <= 1.0:
        return None

    return float(value)


def _collect_samples(rows: list, fields: list) -> dict:
    """给每个未知字段收集几个样本值。

    为什么要给 LLM 看值：
        光看「季度激励」四个字，它可能猜奖金，也可能猜扣款。
        看到值是 5000、5000、0 这种正数、量级和奖金相当，基本就定了。

        值本身是**证据**。不给证据就让它猜，等于让它编。

    只取前几个：判断「这是钱还是编号还是文本」不需要看完整列。
    """
    samples = {field: [] for field in fields}

    for row in rows:
        unmapped = row.get(UNMAPPED_KEY) or {}
        for field in fields:
            if field in unmapped and len(samples[field]) < MAX_SAMPLES_PER_FIELD:
                samples[field].append(unmapped[field])

    return samples


if __name__ == "__main__":
    # 单独验证用：python llm_node.py
    #
    # 只测**不需要网络**的那一层 —— 解析和闸门。
    # 真正调模型的部分放在 payroll_agent.py 的端到端里跑，
    # 因为那一层要花钱，不该每次改代码都跑一遍。
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("== 闸门测试：下面每一条都应该被挡掉，并且记进 problems ==\n")

    cases = [
        (
            "带 markdown 代码块（应该被剥掉，正常通过）",
            '```json\n{"candidates": [{"field": "季度激励", "suggested_mapping": "bonus",'
            ' "confidence": 0.96, "reasoning": "额外收入"}]}\n```',
        ),
        (
            "根本不是 JSON",
            "季度激励应该是奖金吧",
        ),
        (
            "返回空内容",
            "",
        ),
        (
            "编了一个不存在的标准字段",
            '{"candidates": [{"field": "季度激励", "suggested_mapping": "工资", "confidence": 0.9}]}',
        ),
        (
            "提了一个没问它的字段",
            '{"candidates": [{"field": "姓名", "suggested_mapping": "name", "confidence": 0.9}]}',
        ),
        (
            "用了旧键名 mapping 而不是 suggested_mapping",
            '{"candidates": [{"field": "季度激励", "mapping": "bonus", "confidence": 0.9}]}',
        ),
        (
            "confidence 是字符串",
            '{"candidates": [{"field": "季度激励", "suggested_mapping": "bonus", "confidence": "很高"}]}',
        ),
        (
            "confidence 超出范围",
            '{"candidates": [{"field": "季度激励", "suggested_mapping": "bonus", "confidence": 1.5}]}',
        ),
        (
            "confidence 是 true（bool 会被当成 1.0 混过去）",
            '{"candidates": [{"field": "季度激励", "suggested_mapping": "bonus", "confidence": true}]}',
        ),
        (
            "一条好的 + 一条坏的",
            '{"candidates": ['
            '{"field": "季度激励", "suggested_mapping": "bonus", "confidence": 0.96, "reasoning": "额外收入"},'
            '{"field": "房补", "suggested_mapping": "不存在的字段", "confidence": 0.9}]}',
        ),
    ]

    for title, raw in cases:
        result = _parse_candidates(raw, ["季度激励", "房补"])
        print(f"-- {title}")
        print(f"   通过 {len(result['candidates'])} 条：{result['candidates']}")
        for p in result["problems"]:
            print(f"   挡掉：{p}")
        print()

    print("== 不花钱的异常路径：拿不到 key 时应该报 problem，而不是抛异常 ==\n")

    def without_key():
        """打桩 _build_client 来模拟「没有 key」。

        为什么不直接删掉环境变量里的 key：
            _build_client 每次调用都跑一次 load_dotenv()，
            它会立刻把 .env 里的 key 重新读回环境变量 ——
            删了也白删，测出来的是假绿。

        为什么包成函数而不是在 __main__ 里直接 global：
            Python 要求 global 声明出现在同一作用域里对该名字的
            **任何使用之前**。写在 __main__ 里就得小心翼翼安排顺序，
            包一层之后 global 就是这个函数的第一句，没有顺序问题。
        """
        global _build_client

        real = _build_client
        _build_client = lambda: None
        try:
            return analyze_unknown_fields(["餐补"], {"餐补": [300]}, [])
        finally:
            _build_client = real

    print(f"   {without_key()}")
    print("\n   ^ 候选是空的、报了一条 problem、没有抛异常 —— 这就是要的行为。")
