"""run_summary.py —— 一次运行一行。

职责：把「这一趟跑得怎么样」收成**一条记录**，落到 runs/summary.jsonl。

    schema_version  task_id  start_time  end_time  status
    failed_node  error_type  fail_category
    llm_calls  input_tokens  output_tokens  review

它和 logger 的关系是**粒度**，不是内容：

    logger.py        一次运行 **N 行**（每个节点进出一行、任务级几行）
                     回答「按时间顺序发生了什么」
    run_summary.py   一次运行 **1 行**
                     回答「这一趟整体成没成、栽在哪」

    所以它放在 logger 的下游：summary 里每个字段 logger 都已经知道了，
    它不产生新事实，只是换个粒度收起来。翻日志本来也能拼出这几个字段 ——
    这个模块做的事就是别让人每次手工去翻。

刻意**不记**什么（这三条都是第一阶段就定下来的，不是这里新加的）：

    不记**完整 State**，也不记「State 快照的路径」。
        State 里装的是 raw_rows（姓名、身份证号、银行卡号）、
        calculation_result（工资明细）、llm_analysis。
        「把 State 写进文件、这里只存路径」和「把 State 写进这里」
        是同一件事换了个门 —— 只要那个文件存在，工资数据就落了盘。
        所以本阶段一条 State 都不落，这个字段**根本不存在**。

    不记**异常消息**，只记类型名（error_type）。
        异常消息里装什么由写 raise 的人决定，而这一层没有任何办法
        可靠判断哪条安全。截断不是脱敏。详细内容交给 logger 那一层 ——
        它同样只记类型名，两边口径一致（共用 logger.error_type）。

    不记 file_path。
        路径里可能带客户名（"XX公司_工资表.xlsx"）。要关联是哪张表，
        以后加的是文件名的哈希，不是路径本身。

关于这个文件本身：

    jsonl（一行一个 JSON 对象），不是一个大 JSON 数组。
    理由和 agent.log 一样：append 就够了，不用「读出整个文件 → 改 → 写回」，
    进程被掐也只是少最后一行，不会让整个文件变成坏 JSON。

    和日志一样是**证据**，不是账本 —— 缺一行不代表那次运行没发生。

---- V2 加的四个字段，各自的理由 ----

加字段这件事本身要付代价：schema 一旦变了，**老行没有新字段**，
所有读它的人都得防一手。所以每加一个都得说清楚为什么值这个代价。

    schema_version
        这一行是**哪一代格式**写的。老行根本没有这个键。

        为什么放在**每一行**上，而不是文件开头加一行标记：
        读的人读的是「最近 N 条」（见 read_runs），
        文件开头那行标记根本不在窗口里 —— 那时它等于不存在。
        版本必须跟着数据走，不能跟着文件走。

    fail_category
        failed_node + error_type 说清了「栽在哪」和「什么类型」，
        但答不了人真正要做的第一个决定：**找谁**。
        四种，见下面的常量。粗粒度是故意的 —— 十种分类在几十条
        数据上每种就一两条，分完还是看不出规律，只是把噪声换了个摆法。

    review
        人工复核的结果：批了哪些字段、否了哪些、哪些没人管。
        没有它，「这个字段老被否决」这类结论**在数据上不可能得出** ——
        agent.log 只知道走到了 human_review 这个节点，不知道结论是什么。

        装的是**字段名**（列名），不含金额和人员信息。
        「不记业务数据值、只记键名」这条口径没变。

    （llm_calls 三个第一阶段就有了，不是这次加的。）
"""

import datetime
import json
import logging
from collections import deque
from pathlib import Path

import logger


# 存储格式版本。**只在加字段的时候往上加一**，不是每次改动都加。
#
# 它回答的是「这批行能不能一起算」：
#     v1 的行没有 fail_category / review / schema_version
#     v2 的行有
# 把两代混在一起统计，得到的平均值不属于任何一代。
#
# 注意：老行**没有这个键**，不是值为 1。
# 所以读的时候用 record.get("schema_version")，拿到 None 就是 v1。
SCHEMA_VERSION = 2


# 一条记录有哪些字段。就是 write() 里那个字典的键。
#
# 单独列一份，是给 normalize() 用的 —— 它拿这张表去补老行缺的键。
# 改了 write() 记得改这里；test_agent_logging 里有一条用例把这张表钉死了，
# 漏改会红，不会悄悄漂走。
RECORD_FIELDS = (
    "schema_version",
    "task_id",
    "start_time",
    "end_time",
    "status",
    "failed_node",
    "error_type",
    "fail_category",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "review",
)


# 落在项目根目录的 runs/，和 logger.py 并排。
#
# 不放进 logs/：logs/agent.log 是**事件流**（一次运行很多行、给人扫的），
# 这里是**运行记录**（一次运行一行、给人和脚本读的）。
# 两种东西放两个目录，ls 一眼就分得清；混在 logs/ 里，
# 以后写脚本的人会以为它只是「另一种格式的日志行」。
SUMMARY_FILE = Path(__file__).parent / "runs" / "summary.jsonl"


# 一次读多少条运行记录。
#
# n = 30 是这么定的：
#
#   下限  少于 20 条，「占比」「平均」这类话就不该说出口 ——
#         20 条里差 1 条，占比就跳 5 个百分点，那不是规律是抖动。
#
#   上限  读得太远没意义。几周前那批行可能根本是另一个版本的代码写的，
#         和今天的混在一起算平均，等于把两个时代的数据搅成一锅。
#
# 30 是两头中间留的余量。要看更远，显式传更大的 limit ——
# 而不是让这个默认值悄悄变大。
DEFAULT_WINDOW = 30


# ---- 失败分类 ----
#
# 粗粒度，只看「该找谁」。人拿到一条失败记录，第一个要做的决定就是这个：
#
#     input_error      找客户/业务方 —— 表本身就不行（文件打不开、表头认不出）
#     business_halt    找业务方     —— 流程没坏，是人工否决了字段，等口径
#     llm_error        找外部       —— 模型那一环超时 / 返回垃圾
#     agent_error      找自己       —— 剩下的，基本就是我们自己的 bug
CAT_INPUT = "input_error"
CAT_BUSINESS_HALT = "business_halt"
CAT_LLM_ERROR = "llm_error"
CAT_AGENT_ERROR = "agent_error"

FAIL_CATEGORIES = (CAT_INPUT, CAT_BUSINESS_HALT, CAT_LLM_ERROR, CAT_AGENT_ERROR)

# LLM 那个节点的名字。
#
# 和 payroll_agent.build_graph 里 add_node 的名字**必须一致** ——
# 改那边记得改这里。
#
# 为什么不从 payroll_agent import：那个模块 import 了本模块，
# 反过来 import 会成环。所以这里自己声明一遍。
# （和 validation.REQUIRED_INPUTS 声明一遍自己是同一个路子：
#   与其搞个共享常量模块，不如让两边各自说清楚「我认为它叫什么」，
#   哪天真改了名字，两边对不上是**看得见**的。）
LLM_NODE = "llm_rule_analysis"


def write(
    task_id: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    status: str,
    error=None,
    *,
    fail_category: str | None = None,
    review: dict | None = None,
) -> None:
    """记下一次运行的结果。

    failed_node 不用传 —— 从 logger 的执行流上取（谁记的 node_failed，
    谁就把节点名留在了那里，见 logger.last_failed_node）。

    fail_category 和 review 由 run() 传进来，不从执行流上取。
    理由：这两个都不是「节点留下的痕迹」，是 **run() 才知道的结论** ——
    fail_category 要同时看 parse_error 和 halt_reason 才判得出来，
    review 更是只有跑完之后才存在。执行流上没地方挂它们，
    硬挂上去等于让 logger 去理解业务。

    **这个函数不会抛异常。** 理由有两条，第二条才是要紧的：

        1. 它和日志出口是同一类东西：记录功能坏掉，不该把工资流程带崩。
        2. 调用它的一处在 run() 的 except 分支里，写完紧接着就是 raise。
           如果这里抛了，抛出来的会是个 OSError，**把原来那个业务异常顶掉** ——
           那就成了「为了记一次失败，把失败的原因弄丢了」。

    写不进去的时候往 logger 里记一条 WARNING：不崩，但也不装作没发生。
    """
    usage = logger.llm_usage()

    record = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "start_time": start_time.strftime(logger.DATE_FORMAT),
        "end_time": end_time.strftime(logger.DATE_FORMAT),
        "status": status,
        "failed_node": logger.last_failed_node(),
        "error_type": logger.error_type(error) if error is not None else None,
        # 该找谁。None = 这次没失败。
        # 成功却填了分类，会让人以为「成功了但属于 input_error」—— 所以只有失败才填。
        "fail_category": fail_category,
        # LLM 用量。同样从执行流上取，不由 run() 逐个传。
        #
        # llm_calls 和两个 token 数要**一起看**：
        #     calls=0                 压根没走过 LLM（列全认识）
        #     calls=1, tokens=0       调了，但没拿到用量（失败 / 服务端没返回）
        #     calls=1, tokens=1280    正常
        # 只看 token 的话，中间那种和第一种分不开 —— 失败看起来是免费的。
        "llm_calls": usage["llm_calls"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        # 人工复核的结论。None = 这次压根没走到复核那一步。
        "review": review,
    }

    try:
        SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SUMMARY_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        # 这里只能靠 logger 说出来 —— summary 自己的出口就是坏掉的那个。
        logger.log_event(
            logger.SUMMARY_WRITE_FAILED,
            "运行记录没写进去",
            task_id=task_id,
            error=exc,
            level=logging.WARNING,
        )


def classify(
    *, parse_error=None, halt_reason=None, failed_node=None
) -> str | None:
    """这次失败该归哪一类。没失败就是 None。

    判定是**有先后**的，不是并列的 —— 因为几种情况会同时成立：

        文件不存在时 parse_error 和 halt_reason 都有值。
        它既是「输入不行」，也是「流程停住」。二选一必须选一个。

    顺序，和顺序的理由：

        1. parse_error  -> input_error
           输入这一关就没过。哪怕流程也因此停住了，根因还是在输入。
           归成 business_halt 的话，看报告的人会去找业务方问口径 ——
           而其实该做的事是去问客户「你这张表为什么打不开」。

        2. halt_reason  -> business_halt
           文件读得进来、字段认得出来，是业务规则上该停
           （典型：人工否决了某个字段）。
           这种不该改代码，该去要一个业务口径。

        3. failed_node 是 LLM 那一步 -> llm_error
           走到这里说明前两关都过了，是外部模型这一环崩的。
           归成 agent_error 的话，报告会让人去翻我们自己的代码 ——
           而其实该做的是加超时和重试。

        4. 剩下的 -> agent_error
           排除了输入、业务、外部之后，剩下的就是我们自己的问题。
           这个桶兜底，所以前三条必须**排干净** ——
           一旦有一条判歪了，它不会消失，只会伪装成「我们的 bug」，
           让人花一整天去查一个根本不在代码里的问题。
    """
    if parse_error:
        return CAT_INPUT

    if halt_reason:
        return CAT_BUSINESS_HALT

    if failed_node is None:
        # 没抛异常，也没停 —— 这次是成功的。
        return None

    if failed_node == LLM_NODE:
        return CAT_LLM_ERROR

    return CAT_AGENT_ERROR


def review_record(*, approved, rejected, undecided, reviewer=None) -> dict:
    """把 human_review 里的三个名单收成 summary 的一格。

    只挑**字段名**，不挑金额也不挑人 —— 和 logger 那三道闸门的口径一致。

    candidates 是三者之和，但**单独存一份**而不是让读的人自己加：
    它是「这一趟到底有没有活要人干」的直接答案，
    读的人第一眼要的就是它，不该逼他去算一个加法。
    """
    approved = list(approved or [])
    rejected = list(rejected or [])
    undecided = list(undecided or [])

    return {
        "candidates": len(approved) + len(rejected) + len(undecided),
        "approved": approved,
        "rejected": rejected,
        "undecided": undecided,
        "reviewer": reviewer or None,
    }


def read_all(path: Path | None = None) -> list:
    """把 summary.jsonl 读成一组 dict。文件不存在就是空列表。

    存在的理由是「有写就该有读」—— 没有读的地方，格式对不对没人知道，
    字段加错了也要等到真出事才发现。测试用的是这个函数。

    **原样返回，不补缺的字段**（和 read_runs 的差别就在这儿）。

    这不是「忘了 normalize」，是故意的 —— 两个读法服务两种目的：

        read_all   回答「磁盘上究竟写了什么」。
                   测试用它。补了缺字段的话，write() 哪天漏写一个，
                   normalize 会把它填成 None 补回去，字段表的断言照样绿 ——
                   **测试就废了**。

        read_runs  回答「这些数据能支撑什么结论」。
                   分析用它。老行缺字段是常态，补 None 才能和安全地读。

    **整个文件全读进来**，不做窗口限制。适合测试和排查，
    不适合日常分析 —— 那用 read_runs。
    """
    target = path or SUMMARY_FILE
    if not target.exists():
        return []

    records = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def read_runs(limit: int = DEFAULT_WINDOW, path: Path | None = None) -> list:
    """读**最近** limit 条运行记录，按时间从旧到新返回。

    为什么是「最近 N 条」而不是「全部」：见 DEFAULT_WINDOW。

    用 deque(maxlen=limit) 而不是 read_all()[-limit:]：
    后者要先把整个文件读进内存再切。这个文件是只增不减的，
    跑上一年它就是几千行 —— 而每次分析只看得见最后 30 行，
    没必要为了这 30 行把前面全搬进来。

    老行会缺新字段（v1 的行没有 fail_category / review / schema_version）。
    这里**补 None，不补一个默认值**：

        补 0 的话，一条「当年还没记 llm_calls」的行会被算成
        「这次运行一次 LLM 都没调」—— 报告上看着一切正常，
        而那个结论是凭空造出来的。

        补 None 则读的人被迫分得清「没有这件事」和「不知道有没有这件事」，
        这正是本模块从一开始就在守的那条线（calls=0 和 tokens=0 要分开看）。

    坏行（半截 JSON）跳过，并往 logger 记一条 WARNING。
    **不静默跳过** —— 少一行统计就少一个样本，而少了一个样本，
    读报告的人看不出来，只会以为样本本来就那么多。
    """
    target = path or SUMMARY_FILE
    if not target.exists():
        return []

    with open(target, encoding="utf-8") as handle:
        lines = deque((line for line in handle if line.strip()), maxlen=max(limit, 0))

    records = []
    for line in lines:
        try:
            records.append(normalize(json.loads(line)))
        except json.JSONDecodeError as exc:
            logger.log_event(
                logger.SUMMARY_READ_SKIPPED,
                "运行记录里有一行读不出来，已跳过",
                error=exc,
                level=logging.WARNING,
            )

    return records


def normalize(record: dict) -> dict:
    """把一条记录补成当前字段表的样子。缺的键补 None。

    只对**已知字段**补。记录里多出来的键原样留着 ——
    那可能是别人手工加的排查信息，删掉等于毁证据。
    """
    filled = dict(record)
    for field in RECORD_FIELDS:
        filled.setdefault(field, None)
    return filled


if __name__ == "__main__":
    # 跑一遍看看：python run_summary.py
    #
    # 直接在项目里的 runs/summary.jsonl 追加两条假的 —— 不读真表、不跑图。
    print(f"这份文件会写到：{SUMMARY_FILE}")
    print("（下面是演示，会真的往那个文件追加两行）")
    print()

    now = datetime.datetime.now()

    # 成功的一次
    write(
        logger.new_task_id(),
        now,
        now,
        status=logger.SUCCESS,
    )

    # 失败的一次：走 logger 的执行流，failed_node 才有人填
    with logger.task_scope(logger.new_task_id()):
        logger.node_failed(
            "calculate",
            message="节点抛出异常",
            error=ValueError("张三 的工资 12000 元算不出来"),
        )
        write(
            logger.current_task_id(),
            now,
            now,
            status=logger.FAILED,
            error=ValueError("张三 的工资 12000 元算不出来"),
        )

    print(f"现在文件里有 {len(read_all())} 行。最后两行：")
    for record in read_all()[-2:]:
        print("  " + json.dumps(record, ensure_ascii=False))
