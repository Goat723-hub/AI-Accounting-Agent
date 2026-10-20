"""logger.py —— 统一日志员。

职责：给整条工作流提供**一个**记录运行事件的地方。

    哪个任务、哪个节点、什么时候、发生了什么、成了没有、失败的话是什么错。

调用方式是函数调用（log_event / node_started / ...），
不是网页，也不是 HTTP 接口。各个 Node 需要记录时调这里，
格式和出口由这一处统一管。

本模块有**两个出口**，它们不是同一件事的两种格式，是两种不同的读者：

    logs/agent.log      **事件流**，一次运行很多行，人扫的。
                        字段固定（task / node / event / status / error），
                        grep 方便 —— 「哪个节点出事了」在这里回答。

    logs/trace.jsonl    **节点轨迹**，一个节点一行 JSON，机器读的。
                        记的是节点进来时手上那个 State 的**形状**
                        （哪些键有内容、容器多大），不含任何值 ——
                        「它当时看到的是什么」在这里回答。

    两个文件靠 task_id 对齐，**结合着看**：agent.log 说「谁出了事」，
    trace.jsonl 说「它当时看到了什么」。

    分两个文件而不是合成一个，是因为读者不同：JSON 混进 agent.log，
    每行都要跟引号打架；事件流塞进 jsonl，就丢了一行扫过去的可读性。

    两个文件都有**恒定的大小上限**：超过 MAX_FILE_BYTES 就在两次运行之间
    滚一代，只留一代（见 rotate_if_needed）。不留的话，跑得越久越不敢删，
    最后变成「一个谁也不敢打开的四十几兆的文本文件」——
    那样的日志等于没有。

为什么用标准库 logging，不自己写个 open(..., "a") 的函数：

    因为「往哪写」和「记什么」是两件事。用 logging 的话，以后要加文件轮转、
    要接到别的采集端，换的是 handler，调用点一行都不用动。
    自己写死追加的话，那天得回头改所有调用点 —— 而调用点会越加越多。

刻意不做的：

    不生成 task_id（那是调用方的事，见文件末尾的「下一步」）、不重试、
    不改任何 Node 的业务逻辑、不做网页 / 数据库 / HTTP。

这个模块最重要的一条约束是**安全**：

    工资表项目里，日志最容易出事的方式不是格式写错，是有人图省事
    把整个 State 或者一整行数据丢进日志 —— 员工姓名、身份证号、银行卡号、
    工资明细就这么进了 logs/agent.log，而且没人会去翻它。

    所以这里有三道**结构性的**闸门（都是靠代码挡住的，不是靠注释提醒）：

        1. 字段只收字符串。传进来一个 dict，记的是 "<dict>"，不是它的内容。
           手滑把整个 State 传进 message 的那一次，落盘的是一个类型名。
        2. error 只记**异常类型名**，消息文本一个字都不记。异常消息里装什么
           由写 raise 的人决定，而日志这一层**没有办法**可靠地判断哪条消息
           安全 —— 截断不是脱敏：姓名两个字，截到 300 字也不会变成星号。
        3. 要记结构，只能走 describe()（写成一行文本）或者 shape_of()
           （写成 JSON 能装的形状）—— 两个都只报「有哪些键、每项多大」，
           不报值。

关于 task_id：

    它由 run() 生成，通过 contextvars 绑在当前执行流上。
    Node 记日志时不用传、也不用知道它从哪来 —— 这是「不用为了日志
    去改九个节点」的实现方式。

    刻意**没有**塞进 State：State 里每个键都会有节点去读、Router 去判，
    多一个没人读的键，就是多一个以后会被误当成业务字段的东西。

关于这个模块**不保证**什么（说清楚比含糊强）：

    日志是**证据**，不是账本。进程被掐、断电、文件系统没来得及 flush，
    最后一条事件就可能不在 —— 所以**不能**因为日志里找不到 task_completed
    或 task_failed，就断定那次任务没跑完或者结果如何。
    业务事实的来源是 State 和 memory.json，不是这里。

    trace.jsonl 是同一条规矩，而且它更容易被误会：它是顺序的、
    一个节点一行，看着很像「这次运行完整走过了这几个节点」——
    但它同样可能在最后一行断掉。**要判断这次跑完了没有，看 summary，
    不看轨迹。**

    本阶段也**不**提供：历史任务查询、断点恢复、重试、实时进度。
"""

import datetime
import json
import logging
import os
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path


# ============================================================
# 配置
# ============================================================

# 日志落在项目根目录的 logs/agent.log，和 logger.py 并排。
#
# 不放 data/ —— 那边装的是演示用的 xlsx（测试数据），这里是运行痕迹，
# 性质不同，混一起以后不好清理。
#
# logs/ 目录不存在时会在第一次写日志前自动创建（见 _file_handler）。
LOG_FILE = Path(__file__).parent / "logs" / "agent.log"

# 节点轨迹落在**同一个目录**、另起一个文件。
#
# 为什么不像 runs/summary.jsonl 那样另开目录：这个和 agent.log 是
# 同一次运行的两面（事件流 / 节点形状），task_id 对齐着一起看，
# 放一起才找得到。而 summary 是**另一次粒度**的东西（一次运行一行），
# 混进 logs/ 会被当成「另一种格式的日志行」。
#
# logs/ 已经在 .gitignore 里，这个文件跟着一起被挡，不用改 ignore。
TRACE_FILE = Path(__file__).parent / "logs" / "trace.jsonl"


# ---- 文件长了怎么办 ----
#
# 一个日志类文件超过这个大小就滚一代：agent.log → agent.log.1，
# 新的一代从空文件重开。**只留一代**，所以磁盘占用恒定
# （每个文件最多 2 × 这个数，两个文件加起来 ≤ 20 MB）。
#
# 5 MB 是怎么定的，两头都有理由：
#
#   下限  不能定得太小。benchmark 跑一次就是 27 次运行，
#         上限太小的话，真出事的那次很可能已经被挤掉了 ——
#         而「日志是证据」这条线，最怕的就是证据被自动删掉。
#         几周的日常演示才几百 KB，5 MB 大约是一千多次运行的量。
#
#   上限  不能定得太大。排查的第一步是**打开这个文件**，
#         而几 MB 以上的文本文件在编辑器里已经开始卡了。
#
# 需要留更久，就把这个数调大 —— 一行。**不要**改成「多留几代」：
# 那等于换个地方删旧文件，而且要在磁盘上维护一串带编号的副本，
# 翻起来比直接调大这个数麻烦得多。
MAX_FILE_BYTES = 5 * 1024 * 1024

# 滚出来的那一代叫什么。写在名字后面（agent.log.1）而不是换扩展名，
# 是为了让它和原文件的亲戚关系一眼可见 —— 也更不容易被别的工具
# 按 *.log 之类的通配符误扫进来。
ROTATED_SUFFIX = ".1"

# logging 按名字管理 logger，同一个名字拿到的是同一个对象。
# 所以「配过 handler 没有」可以直接问这个对象，不用自己记标志位 ——
# 标志位在 importlib.reload 之后会归零，而对象上的 handler 不会。
#
# 名字挑项目专属的，不用 root，也不和别的库撞。
LOGGER_NAME = "payroll_agent"


# ---- 事件名 ----
#
# 写成常量，不在各处直接写字符串。理由有两条：
#
#   1. 手打的字符串拼错了不报错，只会静默多出一种事件名，
#      直到某天按事件名统计时才发现有两拨数据
#   2. 事后 grep 日志前，先在这里看到事件名的全集
TASK_STARTED = "task_started"
TASK_COMPLETED = "task_completed"
TASK_FAILED = "task_failed"
NODE_STARTED = "node_started"
NODE_COMPLETED = "node_completed"
NODE_FAILED = "node_failed"

# run-summary 自己的出口坏了（runs/ 建不出来、磁盘满……）。
# 它不是任务事件，是「记录功能出问题」本身 —— 所以另起一个事件名，
# grep 的时候不会和业务失败混在一起。
SUMMARY_WRITE_FAILED = "summary_write_failed"

# 运行记录里有读不出来的行，被跳过了。
#
# 和上面那个是一对：一个管写不进去，一个管读不出来。
# 两个都算「记录功能出问题」，都不是业务失败 —— 单独成事件名。
#
# 为什么读不出来也要记一笔：跳过的行不会被任何统计看见，
# 但它**真的占了一个样本**。不吭声的话，读报告的人会把
# 「我跳了 3 行」当成「本来就只有这些」。
SUMMARY_READ_SKIPPED = "summary_read_skipped"

# 节点轨迹没写进去（logs/ 建不出来、磁盘满……）。
#
# 同样不是任务事件，是「记录功能出问题」—— 单独一个事件名，
# grep 业务失败的时候不会把它一起捞出来。
TRACE_WRITE_FAILED = "trace_write_failed"

# 文件该滚了，但滚不动（被别的进程占着、没权限……）。
# 同样是「记录功能出问题」，不是业务失败。
ROTATE_FAILED = "rotate_failed"

# ---- 状态 ----
#
# 和事件名是两个轴，不合并：同一个 node_started 事件以后可能带 status=skipped
# （比如某个节点因为上游没数据被跳过），那时候事件名不变、状态变。
RUNNING = "running"
SUCCESS = "success"
FAILED = "failed"

# 字段缺省值。用 "-" 而不是空串：
# 日志是横向扫的，一列里有的有值有的空着，看不出是「本来就没有」
# 还是「忘了填」。一个 "-" 至少说明这里是空着的。
EMPTY = "-"

# 单条日志的字数上限。
#
# 这只防「一行长到没法看」，**不是**安全边界 ——
# 安全边界是「非字符串一概不展开」（见 _safe_text）和
# 「异常只记类型名」（见 error_type）。
#
# 留着它是因为 message 是调用方手写的，写长了不该把日志撑成一面墙。
MAX_MESSAGE_CHARS = 500

# 统一格式：字段顺序固定，一行一条。
#
# 全字段都摆在一行里（不是 JSON），是为了 grep 方便：
#   想看某个任务   grep "task=t-001"
#   想看某类事件   grep "event=node_failed"
#   想看所有失败   grep "status=failed"
# 换成 JSON 也能查，但命令行上要多绕一层引号。
#
# %(message)s 是 logging 自带的标准字段（就是上面那句人看的话），
# 其余五个由 log_event 通过 extra= 挂上去。
LOG_FORMAT = (
    "%(asctime)s | task=%(task_id)s | node=%(node_name)s | "
    "event=%(event)s | status=%(status)s | error=%(error)s | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================
# 任务身份 —— task_id 从哪来、怎么传到各个 Node
# ============================================================
#
# 为什么用 contextvars，而不是往 State 里加一个 task_id 键：
#
#   task_id 是「这次运行」的标识，不是业务流程要用的数据。
#   放进 State 意味着它要跟着每个节点传一圈、被 checkpoint 序列化、
#   出现在最终结果里 —— 而没有任何节点会读它。
#
#   contextvars 跟着**执行流**走：谁在跑，谁就看得见。
#   节点侧完全不需要知道有这个东西存在。
#
# 它在当前执行模型下可靠 —— 但**只说「读」这一半**。
#
# LangGraph 把每个节点放在一个**拷贝出来的 context** 里执行。于是：
#
#     节点里读 run() 设的 task_id       可以   （向下传播）
#     run() 里读节点设的值              不行   （写只落在那份拷贝上，出不来）
#
# 这不是推测，是实测出来的（test_logger 里有一条 copy_context 的用例钉着它）。
# task_id 是「往下发、节点只读」，所以正好落在能用的那一半；
# 而 failed_node 是「节点写、run() 读」，正好落在不能用的那一半 ——
# 所以它不能是 ContextVar[str]，得换个装法，见 _RunMarks。
#
# 边界说清楚：如果以后改成并行分支、线程池或进程池，
# 连「读」那一半也会断（进程池一定跟不过去）。
# 到那时要改的是这一小段和 run()，不是九个节点 ——
# 这也正是把机制收在这里的原因。


_current_task_id: ContextVar[str | None] = ContextVar("payroll_task_id", default=None)


class _RunMarks:
    """一次运行里「从节点往上冒」的标记。

    现在有两类：

        failed_node                  最后写成什么（谁失败的）
        llm_calls / input_tokens /
        output_tokens                **累加**（这一趟花了多少）

    两类都只增不减、都只属于这次运行 —— 所以装在同一个对象里，
    每跑一次换一个新的。

    为什么不用 ContextVar[str]：

        上面那条 —— 写不回外层。实测过：节点里读得到 task_id，
        但节点里 set() 的东西 run() 那边看不见。

    为什么这样就work：

        contextvars 拷贝 context 时，拷的是「变量 → 值」这张表，
        而**值本身是引用，不是深拷贝**。task_scope 建一个 _RunMarks
        放进上下文，节点拿到的是**同一个对象**，改它里面的字段
        就绕过了「set 不冒泡」这件事。

    为什么建在 task_scope 里而不是模块级单例：

        模块级的话，两次运行共用一份，上一次的失败节点会漏到下一次 ——
        第二次明明成功了，summary 里却写着 failed_node=calculate。
        每次运行一份新的，这种串味就不可能发生。

    为什么是可变的：这里要的恰恰是「共享并原地改」。
    换成不可变的就得每改一次 set 一次，又回到传不出去的老问题上。
    """

    __slots__ = ("failed_node", "llm_calls", "input_tokens", "output_tokens")

    def __init__(self) -> None:
        self.failed_node: str | None = None

        # LLM 用量。这三个是**累加**的，和 failed_node 的「覆盖」不一样。
        #
        # llm_calls 记的是**发起过几次调用**，不是只记 token 数 —— 因为
        # 「压根没调用」和「调了但没拿到用量（网络失败）」在 token 上
        # 都是 0。少了这个计数，**失败看起来是免费的**。
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0


_run_marks: ContextVar[_RunMarks | None] = ContextVar("payroll_run_marks", default=None)


def new_task_id() -> str:
    """生成一个新的任务 ID。

    用 uuid4 而不是「时间戳 + 序号」：同一秒里连着跑两次也必须能区分开。
    取前 12 位十六进制 —— 这个 ID 只用来在日志里认人，
    不承担防碰撞的严格责任（真要防碰撞，日志格式里还有时间戳兜着）。
    """
    return uuid.uuid4().hex[:12]


@contextmanager
def task_scope(task_id: str):
    """在 with 块范围内，把 task_id 绑到当前执行流上。

    块里所有 node_xxx 调用**不传 task_id 也能自动带上它**。

    用 token + reset 而不是退出时 set(None)：嵌套时（连着跑两次 run，
    或者测试里套着用）能正确回到上一层，不会把外层的 task_id 抹掉。

    failed_node 跟着一起进出：每次运行都发一份**新的** _RunMarks。
    复用同一份的话，第一次那个失败的节点会留在里面 ——
    第二次明明成功了，summary 里却写着 failed_node=calculate。
    """
    task_token = _current_task_id.set(task_id)
    marks_token = _run_marks.set(_RunMarks())
    try:
        yield task_id
    finally:
        _current_task_id.reset(task_token)
        _run_marks.reset(marks_token)


def current_task_id() -> str | None:
    """当前执行流上绑着的 task_id。没绑过就是 None。"""
    return _current_task_id.get()


def last_failed_node() -> str | None:
    """当前执行流上最后一个失败的节点名。没失败过就是 None。

    给 run-summary 读的。运行之外（没有 task_scope）读到的是 None。
    """
    marks = _run_marks.get()
    return marks.failed_node if marks is not None else None


def note_llm_call(
    *, input_tokens: int | None = None, output_tokens: int | None = None
) -> None:
    """记一次 LLM 调用。在**调用之后**记 —— 成功失败都算一次。

    token 传 None 表示「这次没拿到用量」（调用失败、或者服务端没返回
    usage 字段）。**不要传 0**：0 是「拿到了，而且就是 0」，
    和「没拿到」是两回事，混起来之后失败就变成免费的了。

    它同样走 _RunMarks 而不是 set() 一个新值 —— 调用发生在节点里面，
    而节点跑在拷贝出来的 context 里，set() 传不回 run()（见 _RunMarks）。
    """
    marks = _run_marks.get()
    if marks is None:
        # 不在任何一次运行里（比如有人单独调 llm_node 的函数）。
        # 丢掉这次计数，不崩 —— 和 node_started 没有 task_id 时记占位符同理。
        return

    marks.llm_calls += 1
    if input_tokens is not None:
        marks.input_tokens += input_tokens
    if output_tokens is not None:
        marks.output_tokens += output_tokens


def llm_usage() -> dict:
    """这次运行的 LLM 用量。不在运行里、或者一次都没调用过，就是全 0。"""
    marks = _run_marks.get()
    if marks is None:
        return {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}

    return {
        "llm_calls": marks.llm_calls,
        "input_tokens": marks.input_tokens,
        "output_tokens": marks.output_tokens,
    }


# ============================================================
# 出口
# ============================================================


def get_logger() -> logging.Logger:
    """拿到 logger，第一次调用时把 handler 装好。

    重复调用、模块被多次 import 都安全 —— handler 只在**当前没有 handler**
    时才装。判断问的是 logging 自己那个按名字缓存的对象，不是模块里的变量，
    所以下面两种情形都能挡住：

        同一个进程里第 N 次调用        handlers 非空，直接返回
        模块被 importlib.reload 之后   handlers 还在，还是直接返回

    ——「同一条日志打了两遍」最常见的原因就是重装 handler，
    而它长得不像 bug：日志内容全对，只是每条都有两份。翻起来很费时间。

    代价写明：如果**别处**先往这个同名 logger 上挂了自己的 handler，
    这里就不再装我们那两个。所以名字挑了 "payroll_agent" 这种项目专属的。
    """
    log = logging.getLogger(LOGGER_NAME)

    if log.handlers:
        return log

    log.setLevel(logging.INFO)

    # 不往 root 传。
    #
    # 传上去的话，任何一处 logging.basicConfig()（或者将来哪个库调了它）
    # 都会让同一条记录在两个地方各打一遍 —— 这是「日志重复」的另一个来源，
    # 而且比重复装 handler 更隐蔽：重复的是别人的输出。
    log.propagate = False

    log.addHandler(_console_handler())

    # 文件出口可能建不出来（见 _file_handler）。建不出来就只留控制台 ——
    # 「日志功能降级」比「工资流程起不来」划算得多。
    file_handler = _file_handler()
    if file_handler is not None:
        log.addHandler(file_handler)

    return log


def _console_handler() -> logging.Handler:
    """控制台出口：开发时能直接看到流程走到哪了。

    流对象在**这一行**取定。之后再 sys.stdout = xxx 不会跟着变 ——
    想要跟着变，就得每次 emit 都重读一次 sys.stdout，为这点便利把 handler
    写复杂不值当。测试里要换出口，把 handler 清掉重配即可（见 test_logger.py）。

    Windows 控制台默认是 GBK 编码。装不下的字符会让 StreamHandler 走
    handleError，在 stderr 上吐一段 "--- Logging error ---"：
    日志本身没写坏，但那段噪音看着像出了大事。改成 replace 之后，
    顶多某个字变成 ?，别的照常出来。
    """
    stream = sys.stdout

    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            # 换不了就算了 —— 这条不是功能，只是少一点噪音，
            # 不能因为它让整个日志模块起不来
            pass

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    return handler


def _file_handler() -> logging.Handler | None:
    """文件出口：logs/agent.log，追加写。**建不出来就返回 None，不抛异常。**

    encoding="utf-8" 不能省。Windows 上不写就是按系统默认编码（GBK）落盘，
    用别的工具打开就是乱码 —— 而中文在这个项目的日志里是常态，不是例外。

    mkdir 放在这里，不放模块顶层：import 一个模块不该在磁盘上建目录。
    只有真的要用到文件出口时才建。

    建不出来（目录没权限、路径被一个同名文件占着）时兜住不抛：

        这是日志模块。它自己坏掉不该把工资流程一起带崩 ——
        为了记录「发生了什么」而让「什么」发生不了，是拿反了。

    代价写清楚：那种情况下只有控制台有日志，agent.log 是空的，
    **而且不会有任何提示**。所以部署完第一件事是确认文件真的在长。
    另外它只试一次：第一次建失败之后控制台 handler 已经在了，
    下次 get_logger() 看到 handlers 非空就直接返回，不会再重试。
    """
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    except OSError:
        return None

    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    return handler


# ============================================================
# 记日志
# ============================================================


def log_event(
    event: str,
    message: str = "",
    *,
    task_id: str | None = None,
    node_name: str | None = None,
    status: str | None = None,
    error=None,
    level: int = logging.INFO,
) -> None:
    """记一条事件。这是唯一真正的出口，下面几个简写都是它的转发。

    参数：
        event       事件类型，用上面那几个常量
        message     人看的一句话。**只写元数据**，不要往里贴数据行
        task_id     这次任务。**不传就从当前执行流上取**（见 task_scope）——
                    所以节点侧一行都不用改也能带上它。两边都没有就记占位符
        node_name   哪个节点
        status      running / success / failed
        error       异常对象。记的是**它的类型名**，消息文本不记
                    （见 error_type）
        level       logging 的级别

    为什么关键字参数是 keyword-only（那个 *）：

        前面这几个字段全是字符串，如果都能按位置传，
        log_event(NODE_FAILED, "parse", "t-1") 这种调用编译得过、意思全错，
        而且**不报错** —— 只是日志里 node 和 task 换了个位置，
        事后按 node 统计出来的数字全是错的。

        强制写名字，多打几个字，换掉这一类事故。
    """
    log = get_logger()

    # 调用方没给 task_id 就从当前执行流上取 ——
    # 节点侧因此不用知道 task 是怎么发下来的。
    if task_id is None:
        task_id = _current_task_id.get()

    # extra 是 logging 给自定义字段留的口子，它有两个规矩：
    #
    #   1. 键不能和 LogRecord 自带的撞（message、name、levelname……）——
    #      撞了会直接抛 KeyError
    #   2. 格式化串里用到的键，这条记录必须全有 —— 缺一个也是 KeyError，
    #      而且是在 emit 里抛，那行日志直接丢掉
    #
    # 所以五个键在这里一次性补齐，调用方少给哪个都不会把日志打没。
    #
    # 五个值全部走 _safe_text / error_type：**没有一处 str(对象)**。
    # 这是本模块的安全底线，改动这里之前先看模块开头那段。
    log.log(
        level,
        _safe_text(message, MAX_MESSAGE_CHARS),
        extra={
            "task_id": _field(task_id),
            "node_name": _field(node_name),
            "event": _field(event),
            "status": _field(status),
            "error": error_type(error),
        },
    )


# ---- 意图化的简写 ----
#
# 调 log_event(TASK_STARTED, ..., status=RUNNING) 能写，但每个调用点都要
# 记一遍「哪个事件配哪个状态」，总有一次会记错。常量配错不报错，
# 只会让日志里出现 task_started + status=success 这种没人能解释的组合。
#
# 简写把「事件 + 状态 + 级别」这三件绑死的事固定在一个地方。
#
# 所有简写的可选参数都是 keyword-only（那个 *），这条不是风格问题：
#
#     node_xxx(name, task_id, message)     第二个位置参数是 task_id
#     task_xxx(task_id, message)           第二个位置参数是 message
#
# 两个同名位置的参数含义不一样，而它们**都是字符串**。
# 写 node_completed(name, "交给 extract") 编译得过、跑得通、不报错 ——
# 只是 task_id 那一格变成了「交给 extract」，而真正的 task_id 没了。
# 这种错不会红，只会让日志里出现一个谁也认不出来的任务 ID。
#
# 实测踩过一次（V2 接入当天就是这么写的），所以加 * 焊死。


def task_started(task_id: str, *, message: str = "") -> None:
    """一个任务开跑。"""
    log_event(TASK_STARTED, message, task_id=task_id, status=RUNNING)


def task_completed(
    task_id: str, *, message: str = "", status: str = SUCCESS
) -> None:
    """任务结束了。

    status 默认 success；**业务流程自己停下的**那种失败传 FAILED
    （比如人工否决了字段、文件读不进来 —— 流程走到 exception 节点就收住了）。

    它和「程序崩了」的区别落在 event 上，不是落在 status 上：

        task_completed + status=failed   流程正常返回了，只是没算成
        task_failed                      抛异常了，什么都没返回

    两个字段要一起看。只看 status=failed 会把这两种情况混成一种，
    而它们要处理的人完全不同：一个去找业务方确认口径，
    一个去看 traceback。
    """
    log_event(TASK_COMPLETED, message, task_id=task_id, status=status)


def task_failed(task_id: str, *, message: str = "", error=None) -> None:
    """任务抛出了未捕获的异常。

    用 ERROR 级别：收日志的人按级别过滤时，失败不会被漏掉。

    注意这个函数**只负责记**。异常本身由调用方（run）原样往上抛 ——
    日志里有一行 task_failed，不代表异常被处理过了。
    """
    log_event(
        TASK_FAILED,
        message,
        task_id=task_id,
        status=FAILED,
        error=error,
        level=logging.ERROR,
    )


def node_started(
    node_name: str, *, task_id: str | None = None, message: str = ""
) -> None:
    """一个节点开始执行。"""
    log_event(
        NODE_STARTED, message, task_id=task_id, node_name=node_name, status=RUNNING
    )


def node_completed(
    node_name: str, *, task_id: str | None = None, message: str = ""
) -> None:
    """一个节点执行成功。"""
    log_event(
        NODE_COMPLETED, message, task_id=task_id, node_name=node_name, status=SUCCESS
    )


def node_failed(
    node_name: str,
    *,
    task_id: str | None = None,
    message: str = "",
    error=None,
) -> None:
    """一个节点执行失败。

    error 传**异常对象**。落盘的只有类型名（"ValueError"），
    异常消息一个字都不记 —— 它可能装着从工资表里读出来的姓名和金额。

    这人看的一句话写在 message 里，由调用方负责它不含业务数据。
    """
    # 顺手记下「栽在哪」给 run-summary 用。放这里而不是放 traced()，
    # 是为了让「有节点失败」和「记下是哪个节点」永远同时发生。
    #
    # 注意是改对象里面的字段，不是 set() 一个新值 —— 节点跑在拷贝出来的
    # context 里，set() 传不出去（见 _RunMarks）。
    marks = _run_marks.get()
    if marks is not None:
        marks.failed_node = node_name

    log_event(
        NODE_FAILED,
        message,
        task_id=task_id,
        node_name=node_name,
        status=FAILED,
        error=error,
        level=logging.ERROR,
    )


def rotate_if_needed() -> None:
    """文件太大了就滚一代。**在两次运行之间调用**，一次运行中途不切。

    为什么在运行之间切，而不是写到一半发现超了就切：

        写到一半切的话，**一次运行的节点会散在两个文件里** ——
        而这两个文件存在的意义就是「结合着看」。排查的人找到一半，
        发现后半截在另一个文件，还不知道该从哪一行接。

    所以上限是**软的**：最坏情况是超出一次运行的量（几 KB）。
    要硬上限就得付出「一次运行被劈开」的代价，不值。

    为什么不用 logging 自带的 RotatingFileHandler：

        它也是按写入量切的 —— 同样会把一次运行劈开。
        而且它管的是 logging 的出口，管不了 trace.jsonl
        （那个不是走 logging 写的）。两个文件两种切法，
        以后会有人以为它们的行为是一样的。

    跨进程安全：不保证。两个进程同时滚会有一个失败，失败的那边
    记一条 WARNING 然后**继续照原样往老文件里写** —— 不崩，也不丢。
    这个项目是单进程跑的，真要多进程再谈。
    """
    over = [path for path in (LOG_FILE, TRACE_FILE) if _over_limit(path)]
    if not over:
        # 绝大多数调用走这条：两次 stat，什么都不做
        return

    # 先摘 handler，其中文件那个**关掉**。Windows 上文件还被打开着就改不了名
    # （WinError 32），而这里失败的表现很隐蔽：日志看起来一切正常，
    # 只是**永远不滚**。
    _reset_handlers()

    for path in over:
        _rotate_over_limit(path)

    # 按**新的**路径把出口重新装好。
    # 这一步不能省：get_logger() 看到 handlers 非空就直接返回，
    # 少装一次的话，agent.log 会从这一刻起静悄悄地不再增长。
    get_logger()


def _over_limit(path: Path) -> bool:
    """这个文件是不是该滚了。文件不存在就是不用滚。"""
    try:
        return path.exists() and path.stat().st_size > MAX_FILE_BYTES
    except OSError:
        # 读不到大小（权限、路径被占）就当它不用滚 ——
        # 这是记录功能，不该因为这个问题把流程拦下
        return False


def _reset_handlers() -> None:
    """把挂着的 handler 全摘掉；文件那个顺手关掉。

    文件 handler 必须 close，不然 Windows 不让改名。
    控制台那个**只摘不关**：

        StreamHandler.close() 在有些 Python 版本里会把 stream 一起关掉，
        而它的 stream 是 sys.stdout。被关掉之后，之后所有的 print 都会抛
        "I/O operation on closed file" —— 现象是「滚了一次日志之后，
        程序开始莫名其妙地崩」，而没人会想到是日志干的。
        （实测本机的 3.12 不关，但这是版本行为，不值得赌。）

    全摘掉是为了让 get_logger() 能重新装 —— 它看到 handlers 非空就返回。
    """
    log = logging.getLogger(LOGGER_NAME)
    for handler in list(log.handlers):
        log.removeHandler(handler)
        if isinstance(handler, logging.FileHandler):
            handler.close()


def _rotate_over_limit(path: Path) -> None:
    """path → path.1，覆盖上一代。失败不抛，记一条 WARNING。"""
    rotated = path.with_name(path.name + ROTATED_SUFFIX)
    try:
        # os.replace 而不是 Path.rename：前者在目标已存在时**直接覆盖**，
        # 后者在 Windows 上会抛 FileExistsError。
        # 「只留一代」就是这一行实现的 —— 旧的那代在这里被丢掉。
        os.replace(path, rotated)
    except OSError as exc:
        # 滚不动就照原样继续往老文件里写。不崩，也不装没发生。
        #
        # message 里用的是 **文件名**，不是路径：完整路径带着这台机器的
        # 目录结构和用户名（C:\Users\谁\Desktop\...），对排查没用。
        log_event(
            ROTATE_FAILED,
            f"{path.name} 滚不动，继续往原文件里写",
            error=exc,
            level=logging.WARNING,
        )


# ============================================================
# 节点轨迹
# ============================================================
#
# 事件流（上面那些）回答「谁在什么时候做了什么」。
#
# 它答不了的，是有一次真的卡住过的那种问题：
#
#     某个节点判断错了。翻 agent.log，九个节点全都 node_completed、
#     状态全是 success，看不出任何异常 —— 因为**没有一步是错的**，
#     错的是某一步**看到的东西和它以为的不一样**。
#
# 所以还需要第二样东西：每个节点进来时，手上那个 State 长什么样。
#
# 「长什么样」在这里是字面意思 —— 只记形状，不记值（见 shape_of）。
# 排查这类问题看形状就够了：这一格是空的、这一步只看到 0 行、
# memory_context 里一条规则都没有 —— 这些判断全都不需要知道姓名和金额。


def shape_of(value):
    """把一个值压成「形状」，结果能直接塞进 JSON。**不含任何内容。**

        shape_of({"raw_rows": [...20 行...], "file_path": "data/工资表.xlsx"})
        ->  {"raw_rows": "list(20)", "file_path": "str(13)"}

    和 describe() 的分工，是**媒介**不同，不是尺度不同：

        describe()   压成一行文本   —— 给 agent.log 里的 message 用，
                                     那个文件是一行扫过去的
        shape_of()   压成 JSON 结构 —— 给 trace.jsonl 和内存里的轨迹用，
                                     那个文件是给机器读、要按节点切片的

    两个都不能报值，这一条是一样的。

    字典只往下钻一层，和 describe 一样：memory_context 报成
    {"approved_rules": "list(1)", "historical_rules": "list(0)"}
    比 "dict(2键)" 有用得多 —— 后者看不出机器到底能不能用这些规则。
    再往下钻收益递减，噪音翻倍。
    """
    if isinstance(value, dict):
        return {str(key): _inner_shape(item) for key, item in value.items()}

    return _leaf_shape(value)


def _inner_shape(value):
    """字典的**值**这一层：是字典就再钻一层，其余压成叶子。

    为什么要多钻这一层：State 里最需要看形状的两个键
    （memory_context、validation_result）装的是字典，而
    「规则库里到底有没有能用的规则」这件事，只有钻进去才看得见 ——
    报成 "dict(2键)" 的话，「一条规则都没有」和
    「有两条但全是历史记录」长得一模一样。
    """
    if isinstance(value, dict):
        return {str(key): _leaf_shape(item) for key, item in value.items()}

    return _leaf_shape(value)


def _leaf_shape(value):
    """压到「一层」为止：容器报几个，标量报是个什么东西、多大。

    为什么全是字符串（"list(20)" 而不是 20）：

        数字和下标的区别会没掉。{"raw_rows": 0} 到底是**空列表**
        还是**数字零**？读的人和写脚本的人都得回去猜。
        "list(0)" 和 "number" 摆在那儿，谁都不用猜。

    为什么 str 只报长度：

        见本文件开头 —— 那是这一层存在的全部理由。
    """
    if value is None:
        return None

    # bool 要挡在数字前面：bool 是 int 的子类，
    # 不挡的话 True 会报成 "number"，而「是不是布尔值」正是
    # parse_error 这类字段最关键的一点。
    if isinstance(value, bool):
        return "bool"

    if isinstance(value, (int, float)):
        return "number"

    if isinstance(value, str):
        return f"str({len(value)})"

    if isinstance(value, dict):
        return f"dict({len(value)}键)"

    if isinstance(value, (list, tuple, set)):
        return f"{type(value).__name__}({len(value)})"

    return type(value).__name__


def _worth_reporting(value) -> bool:
    """这一格值不值得报。空列表、空字典、空串、None 都不值得。

    State 有十几个键，大部分在这次运行里是空的（没人复核、没有冲突……）。
    全报出来是一面墙，真正有信息的那几个被淹掉 ——
    而「哪些格子是空的」这件事，**不报本身就是答案**。
    """
    return value is not None and value != "" and value != [] and value != {}


def _state_shape(state) -> dict:
    """一个 State 的形状：只报有内容的键，值只报形状。

    滤空放在这里、而不是放在拼记录那几行里，是因为「什么值得报」
    和「怎么记这一行」是两件事，混在一起以后加字段会越来越难读。
    """
    if not isinstance(state, dict):
        # 正常不会走到这儿。真走到了（有人把 State 传错了），
        # 报出它是什么类型 —— 而不是崩，或者把它 str 出来
        return {"<不是 dict>": _leaf_shape(state)}

    return {
        str(key): shape_of(value)
        for key, value in state.items()
        if _worth_reporting(value)
    }


def node_trace(
    node_name: str,
    state: dict,
    *,
    changed=None,
    next_action=None,
    status: str = SUCCESS,
    error=None,
    task_id: str | None = None,
) -> dict:
    """一个节点追一行到 logs/trace.jsonl，**并把这行还给调用方**。

    为什么还回来：内存里那份轨迹（traced 的 sink）如果自己另拼一份，
    就会有两个「同一个节点长什么样」的实现 —— 它们迟早会漂开，
    而漂开的方式是最难查的那种：**内存里看到的和文件里写的不一样**，
    谁也不知道该信哪个。所以这里只做一份，两边共用。

    参数：

        state        节点进来时手上那个 State。**原样传进来就行** ——
                     压形状这件事在这里做，不在调用方做。
                     这正是闸门的装法：调用方手滑不会造成泄露，
                     因为调用方根本没有「写值」这个选项。
        changed      节点这次改了哪些键（键名，不是值）。
                     失败时传 None。
        next_action  交给谁。失败时传 None。
        status       success / failed
        error        异常对象，只记类型名

    None 和空的区别在这行里是有意义的：

        changed=None          节点**没跑完**，没有局部更新
        changed=[]            跑完了，但一个键都没改
        next_action=None      到头了（流程到此为止）

    这个方法**不会抛异常**。理由和 log_event 一样：记录功能坏掉
    不该把工资流程一起带崩。写不进去就往 agent.log 记一条 WARNING ——
    不崩，但也不装作没发生。
    """
    if task_id is None:
        task_id = _current_task_id.get()

    record = {
        "time": datetime.datetime.now().strftime(DATE_FORMAT),
        "task_id": _field(task_id),
        "node": _field(node_name),
        "status": _field(status),
        # 形状在这里统一压。上面那三个字段是**我们自己的**词表
        # （节点名、状态、路由名），走 _field 洗一遍照样只出文本。
        "input": _state_shape(state),
        "changed": None
        if changed is None
        else [_safe_text(name, MAX_MESSAGE_CHARS) for name in changed],
        "next_action": _safe_text(next_action, MAX_MESSAGE_CHARS) or None,
        "error_type": error_type(error) if error is not None else None,
        # 哪一步炸的。只有 error 不是 None 时才有意义 —— 成功那次
        # 两格都是 null，读的人一眼知道这条节点是好好跑完的
        "error_at": error_where(error),
    }

    _append_json(TRACE_FILE, record, task_id, TRACE_WRITE_FAILED, "节点轨迹没写进去")
    return record


def _append_json(path: Path, record: dict, task_id, event: str, note: str) -> None:
    """往一个 jsonl 追加一行。**不抛异常。**

    和 run_summary.write 的错误处理是同一条规矩：记录功能自己坏掉，
    不该把业务带崩 —— 而且这里更严重一点，node_trace 是在节点跑的
    路径上调用的，它抛异常会被 traced 那个 except 抓住，
    记成「节点抛出异常」，然后整个流程停下来。
    为了记一行「这个节点长什么样」，让这个节点真的失败，是拿反了。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        log_event(
            event,
            note,
            task_id=task_id,
            error=exc,
            level=logging.WARNING,
        )


# ============================================================
# 安全摘要
# ============================================================


def describe(value) -> str:
    """把一个值压成「形状」，不含内容。

    给「我手上有这么个东西，想知道它长什么样」准备的出口：

        describe({"raw_rows": [...3 行...], "file_path": "data/工资表.xlsx"})
        ->  "{raw_rows: list(dict(7键) x3), file_path: str(13)}"

    键名报，值不报。差别在于：键名是程序里写死的（raw_rows、file_path、
    problems），值是数据。而排查大多数问题（数据收没收到、几条、
    是不是 None、字段认出来没有）看形状就够了。

    字典只往下钻一层：memory_context 报成
    "{approved_rules: list(dict(4键) x1), historical_rules: list(0)}"
    比 "dict(2键)" 有用得多 —— 后者看不出机器到底能不能用这些规则。
    再往下钻收益递减，噪音翻倍。

    它和 shape_of() 是同一个尺度、两种装法（文本 / JSON），
    见 shape_of 的说明。

    它不是万能的：只保护「用 describe 记结构」这一条路。
    真正让数据不落盘的是**没有**那个入口 —— 见本文件开头。
    """
    if isinstance(value, dict):
        if not value:
            return "{} 空"
        body = ", ".join(f"{key}: {_shape(item)}" for key, item in value.items())
        return "{" + body + "}"

    return _shape(value)


def _shape(value) -> str:
    """一个值「是什么类型、有多大」。永远不含内容。"""
    if value is None:
        return "None"

    # bool 是 int 的子类，得挡在数字前面 ——
    # 不挡的话 True 会报成 number，而「是不是布尔值」正是
    # extraction.py 里专门处理过的一种脏数据（见 _to_number）
    if isinstance(value, bool):
        return "bool"

    if isinstance(value, (int, float)):
        return "number"

    if isinstance(value, str):
        return f"str({len(value)})"

    if isinstance(value, dict):
        return f"dict({len(value)}键)"

    if isinstance(value, (list, tuple, set)):
        if not value:
            return f"{type(value).__name__}(0)"
        # 报第一个元素的形状：一眼看出「里面装的是行还是字符串」。
        # 空列表走上面那条 —— next(iter([])) 会抛 StopIteration
        return f"{type(value).__name__}({_shape(next(iter(value)))} x{len(value)})"

    return type(value).__name__


# ============================================================
# 内部工具函数
# ============================================================


def _field(value) -> str:
    """必填字段的值：空的填占位符，保证格式里那一格永远有东西。"""
    return _safe_text(value, MAX_MESSAGE_CHARS) or EMPTY


def _safe_text(value, limit: int) -> str:
    """把值洗成「一行、有上限」的文本。**只认字符串，其余一概不展开。**

    非字符串回一个尖括号包起来的类型名，这是本模块最要紧的一条防线：

        log_event(NODE_FAILED, state)          <- 手滑把整个 State 传进来
        log_event(NODE_STARTED, rows)          <- 或者一整张表

        上一版这里是 str(value)：员工姓名、身份证号、工资明细**全进了日志**，
        而且是悄悄进的 —— 代码看起来完全正常，测试也不会红。
        现在落盘的是 "<dict>" / "<list>"：看得见「有人传错了」，
        但一个字的数据都没出去。

    另外三件事和上一版一样，每一件都对应一种日志被写坏的方式：

        换行 -> 空格     一条日志占好几行，"|" 分段就废了；
                         而且能靠塞换行伪造出另一条记录（"\\n2026-01-01 | ..."）
        去掉首尾空白     日志是横向扫的，前面多几个空格看不出来
        超长截断         见 MAX_MESSAGE_CHARS

    要记一个结构，走 describe() —— 那是唯一被允许的结构化入口，
    而且它只报形状。
    """
    if value is None:
        return ""

    if not isinstance(value, str):
        return f"<{type(value).__name__}>"

    text = value.replace("\r", " ").replace("\n", " ").strip()

    if len(text) > limit:
        # 把原长写出来。只截不报的话，看的人分不清
        # 「就这么多」和「被截了」
        return f"{text[:limit]}…（已截断，原长 {len(text)} 字）"

    return text


def error_where(error) -> str | None:
    """异常是在**哪一行**抛出来的：最里面那一帧的「文件:行号 函数名」。

        "calculation.py:88 calculate_payroll"

    它和 error_type 是一对，合起来回答 node=calculate + error=KeyError
    之后紧接着的那个问题：**哪一步炸的**。

    为什么不记 traceback 原文：理由和 error_type 完全一样 ——
    traceback 里带着每一帧的源码行，而带源码行就意味着**可能带着当时的变量值**
    （"KeyError: '张伟'"），那正是这个项目最不该落盘的东西。
    这里只取三个**代码坐标**：哪个文件、第几行、哪个函数。
    它们和工资表里的任何数据都无关。

    取最里面那一帧、而不是我们自己的那一帧：
    最深那帧才是真的抛出来的地方。如果它在第三方库里
    （某个单元格是空的，openpyxl 自己炸了），你会直接看到它 ——
    那本身就是一条信息：不是我们的逻辑写错了，是数据碰到了边界。
    硬把它换成我们的调用点，反而把这条信息抹掉了。

    没有 traceback（异常对象是凭空造的，没人 raise 过）就是 None。
    """
    tb = getattr(error, "__traceback__", None)
    if tb is None:
        return None

    while tb.tb_next is not None:
        tb = tb.tb_next

    code = tb.tb_frame.f_code
    # 只取文件名，不取路径：完整路径里带着这台机器的目录结构
    # （C:\Users\谁\Desktop\...），对排查没有用，对「这份文件给谁看」有影响
    return f"{Path(code.co_filename).name}:{tb.tb_lineno} {code.co_name}"


def error_type(error) -> str:
    """异常 -> **只有它的类型名**。消息文本一个字都不记。

    这是这一版最要紧的一处改动。上一版记的是：

        f"{type(error).__name__}: {error}"

    想法是「类型加消息，够查了吧」。但异常消息里装什么，写 raise 的人说了算：
    "张三 的工资算不出来"、一整行单元格内容、带客户名的文件路径，
    都可能出现在里面 —— 而日志这一层**没有任何办法**可靠地判断
    哪条消息安全、哪条不安全。

    截断不是脱敏：把 "张三 的工资 12000 算不出来" 截到 300 字，
    张三还是张三，12000 还是 12000。

    所以只留类型名。定位够了 —— node=calculate + error=ValueError +
    task=a1b2c3 已经能让你找到那一行代码，剩下的事在调试器里重跑一遍就有。

    traceback 同样不记：它经常把当时的变量值一起打出来（"KeyError: '张伟'"），
    那正是这个项目最不该落盘的东西。
    """
    if error is None:
        return EMPTY

    if isinstance(error, BaseException):
        return type(error).__name__

    # 传了字符串之类的东西：把类型报出来，内容不记。
    # 「要写一句人看的错误说明」请走 message —— 那句话是调用方写的，
    # 由调用方负责它不含业务数据。
    return f"<{type(error).__name__}>"


if __name__ == "__main__":
    # 跑一遍看看：python logger.py
    #
    # 这段照着 payroll_agent.run() 的形状写：发一个 task_id、绑到执行流上、
    # 任务开头记一笔、节点进出各记一笔、收尾记一笔。
    # 跑完去 logs/agent.log 看，那几行就是文件出口的样本。
    #
    # 这里**不读真表、不调 LLM**，用的全是编的元数据。

    # ---- 正常跑完的一次 ----
    #
    # 节点那几行**都没有传 task_id** —— task 那一格照样是下面这个 id。
    # 这就是「接入 Logger 不用改九个节点」的实现方式。
    ok_id = new_task_id()

    with task_scope(ok_id):
        task_started(ok_id, message="开始跑工资流程（示例，不读真表）")

        node_started("parse")
        node_completed(
            "parse",
            message=f"进 {describe({'file_path': 'data/payroll_demo.xlsx', 'raw_rows': [{}, {}, {}]})}",
        )

        node_started("extract")
        node_completed("extract", message="交给 memory_lookup")

        task_completed(ok_id)

    # ---- 出事的一次 ----
    #
    # 异常里那句「张三 的工资 12000 元算不出来」一个字都不会进日志。
    bad_id = new_task_id()

    with task_scope(bad_id):
        task_started(bad_id)
        node_started("calculate")
        node_failed(
            "calculate",
            message="节点抛出异常",
            error=ValueError("张三 的工资 12000 元算不出来"),
        )
        task_failed(
            bad_id,
            message="流程抛出未捕获异常",
            error=ValueError("张三 的工资 12000 元算不出来"),
        )

    # ---- 手滑把整个对象传进来 ----
    #
    # 落盘的是 "<dict>"，不是里面的姓名和金额。
    # 这是本模块和「随便写个 print」最大的区别。
    node_started("human_review", message={"姓名": "张三", "基本工资": 12000})

    # ---- 不在任何 task_scope 里 ----
    #
    # task 那一格是占位符，不崩、不丢行。
    log_event(NODE_STARTED)

    # ---- 节点轨迹：整个 State 丢进去，出来的是形状 ----
    #
    # 这是这个模块第二个出口的演示，也是它的核心主张：
    # 上面那个 dict 里装着姓名和身份证号，落盘的却是 "list(1)"、"str(18)"。
    # 对比一下这两个文件，就知道为什么轨迹可以放心收整个 State。
    with task_scope(new_task_id()):
        node_trace(
            "parse",
            {
                "file_path": "data/payroll_demo.xlsx",
                "raw_rows": [{"姓名": "张三", "身份证号": "110101199001011234"}],
                "memory_context": {"approved_rules": [], "historical_rules": []},
                # 空的那几格不会出现在轨迹里 —— 不报本身就是答案
                "problems": [],
                "halt_reason": None,
            },
            changed=["raw_rows"],
            next_action="extract",
        )

    print(f"演示日志已写入：{LOG_FILE}")
    print(f"演示轨迹已写入：{TRACE_FILE}")
