"""memory.py —— 规则记忆员。

职责：保存和查询**已确认的字段映射规则**。

    extraction 认不出的字段（"餐补"），第一次要问 LLM、再让人确认；
    第二次就该直接查出来 —— 这个模块就是那个「记住」的地方。

刻意不做的：

    不参与工资计算、不改 calculation 的规则、不改 validation 的逻辑、
    不调用 LLM。

    最后一条尤其重要：Memory 一旦能自己产生规则，
    「人工确认」这道闸门就形同虚设了。

为什么不用向量数据库：

    这里要的是**确定性**。同一个字段名，今天查和明天查必须给出
    同一个答案。相似度检索做不到这一点 —— 它可能今天给 bonus、
    明天给 deduction，而且不告诉你它变了。

    字段映射是业务规则，不是语义相似度问题。
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from extraction import FIELD_MAP, UNMAPPED_KEY, extract_row


# 规则库落在项目根目录，和 memory.py 并排。
#
# 不放 data/ —— 那边装的是演示用的 xlsx（测试数据），
# 这里是运行时状态，性质不同，混一起以后不好清理。
MEMORY_PATH = Path(__file__).parent / "memory.json"

# 存储格式版本。以后加字段（比如规则生效时间）时有地方判断要不要迁移。
SCHEMA_VERSION = 1


# 规则状态。只有人能把 pending 推到 approved —— 这是整个 Memory 的安全边界。
PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"

# mapping 只能填这些标准字段名。
# 从 extraction 拿，不自己抄一份 —— 那边加了新字段这里自动跟上。
VALID_MAPPINGS = tuple(FIELD_MAP)


# ============================================================
# 三个核心操作
# ============================================================


def query_rule(field: str, statuses: tuple = (APPROVED,)) -> dict:
    """查一个字段的映射规则。

    默认只查**已批准**的（没批准的规则不能用）。

    但还有一个问题也需要问这里：「这个字段我见过没有」——
    见过的 pending 不该再问一遍 LLM。
    传 statuses=(PENDING, APPROVED) 就能问这个。

    返回：
        {"found": True, "mapping": "bonus", "status": "approved", "confidence": 0.95}
        {"found": False}
    """
    rule = _load()["rules"].get(_normalize(field))

    if rule is None or rule.get("status") not in statuses:
        return {"found": False}

    return {
        "found": True,
        "mapping": rule["mapping"],
        "status": rule["status"],
        "confidence": rule.get("confidence"),
    }


def save_candidate_rule(field: str, mapping: str, confidence: float) -> dict:
    """存一条 LLM 提的候选规则。状态是 pending，**不是** approved。

    LLM 的输出永远只是候选。它再怎么自信，也得等人点头。

    返回：
        {"saved": True, "field": "餐补", "status": "pending"}
        {"saved": False, "reason": "..."}
    """
    field = _normalize(field)

    if not field:
        return {"saved": False, "reason": "字段名是空的"}

    if mapping not in VALID_MAPPINGS:
        # 不拦的话，"餐补 -> 工资" 这种垃圾会安静地存进去，
        # 之后每张表都按它错算，而且没人知道是什么时候错的。
        return {
            "saved": False,
            "reason": f"{mapping!r} 不是标准字段，只能是 {list(VALID_MAPPINGS)}",
        }

    data = _load()
    existing = data["rules"].get(field)

    if existing:
        if existing["status"] == APPROVED:
            # 人工确认过的东西，机器不能覆盖
            return {
                "saved": False,
                "reason": (
                    f"{field} 已经有批准过的规则（-> {existing['mapping']}），"
                    f"LLM 的提议不能覆盖它"
                ),
            }

        if existing["status"] == REJECTED and existing["mapping"] == mapping:
            # 同样的提议再提一遍是浪费时间。但换个映射就是新信息了，放行。
            return {
                "saved": False,
                "reason": f"{field} -> {mapping} 之前被人工否过，不再重复提",
            }

    data["rules"][field] = {
        "field": field,
        "mapping": mapping,
        "status": PENDING,
        "confidence": confidence,
        "source": "llm",
        "created_at": _now(),
        "decided_at": None,
        "notes": "",
    }
    _save(data)

    return {"saved": True, "field": field, "status": PENDING}


def approve_rule(field: str) -> dict:
    """人工确认：pending -> approved。

    这是唯一能把规则变成「可用」的入口。

    返回：
        {"approved": True, "field": "餐补", "mapping": "bonus", "changed": True}
        {"approved": False, "reason": "..."}
    """
    return _decide(field, APPROVED)


def reject_rule(field: str, reason: str = "") -> dict:
    """人工否决：pending -> rejected。

    为什么要留这一态而不是直接删掉记录：
    删了的话，下次 LLM 还会提同样的建议，人还得再拒一次。
    留着 rejected，save_candidate_rule 就知道「这个提过了，别再来」。

    也对 approved 的规则生效 —— 人是可以撤回规则的。

    返回：
        {"rejected": True, "field": "餐补", "changed": True}
        {"rejected": False, "reason": "..."}
    """
    field = _normalize(field)

    result = _decide(field, REJECTED)

    if result.get("changed") and reason:
        data = _load()
        data["rules"][field]["notes"] = reason
        _save(data)

    return result


# ============================================================
# 把规则应用到数据上
# ============================================================


def apply_rules(rows: list, rules: list) -> dict:
    """把命中的规则应用到行上。

    规则说 餐补 -> bonus，就把每行 unmapped 里的「餐补」搬进 bonus。

    输入：
        rows:  extraction 输出的 rows
        rules: [{"field": "餐补", "mapping": "bonus"}, ...]

    输出：
        {"rows": [...], "problems": [...]}   —— 跟其他层的风格一致

    只在目标字段**原本不存在**时才写入。
    要是这行本来就有 bonus（表里已经有「奖金」列），就不覆盖，记一条 problem。
    两列都是奖金类收入，该相加还是该取哪个，是业务判断，
    不是 Memory 能替人决定的 —— 猜错了会安静地算错工资。
    """
    if not rules:
        return {"rows": rows, "problems": []}

    mapping_of = {rule["field"]: rule["mapping"] for rule in rules}

    new_rows = []
    problems = []

    for index, row in enumerate(rows, start=1):
        unmapped = row.get(UNMAPPED_KEY)

        if not unmapped:
            new_rows.append(row)
            continue

        new_row = dict(row)
        new_unmapped = dict(unmapped)

        for field, mapping in mapping_of.items():
            if field not in new_unmapped:
                continue

            if mapping in new_row:
                problems.append(
                    f"第 {index} 行：规则想把 {field} 映射到 {mapping}，"
                    f"但这一行已经有 {mapping} 了，不覆盖"
                )
                continue

            value = new_unmapped.pop(field)
            new_row[mapping] = _clean_like_extraction(mapping, value, problems)

        # 认出来的拿走了，剩下的还留在 unmapped 里 —— 不静默丢弃
        if new_unmapped:
            new_row[UNMAPPED_KEY] = new_unmapped
        else:
            new_row.pop(UNMAPPED_KEY, None)

        new_rows.append(new_row)

    return {"rows": new_rows, "problems": problems}


# ============================================================
# 内部工具函数
# ============================================================


def _decide(field: str, status: str) -> dict:
    """approve / reject 共用的落地逻辑。

    两个操作除了目标状态之外一模一样，没必要写两遍。
    """
    field = _normalize(field)
    data = _load()
    rule = data["rules"].get(field)

    if rule is None:
        return {
            status: False,
            "reason": f"没有 {field} 的候选规则，先 save_candidate_rule",
        }

    verb = {"approved": "approved", "rejected": "rejected"}[status]

    if rule["status"] == status:
        # 已经是这个状态了，重复操作不算错
        return _decision_result(verb, field, rule, changed=False)

    rule["status"] = status
    rule["decided_at"] = _now()
    _save(data)

    return _decision_result(verb, field, rule, changed=True)


def _decision_result(verb: str, field: str, rule: dict, changed: bool) -> dict:
    return {
        verb: True,
        "field": field,
        "mapping": rule["mapping"],
        "changed": changed,
    }


def _clean_like_extraction(mapping: str, value, problems: list):
    """借用 extraction 的清洗逻辑，不自己再写一遍。

    FIELD_MAP["bonus"][0] 是 "奖金"，所以把 {"奖金": "500元"} 丢给
    extract_row，它就会按 bonus 这个字段的类型把 500.0 洗出来。

    自己写一遍清洗（剥 ￥、剥「元」、挡 bool）= 把 extraction 抄第二份。
    哪天它改了，这里不会跟着改，而且不会有任何报错。

    传进去的 problems 列表会收到清洗过程中的问题
    （比如「值 '面议' 不是数字」），原样往上传。
    """
    raw_name = FIELD_MAP[mapping][0]
    cleaned = extract_row({raw_name: value}, problems)
    return cleaned.get(mapping)


def _normalize(field) -> str:
    """字段名归一化。

    parser 洗表头时会剥掉 BOM 和全角空格 —— 这两样肉眼看不见，
    但会让 "餐补" 和 "餐补 " 变成两个不同的键，
    人工批过的规则第二次就查不到了。
    """
    if field is None:
        return ""
    text = str(field)
    text = text.replace("﻿", "").replace("　", "")
    return text.strip()


def _now() -> str:
    """时间戳。规则是会变的，出事时要能查出「这条什么时候定的」。"""
    return datetime.now().isoformat(timespec="seconds")


def _empty() -> dict:
    return {"version": SCHEMA_VERSION, "rules": {}}


def _load() -> dict:
    """读规则库。

    文件不存在 = 还没记过任何规则，正常情况。
    文件存在但坏了 = 不正常，**不能**静默当空库处理 ——
    那样下一次保存会把它整个覆盖掉，所有已批准的规则一次全没。
    改成挪到一边留证据，流程继续。
    """
    if not MEMORY_PATH.exists():
        return _empty()

    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        try:
            MEMORY_PATH.replace(MEMORY_PATH.with_suffix(".json.corrupt"))
        except OSError:
            pass
        return _empty()

    if not isinstance(data, dict) or not isinstance(data.get("rules"), dict):
        return _empty()

    return data


def _save(data: dict) -> None:
    """写回磁盘。

    先写临时文件再 rename —— rename 在同一个分区上是原子操作。
    直接 open(..., "w") 写到一半崩了（断电、进程被杀），
    memory.json 会变成半截 JSON，下次读直接当损坏文件。

    临时文件必须和目标是同一个目录，跨分区 rename 就不是原子的了。
    """
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=MEMORY_PATH.parent,
        prefix=".memory-",
        suffix=".tmp",
        delete=False,
    )

    try:
        with handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(handle.name, MEMORY_PATH)
    except Exception:
        # 写失败就把临时文件收拾掉，别在项目里留垃圾
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


if __name__ == "__main__":
    # 验证用：python memory.py
    #
    # 完整走一遍「第一次遇到 -> 问 LLM -> 人工确认 -> 第二次直接命中」，
    # 看两次跑出来的结果差在哪。
    import shutil
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from openpyxl import Workbook

    # 演示用临时规则库，**不碰**你项目根目录里那份 memory.json
    #
    # 这里有个坑：python memory.py 跑起来时本文件是 __main__ 模块，
    # 不是 memory 模块。payroll_agent 里的 `from memory import query_rule`
    # 会**再导入一份**，两份各有各的 MEMORY_PATH 全局变量。
    # 所以必须改 memory 那一份 —— 改本文件顶部的那个，
    # 图读到的还是项目根目录（空的），演示会假装规则没生效。
    import memory

    tmp_dir = Path(tempfile.mkdtemp())
    memory.MEMORY_PATH = tmp_dir / "memory.json"

    # 这张表故意没有「奖金」列，只有一列「餐补」。
    # 所以只要规则没建立起来，bonus 就缺失，calculation 会跳过这个人。
    xlsx = tmp_dir / "demo.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["工号", "姓名", "部门", "基本工资", "扣款", "实发工资", "餐补"])
    ws.append(["A001", "张伟", "研发部", 12000, 0, 11780, 3000])
    wb.save(xlsx)
    wb.close()

    from payroll_agent import build_graph, initial_state

    def run_once(title: str) -> dict:
        print(f"\n{'=' * 60}")
        print(title)
        print("=" * 60)

        final = initial_state(str(xlsx))
        for step in build_graph().stream(initial_state(str(xlsx))):
            for node_name, update in step.items():
                final.update(update)
                print(f"  [{node_name:18}] next_action = {update.get('next_action')}")

        calc = final.get("calculation_result") or {}
        people = calc.get("people", [])

        print(f"\n  unknown_fields : {final.get('unknown_fields')}")
        print(f"  memory_rules   : {final.get('memory_rules')}")
        print(f"  算出几个人     : {len(people)}")
        for p in people:
            print(f"      {p['emp_id']} {p['name']}  bonus={p['bonus']}  net_pay={p['net_pay']}")
        print(f"  validation     : passed={final.get('validation_result', {}).get('passed')}")
        print(f"  problems       : {final.get('problems')}")
        return final

    # ---- 第一次遇到「餐补」----
    run_once("第一次遇到「餐补」：规则库里没有 -> 算不出工资")

    # [LLM + 人工] 这两层本模块不实现，直接调 API 把中间那步演出来
    print("\n  ---- 餐补：正常流程 ----")
    print(f"  save_candidate_rule  : {memory.save_candidate_rule('餐补', 'bonus', 0.95)}")
    print(f"  query_rule('餐补')   : {memory.query_rule('餐补')}   <- pending 不算数")
    print(f"  approve_rule         : {memory.approve_rule('餐补')}")
    print(f"  query_rule('餐补')   : {memory.query_rule('餐补')}")

    print("\n  ---- 交通费：被人工否掉的分支 ----")
    print(f"  save_candidate_rule  : {memory.save_candidate_rule('交通费', 'bonus', 0.8)}")
    print(f"  reject_rule          : {memory.reject_rule('交通费', '交通补贴不该算进奖金')}")
    print(f"  同样的提议再提一次   : {memory.save_candidate_rule('交通费', 'bonus', 0.8)}")
    print(f"  查 rejected 状态     : {memory.query_rule('交通费', statuses=('rejected',))}")
    print(f"  换个映射就是新信息   : {memory.save_candidate_rule('交通费', 'deduction', 0.7)}")

    print("\n  ---- 脏输入挡不挡得住 ----")
    print(f"  mapping 不是标准字段 : {memory.save_candidate_rule('房补', '工资', 0.9)}")
    print(f"  字段名是空的         : {memory.save_candidate_rule('  ', 'bonus', 0.9)}")
    print(f"  没候选就批准         : {memory.approve_rule('没这个字段')}")

    # ---- 第二次遇到「餐补」----
    final = run_once("第二次遇到「餐补」：规则命中 -> 直接算出来")

    # ---- apply_rules 的冲突分支 ----
    #
    # 上面那个 demo 表里只有「餐补」没有「奖金」，所以走的是「目标字段空缺，
    # 直接写进去」这条路。冲突那条路得单独造一行来试 ——
    # 它平时走不到，但走错一次就是安静地算错工资。
    print("\n  ---- apply_rules：这行本来就有 bonus ----")
    conflict_rows = [
        {"emp_id": "A002", "bonus": 1000.0, UNMAPPED_KEY: {"餐补": 300}},
    ]
    print(
        "  "
        + json.dumps(
            memory.apply_rules(conflict_rows, [{"field": "餐补", "mapping": "bonus"}]),
            ensure_ascii=False,
        )
    )

    print("\n  ---- 磁盘上留下了什么 ----")
    print((tmp_dir / "memory.json").read_text(encoding="utf-8"))

    # 演示用的临时目录连文件一起删掉。项目里不留任何演示残留。
    shutil.rmtree(tmp_dir, ignore_errors=True)
