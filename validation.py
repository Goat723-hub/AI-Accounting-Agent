"""validation.py —— 检查员。

一句话说清定位：

    Calculation 负责「按固定规则生产结果」。
    Validation  负责「判断这个结果是否可信」。

**Validation 不是第二个 Calculation。**

    它不复制工资公式，也不重新计算 net_pay。
    如果两边用同样的输入、同样的规则、同样的实现，
    Validation 只能证明「代码运行一致」，证明不了「算得对」。
    抄一遍再比对，抄错的地方两边一起错，还查不出来。

Validation 的目标不是产生答案，是验证答案可不可信。

所以它由五个**互相独立**的检查组成，每个只回答一个问题：

    validate_inputs                  输入够不够 Calculation 用
    validate_output_structure        输出符不符合约定的数据结构
    validate_calculation_consistency 输出内部自不自洽
    validate_source_match            跟原始文件对得上吗   ← 唯一的外部证据
    validate_business_constraints    业务上该不该出现

五个函数各跑各的，谁也不知道谁存在，全部跑完再汇总 ——
不是发现第一个错误就停。

另外有**两条互不相干的通道**：

    errors     挡路的问题，passed = (errors 为空)
    warnings   提醒，不影响 passed

    分开放，是因为它们要的是两种处理：错误必须有人管，
    提醒可以就这么过。混在一起，等于逼着人把提醒也当错误处理。

刻意不做的：

    不读文件、不重新计算、不修改 Calculation 的输出、
    不为了让结果通过而自动修正、不调用 LLM。

    修不修、谁来修，是 Human Review 的事。
"""

import math


# ============================================================
# 约定：字段名 / 结构 / 阈值
# ============================================================

# Calculation 算工资必须先拿到这些输入，缺一个就没法算。
#
# 注意：calculation.py 里也有一份同名常量。
# 这份不是「抄过来保持一致」，而是「检查员自己声明一遍它认为需要什么」。
# 真要有哪边改了，这里报出来了才看得见。
REQUIRED_INPUTS = ("base_salary", "bonus", "deduction")

# 用哪个字段把「表里的一行」和「算出来的一行」对上。
MATCH_KEY = "emp_id"

# 表里原本写的实发工资，在标准字段里叫什么。
SOURCE_NET_PAY = "net_pay_in_file"

# Calculation 的结果里，实发工资叫什么。
CALC_NET_PAY = "net_pay"

# Calculation 结果里，除了实发之外的中间结果。
# 这几个缺了，就没法判断内部自洽不自洽。
CONSISTENCY_FIELDS = ("gross", "social", "taxable", "tax")

# 金额比较的容差。
#
# 浮点数不能直接拿 == 比：0.1 + 0.2 != 0.3。
# 一分钱以内算相等，超过就是真的对不上。
MONEY_TOLERANCE = 0.01

# 「扣款偏大」的提醒线：扣款占「基本工资 + 奖金」的比例上限。
#
# 这只是提醒，不是错误 —— 扣款大可能是正常的（补扣社保、补税、赔偿）。
# 超过就报一条 warning，让人看一眼，但不拦住流程。
#
# 0.5 = 扣款超过应发基础的一半就提醒。
# 这个数是业务判断，不是技术判断，要调就调这里。
#
# 注意：这个值必须小于 1.0。
# 等于 1.0 时它和「应发为负」是同一个条件（应发 = 基本工资 + 奖金 - 扣款），
# 而那种情况已经由 negative_gross 当错误报了，这条就永远不会触发。
DEDUCTION_WARNING_RATIO = 0.5


# 错误类型。
#
# 特意用字符串常量而不是枚举：输出要能直接 json.dumps 进 State / 落盘，
# 一眼看得懂，也方便 Agent 按 type 决定下一步该找谁。
E_MISSING_INPUT = "missing_input"                    # 输入缺字段
E_INVALID_INPUT = "invalid_input"                    # 输入在，但不是能用的数值
E_MISSING_EMP_ID = "missing_emp_id"                  # 输入行没有工号，认不出是谁
E_MISSING_CALC_RESULT = "missing_calculation_result"  # 表里有这个人，计算层根本没给结果
E_MISSING_CALC_FIELD = "missing_calculation_field"    # 结果在，但里面某个字段缺
E_DUPLICATE_KEY = "duplicate_emp_id"                 # 输出里工号重复
E_INCONSISTENT = "inconsistent_output"               # 输出内部自相矛盾
E_NET_PAY_MISMATCH = "net_pay_mismatch"              # 跟原始文件对不上
E_BUSINESS_VIOLATION = "business_violation"          # 业务上不该出现
E_NEGATIVE_GROSS = "negative_gross"                  # 应发为负

# 提醒类型。前缀 W_ 和错误分开，读日志时一眼分得清。
W_DEDUCTION_LARGE = "deduction_ratio_high"           # 扣款占比偏高


def validate_payroll(rows: list, people: list) -> dict:
    """跑完五个检查，汇总成 errors + warnings 两份结论。

    输入：
        rows:   extraction.extract_rows(...)["rows"]
        people: calculation.calculate_payroll(...)["people"]

    输出：
        {
            "passed": False,        # 只看 errors，跟 warnings 无关
            "errors": [ {...}, ... ],
            "warnings": [ {...}, ... ],
        }
    """
    errors = []

    # 五个检查互相独立，全部跑完再汇总。
    # 谁先谁后都不影响结果 —— 它们不共享状态，也不提前退出。
    errors += validate_inputs(rows)
    errors += validate_output_structure(rows, people)
    errors += validate_calculation_consistency(people)
    errors += validate_source_match(rows, people)
    errors += validate_business_constraints(rows)

    # 提醒走另一条通道，不进 errors，所以不影响 passed
    warnings = collect_warnings(rows)

    return {"passed": not errors, "errors": errors, "warnings": warnings}


# ============================================================
# 1. 输入
# ============================================================


def validate_inputs(rows: list) -> list:
    """检查 Extraction 的输出够不够 Calculation 用。

    只管「齐不齐、能不能参与计算」，不管「这个数对不对」——
    对不对是 business_constraints 和 source_match 的事。

    也不负责清洗：Extraction 已经标准化并清洗过了。
    这里再洗一遍，就等于把 Extraction 抄了第二份。
    """
    errors = []

    for row in rows:
        emp_id = row.get(MATCH_KEY) or None

        # 工号本身不是 Calculation 要的，但它是两边对账的钥匙。
        # 没有它，后面 source_match 根本认不出谁是谁。
        if not emp_id:
            errors.append({"type": E_MISSING_EMP_ID, "name": row.get("name", "")})

        for field in REQUIRED_INPUTS:
            if field not in row:
                errors.append(
                    {"type": E_MISSING_INPUT, "emp_id": emp_id, "field": field}
                )
            elif not _is_money(row[field]):
                errors.append(
                    {
                        "type": E_INVALID_INPUT,
                        "emp_id": emp_id,
                        "field": field,
                        "value": row[field],
                    }
                )

    return errors


# ============================================================
# 2. 输出结构
# ============================================================


def validate_output_structure(rows: list, people: list) -> list:
    """检查 Calculation 的输出符不符合约定的数据结构。

    要两个参数，因为「结构对不对」有两面：

        对着 people 自己看   该有的字段在不在、工号重不重复
        对着 rows 比着看     表里有的人，是不是都有结果

    只看接口，不判断工资算得对不对 —— 那是另外三个函数的事。
    这一层的意思是「你有没有按说好的格式交货」。
    """
    errors = []
    seen = set()

    # ---- 一面：交上来的东西本身合不合格 ----
    for person in people:
        emp_id = person.get(MATCH_KEY)

        if not emp_id:
            errors.append({"type": E_MISSING_CALC_FIELD, "field": MATCH_KEY})
        elif emp_id in seen:
            # 工号重复意味着有人被覆盖 —— 一整份工资凭空消失
            errors.append({"type": E_DUPLICATE_KEY, "emp_id": emp_id})
        else:
            seen.add(emp_id)

        if not _is_money(person.get(CALC_NET_PAY)):
            errors.append(
                {
                    "type": E_MISSING_CALC_FIELD,
                    "emp_id": emp_id,
                    "field": CALC_NET_PAY,
                }
            )

    # ---- 另一面：该交的人是不是都交了 ----
    by_id = _index_by_emp_id(people)

    for row in rows:
        emp_id = row.get(MATCH_KEY) or None

        # 没工号 -> validate_inputs 已经报过，这里认不出也没法认
        if not emp_id:
            continue

        if emp_id not in by_id:
            errors.append(
                {
                    "type": E_MISSING_CALC_RESULT,
                    "emp_id": emp_id,
                    "name": row.get("name", ""),
                }
            )

    return errors


# ============================================================
# 3. 内部自洽
# ============================================================


def validate_calculation_consistency(people: list) -> list:
    """检查 Calculation 输出的**内部自洽性**。

    这里刻意不重算工资。

    不写 net_pay = gross - social - tax，因为那等于把 Calculation 抄一遍：
    Calculation 要是把公式写错了，抄的这份一样错，对不出来。

    这里只找**明显矛盾** —— 全是**不等式**，不是公式。
    公式说「等于几」，不等式只说「不可能超过谁」。
    不知道税率是多少，照样能判断「社保比应发还多」是错的。
    """
    errors = []

    for person in people:
        emp_id = person.get(MATCH_KEY)

        # 中间结果缺了，后面的关系检查无从谈起
        missing = [f for f in CONSISTENCY_FIELDS if not _is_money(person.get(f))]
        if missing:
            errors.append(
                {
                    "type": E_MISSING_CALC_FIELD,
                    "emp_id": emp_id,
                    "field": "、".join(missing),
                }
            )
            continue

        # 实发缺了，output_structure 那边已经报过，这里不重复报
        net_pay = person.get(CALC_NET_PAY)
        if not _is_money(net_pay):
            continue

        gross = float(person["gross"])
        social = float(person["social"])
        tax = float(person["tax"])
        net_pay = float(net_pay)

        # ---- 下面全是「这不可能」级别的不合理，不涉及任何公式 ----

        if net_pay <= 0:
            errors.append(
                _inconsistent(emp_id, CALC_NET_PAY, net_pay, "实发工资不可能小于等于 0")
            )

        if net_pay > gross:
            errors.append(
                _inconsistent(
                    emp_id, CALC_NET_PAY, net_pay, "实发比应发还多，说明扣减没生效或符号写反了"
                )
            )

        if social < 0 or tax < 0:
            errors.append(
                _inconsistent(emp_id, "social/tax", [social, tax], "扣减项不可能为负")
            )

        if social + tax > gross:
            errors.append(
                _inconsistent(
                    emp_id, "social+tax", social + tax, "社保 + 个税比应发还多"
                )
            )

    return errors


# ============================================================
# 4. 跟原始文件对账
# ============================================================


def validate_source_match(rows: list, people: list) -> list:
    """拿原始文件里的实发工资，跟系统算出来的对账。

    这是 Validation **唯一的外部证据**。

    net_pay_in_file 是客户 HR 填的，我们的代码从来没碰过它。
    前面三个检查说到底都只是在看「我们自己的东西自不自洽」；
    只有这里引入了系统之外的参照物 —— 算错了，是它发现的。
    """
    errors = []
    by_id = _index_by_emp_id(people)

    for row in rows:
        source = row.get(SOURCE_NET_PAY)

        # 表里没写这一列（或者空着）-> 没得对，不是错误。
        # 这张表有没有「实发工资」列，是客户表格自己的事。
        if _is_blank(source):
            continue

        # 认不出人 / 没算出来 -> validate_inputs 和 output_structure 已经报过了，
        # 这里再报一遍只是把同一个问题说三次。
        emp_id = row.get(MATCH_KEY) or None
        person = by_id.get(emp_id)
        if person is None:
            continue

        if not _is_money(source):
            errors.append(
                {
                    "type": E_INVALID_INPUT,
                    "emp_id": emp_id,
                    "field": SOURCE_NET_PAY,
                    "value": source,
                }
            )
            continue

        calculated = person.get(CALC_NET_PAY)
        if not _is_money(calculated):
            continue

        calculated = float(calculated)
        source = float(source)

        # 用容差比，不用 ==。
        # difference 用没舍入过的原始差值判断，只在写进结果时才舍入到分。
        raw_diff = calculated - source

        if abs(raw_diff) > MONEY_TOLERANCE:
            errors.append(
                {
                    "type": E_NET_PAY_MISMATCH,
                    "emp_id": emp_id,
                    "name": row.get("name", ""),
                    "calculated": calculated,
                    "source": source,
                    "difference": round(raw_diff, 2),
                }
            )

    return errors


# ============================================================
# 5. 业务约束
# ============================================================


def validate_business_constraints(rows: list) -> list:
    """业务上根本不该出现的数。

    这些不是「算错了」—— 计算可能一点毛病没有，
    但输入本身就是业务上不成立的（比如基本工资是负的）。
    这类问题 Calculation 修不了，得回去找客户。
    """
    errors = []

    for row in rows:
        emp_id = row.get(MATCH_KEY) or None

        # ---- 金额不能是负的 ----
        for field in REQUIRED_INPUTS:
            value = row.get(field)
            if _is_money(value) and value < 0:
                errors.append(
                    {
                        "type": E_BUSINESS_VIOLATION,
                        "emp_id": emp_id,
                        "field": field,
                        "value": float(value),
                        "reason": f"{field} 不能是负数",
                    }
                )

        # ---- 应发不能是负的 ----
        #
        # 这里算了一次 基本工资 + 奖金 - 扣款。
        # 这不是「重算 Calculation」—— 只是判断「应发」这个概念成不成立，
        # 不涉及社保、个税、税率表里的任何东西。
        # 应发为负说明扣款比该发出去的还多，属于业务上不成立，得拦下来。
        base, bonus, deduction = (
            row.get("base_salary"),
            row.get("bonus"),
            row.get("deduction"),
        )
        if not all(_is_money(v) for v in (base, bonus, deduction)):
            continue

        gross = float(base) + float(bonus) - float(deduction)

        if gross < 0:
            errors.append(
                {
                    "type": E_NEGATIVE_GROSS,
                    "emp_id": emp_id,
                    "gross": round(gross, 2),
                    "reason": "应发为负：扣款超过了「基本工资 + 奖金」",
                }
            )

    return errors


# ============================================================
# 提醒
# ============================================================


def collect_warnings(rows: list) -> list:
    """提醒，不是错误。不影响 passed。

    和五个检查分开，是因为它们要的处理方式不一样：
    错误必须有人管，提醒看一眼就行。

    「扣款偏大」放这儿不放 errors，是因为扣款大未必是错的 ——
    补扣社保、补缴个税、离职赔偿，都会让某一行的扣款特别大。
    报出来让人看一眼，但不该拦住整个流程。
    """
    warnings = []

    for row in rows:
        base, bonus, deduction = (
            row.get("base_salary"),
            row.get("bonus"),
            row.get("deduction"),
        )
        if not all(_is_money(v) for v in (base, bonus, deduction)):
            continue

        base, bonus, deduction = float(base), float(bonus), float(deduction)
        ceiling = (base + bonus) * DEDUCTION_WARNING_RATIO

        # 应发为负已经在 business_constraints 里当错误报了，
        # 这里不重复提醒 —— 一个数只有一次发言权。
        if base + bonus - deduction < 0:
            continue

        if deduction > ceiling:
            warnings.append(
                {
                    "type": W_DEDUCTION_LARGE,
                    "emp_id": row.get(MATCH_KEY),
                    "name": row.get("name", ""),
                    "deduction": deduction,
                    "ceiling": round(ceiling, 2),
                    "reason": (
                        f"扣款 {deduction} 超过应发基础 "
                        f"{base + bonus} 的 {DEDUCTION_WARNING_RATIO:.0%}"
                    ),
                }
            )

    return warnings


# ============================================================
# 内部工具函数
# ============================================================


def _inconsistent(emp_id, field: str, value, reason: str) -> dict:
    """内部自洽检查的错误都长一个样，统一在这儿造。"""
    return {
        "type": E_INCONSISTENT,
        "emp_id": emp_id,
        "field": field,
        "value": value,
        "reason": reason,
    }


def _index_by_emp_id(people: list) -> dict:
    """把 Calculation 的结果按工号索引起来，方便一个个找到人。

    这里不管工号重不重复 —— 那是 output_structure 的职责。
    重复时后一个覆盖前一个，跟那边报出来的口径一致就行。
    """
    return {person.get(MATCH_KEY): person for person in people}


def _is_money(value) -> bool:
    """是不是一个能参与金额计算的数。

    bool 是 int 的子类，不先挡掉的话 True 会被当成 1.0 混过去。

    nan / inf 也要挡：nan 和任何数比较都是 False，
    放进来会让对账静默失效 —— 看着没错，其实根本没比过。
    """
    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    return math.isfinite(value)


def _is_blank(value) -> bool:
    """空值判断：None 或者去空格后为空字符串。"""
    return value is None or str(value).strip() == ""


if __name__ == "__main__":
    # 验证用：python validation.py [xlsx路径]
    #
    # 分两段：
    #   一、真实数据整条链跑一遍
    #   二、手工构造坏数据，把五个检查各触发一次
    import json
    import sys

    # Windows 控制台默认按 GBK 输出，中文会变成乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from calculation import calculate_payroll
    from extraction import extract_rows
    from parser import read_payroll_excel

    target = sys.argv[1] if len(sys.argv) > 1 else "data/payroll_demo.xlsx"

    # ---------- 一、真实数据 ----------
    parsed = read_payroll_excel(target)
    if not parsed["ok"]:
        print(f"解析失败：{parsed['error']}")
        raise SystemExit(1)

    extracted = extract_rows(parsed["rows"])
    calculated = calculate_payroll(extracted["rows"])

    print("== 真实数据 ==")
    print(
        json.dumps(
            validate_payroll(extracted["rows"], calculated["people"]),
            ensure_ascii=False,
            indent=2,
        )
    )

    # ---------- 二、坏数据：五个检查逐个看 ----------
    rows = [
        # 正常
        {"emp_id": "A001", "name": "张伟", "base_salary": 12000.0, "bonus": 3000.0,
         "deduction": 0.0, "net_pay_in_file": 11780.0},
        # 缺 base_salary 这一列
        {"emp_id": "A002", "name": "李娜", "bonus": 5000.0, "deduction": 200.0,
         "net_pay_in_file": 10000.0},
        # 没工号
        {"name": "孙七", "base_salary": 5000.0, "bonus": 0.0, "deduction": 0.0,
         "net_pay_in_file": 5000.0},
        # 基本工资是负数 + 应发为负（扣款比应发还多）
        {"emp_id": "A004", "name": "赵敏", "base_salary": -100.0, "bonus": 0.0,
         "deduction": 99999.0, "net_pay_in_file": 5145.5},
        # 表里写 1000，系统算出来 5897.25 -> 只有 source_match 抓得到
        {"emp_id": "A007", "name": "郑十", "base_salary": 7000.0, "bonus": 500.0,
         "deduction": 0.0, "net_pay_in_file": 1000.0},
        # 应发仍为正（2000），但扣款 3000 超过应发基础 5000 的一半
        # -> 只报 warning，不报 error
        {"emp_id": "A008", "name": "冯十一", "base_salary": 5000.0, "bonus": 0.0,
         "deduction": 3000.0, "net_pay_in_file": 875.0},
    ]

    people = [
        # 正常
        {"emp_id": "A001", "name": "张伟", "gross": 15000.0, "social": 2700.0,
         "taxable": 7300.0, "tax": 520.0, "net_pay": 11780.0},
        # 缺 net_pay
        {"emp_id": "A002", "name": "李娜", "gross": 12800.0, "social": 1800.0,
         "taxable": 6000.0, "tax": 390.0},
        # 工号重复（内容一样，免得干扰 source_match 的判断）
        {"emp_id": "A001", "name": "张伟（重复）", "gross": 15000.0, "social": 2700.0,
         "taxable": 7300.0, "tax": 520.0, "net_pay": 11780.0},
        # 实发比应发还多 + 社保个税加起来比应发还多
        {"emp_id": "A005", "name": "周八", "gross": 1000.0, "social": 5000.0,
         "taxable": 0.0, "tax": 5000.0, "net_pay": 20000.0},
        # 缺中间结果 gross
        {"emp_id": "A006", "name": "吴九", "social": 100.0, "taxable": 0.0,
         "tax": 10.0, "net_pay": 900.0},
        # 内部完全自洽（前四个检查全过），但跟表里写的对不上
        {"emp_id": "A007", "name": "郑十", "gross": 7500.0, "social": 1575.0,
         "taxable": 925.0, "tax": 27.75, "net_pay": 5897.25},
        # 内部自洽，跟表里也对得上 —— 只贡献一条 warning
        {"emp_id": "A008", "name": "冯十一", "gross": 2000.0, "social": 1125.0,
         "taxable": 0.0, "tax": 0.0, "net_pay": 875.0},
    ]

    checks = [
        ("1. validate_inputs", validate_inputs(rows)),
        ("2. validate_output_structure", validate_output_structure(rows, people)),
        ("3. validate_calculation_consistency", validate_calculation_consistency(people)),
        ("4. validate_source_match", validate_source_match(rows, people)),
        ("5. validate_business_constraints", validate_business_constraints(rows)),
    ]

    print("\n== 五个检查，各跑各的 ==")
    for title, found in checks:
        print(f"\n-- {title}：{len(found)} 条 --")
        print(json.dumps(found, ensure_ascii=False, indent=2))

    print("\n-- collect_warnings（提醒，不影响 passed）--")
    print(json.dumps(collect_warnings(rows), ensure_ascii=False, indent=2))

    print("\n-- 汇总 --")
    print(
        json.dumps(
            validate_payroll(rows, people), ensure_ascii=False, indent=2
        )
    )
