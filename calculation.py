"""calculation.py —— 计算员。

职责：拿 Extraction 标准化好的数据，用固定的 Python 业务规则算出工资。

    应发       = 基本工资 + 奖金 - 扣款
    社保       = 基本工资 x 22.5%
    应纳税所得 = max(0, 应发 - 5000 - 社保)
    个税       = 七级超额累进
    实发       = 应发 - 社保 - 个税

刻意不做的：

    不读文件、不认字段叫什么（那是 parser / extraction 的事）、
    不判断结果对不对（那是 validation 的事）、不调用 LLM。

    这里只有「怎么算」，没有「算得对不对」。
"""

import math


# ============================================================
# 业务规则参数
#
# 全部集中在这一段：以后政策变了、换个城市、改个比例，
# 只改这里，不用往下翻代码。
# ============================================================

# 社保个人缴纳比例：养老 8% + 医疗 2% + 失业 0.5% + 公积金 12%
SOCIAL_RATE = 0.225

# 个税起征点（每月）
TAX_THRESHOLD = 5000.0

# 个税七级超额累进税率表：(本级距上限, 税率, 速算扣除数)
#
# 上限是 None 表示「以上」，也就是最后一档没有天花板。
# 有了速算扣除数，就不用真的一级一级分段累加 ——
# 找到落在哪一档，直接 应纳税所得 x 税率 - 速算扣除数。
TAX_BRACKETS = (
    (3000.0, 0.03, 0.0),
    (12000.0, 0.10, 210.0),
    (25000.0, 0.20, 1410.0),
    (35000.0, 0.25, 2660.0),
    (55000.0, 0.30, 4410.0),
    (80000.0, 0.35, 7160.0),
    (None, 0.45, 15160.0),
)

# 算工资必须先拿到这些输入，缺一个就没法算。
#
# 注意：validation.py 里也有一份同名常量。
# 那不是「抄过来保持一致」，而是检查员自己声明一遍它认为需要什么 ——
# 两边是各自独立的主张，不是同一份定义的两个引用。
#
# 但要说清楚：这份重复本身**抓不到** calculation 漏用某个字段。
# validation 用它查的是「表里有没有这一列」，跟「我有没有用这一列」是两回事。
# 真能抓到算错的，是 validation 里的 source_match（跟原始文件对账）。
REQUIRED_INPUTS = ("base_salary", "bonus", "deduction")


def calculate_payroll(rows: list) -> dict:
    """给每个人算工资。

    输入：
        rows: extraction.extract_rows(...)["rows"]

    输出：
        {
            "people": [
                {"emp_id": "A001", "name": "张伟", "department": "研发部",
                 "base_salary": 12000.0, "bonus": 3000.0, "deduction": 0.0,
                 "gross": 15000.0, "social": 2700.0,
                 "taxable": 7300.0, "tax": 520.0, "net_pay": 11780.0},
                ...
            ],
            "skipped": [
                {"emp_id": "A003", "name": "王强", "reason": "缺 base_salary"},
                ...
            ],
        }

    算不了的人**不进 people，进 skipped**，并且带上原因。
    宁可这一行没有结果，也不要给一个「看着像真的、其实是拿 0 凑的」数字 ——
    那种错 validation 根本发现不了。
    """
    people, skipped = [], []

    for row in rows:
        person, reason = calculate_one(row)

        if person is None:
            skipped.append(
                {
                    "emp_id": row.get("emp_id"),
                    "name": row.get("name", ""),
                    "reason": reason,
                }
            )
            continue

        people.append(person)

    return {"people": people, "skipped": skipped}


def calculate_one(row: dict) -> tuple[dict | None, str | None]:
    """算一个人的工资。

    返回 (结果, None) 或者 (None, 算不了的原因)。
    """
    # ---- 1. 关键输入齐不齐、能不能用 ----
    values = {}

    for field in REQUIRED_INPUTS:
        if field not in row:
            return None, f"缺 {field}"

        if not _is_money(row[field]):
            return None, f"{field} 的值 {row[field]!r} 不是数字"

        values[field] = float(row[field])

    # ---- 2. 按规则算 ----
    base = values["base_salary"]

    gross = base + values["bonus"] - values["deduction"]
    social = round(base * SOCIAL_RATE, 2)

    # 起征点和社保都能抵税。
    # 用 max(0, ...) 兜底：工资低于起征点时不是「负税」，是不用交。
    taxable = max(0.0, gross - TAX_THRESHOLD - social)

    tax = round(_income_tax(taxable), 2)

    # ---- 3. 每一步都舍入到分 ----
    #
    # 钱只到分。不舍入的话 12000 x 0.225 这类乘法会攒出一串
    # 0.0000000001 的尾巴，跟表里写的数永远对不上。
    return (
        {
            "emp_id": row.get("emp_id"),
            "name": row.get("name", ""),
            "department": row.get("department", ""),
            # 把输入也带上：以后生成凭证、人工复核时，
            # 一眼能看到这个实发是从哪三个数推出来的
            "base_salary": base,
            "bonus": values["bonus"],
            "deduction": values["deduction"],
            # 计算链
            "gross": round(gross, 2),
            "social": social,
            "taxable": round(taxable, 2),
            "tax": tax,
            "net_pay": round(gross - social - tax, 2),
        },
        None,
    )


# ============================================================
# 内部工具函数
# ============================================================


def _income_tax(taxable: float) -> float:
    """按七级超额累进算个税。

    找到应纳税所得落在哪一档，返回 应纳税所得 x 税率 - 速算扣除数。
    """
    for upper, rate, quick in TAX_BRACKETS:
        if upper is None or taxable <= upper:
            return taxable * rate - quick

    # 走不到这儿：最后一档 upper 是 None，兜住所有情况。
    # 留着是为了万一有人改了 TAX_BRACKETS 忘了最后那档。
    raise ValueError("税率表里没有兜底档，检查 TAX_BRACKETS 最后一行的上限是不是 None")


def _is_money(value) -> bool:
    """是不是一个能参与金额计算的数。

    bool 是 int 的子类，不先挡掉的话 True 会被当成 1.0 混过去。

    nan / inf 也要挡：nan 一路算下去会污染整条计算链，
    最后算出个 net_pay = nan，看着有值，其实什么都不是。
    """
    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    return math.isfinite(value)


if __name__ == "__main__":
    # 真跑一遍整条链：parser -> extraction -> calculation -> validation
    #
    # 用法：python calculation.py [xlsx路径]
    import json
    import sys

    # Windows 控制台默认按 GBK 输出，中文和特殊字符会变成乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from extraction import extract_rows
    from parser import read_payroll_excel
    from validation import validate_payroll

    target = sys.argv[1] if len(sys.argv) > 1 else "data/payroll_demo.xlsx"

    parsed = read_payroll_excel(target)
    if not parsed["ok"]:
        print(f"解析失败：{parsed['error']}")
        raise SystemExit(1)

    extracted = extract_rows(parsed["rows"])
    result = calculate_payroll(extracted["rows"])

    print("== calculation ==")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n== validation ==")
    print(
        json.dumps(
            validate_payroll(extracted["rows"], result["people"]),
            ensure_ascii=False,
            indent=2,
        )
    )
