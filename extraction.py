"""extraction.py —— 信息提取员。

职责：把 Parser 搬出来的原始字段，理解成我们的标准业务字段。

分两步：

    1. 认字段      "固定薪资" / "底薪" / "基本工资"  ->  base_salary
    2. 按类型清洗   base_salary 是金额 -> 洗干净变成数字
                   name 是文本        -> 去掉首尾空格，保持字符串

为什么清洗也在这儿？
    因为「这一列是金额还是文本」本身就是字段含义的一部分。
    Parser 不认识 base_salary，自然也不知道该怎么洗它。

刻意不做的：
    不读文件、不算工资、不校验数据对不对、不调用 LLM。
    V1 只用 Python 固定映射规则，认不出来就记下来，绝不猜。
"""

# 标准字段 -> 它所有可能出现的原始写法
#
# 以后接新客户，只需要往这里加一个同义词，不用改任何逻辑。
FIELD_MAP = {
    "emp_id": ["工号", "员工编号", "员工号", "编号"],
    "name": ["姓名", "员工姓名", "名字"],
    "department": ["部门", "所属部门", "所在部门"],
    "base_salary": ["基本工资", "固定薪资", "底薪", "固定工资"],
    "bonus": ["奖金", "绩效奖金", "绩效", "津贴"],
    "deduction": ["扣款", "扣除", "罚款"],
    "net_pay_in_file": ["实发工资", "实发", "到手工资"],
}

# 标准字段 -> 值类型。决定这个字段的值怎么洗。
FIELD_TYPES = {
    "emp_id": "text",
    "name": "text",
    "department": "text",
    "base_salary": "money",
    "bonus": "money",
    "deduction": "money",
    "net_pay_in_file": "money",
}

# 反过来查：原始写法 -> 标准字段
# 上面按「标准」分组是为了好维护，这里摊平成一张查找表给代码用。
LOOKUP = {
    raw: standard
    for standard, raw_names in FIELD_MAP.items()
    for raw in raw_names
}

# 金额里常见的噪音字符，转数字前先剥掉
NUMBER_NOISE = ("￥", "¥", ",", "，", "元", " ")

# 认不出来的字段放到输出行的这个键下面。
# 用下划线开头，避免和标准字段重名。
UNMAPPED_KEY = "unmapped"


def extract_row(raw_row: dict, problems: list | None = None) -> dict:
    """一行的原始字段 -> 标准字段（含清洗）。

    输入：
        {"姓名": "张三", "固定薪资": "¥12,000", "绩效奖金": 3000}

    输出：
        {"name": "张三", "base_salary": 12000.0, "bonus": 3000.0}

    认不出来的字段不猜、也不丢，原样放进 unmapped 里跟着这一行走。
    """
    if problems is None:
        problems = []

    result = {}
    unmapped = {}

    for raw_name, value in raw_row.items():
        standard = LOOKUP.get(str(raw_name).strip())

        if standard is None:
            # 认不出来 -> 不猜它是什么，也不猜它该怎么洗，原样留着
            unmapped[raw_name] = value
        else:
            result[standard] = _clean_value(standard, value, problems)

    # 不静默丢弃：有认不出的字段就挂到这一行上
    if unmapped:
        result[UNMAPPED_KEY] = unmapped

    return result


def extract_rows(raw_rows: list) -> dict:
    """把一批原始行转换成标准字段。

    输入：Parser 读出来的行，键是原始列名。
    输出：
        {
            "rows":           [ {标准字段}, ... ],
            "unknown_fields": [原始列名, ...],   # 认不出来的，去重后的清单
            "problems":       [str, ...],        # 清洗中记下的问题
        }

    unknown_fields 是个总览，方便一眼看出「这张表有哪些列我不认识」，
    好决定要不要往 FIELD_MAP 里补同义词。
    """
    rows = []
    unknown_fields = []
    problems = []

    for index, raw_row in enumerate(raw_rows, start=1):
        row = extract_row(raw_row, problems)

        for name in row.get(UNMAPPED_KEY, {}):
            if name not in unknown_fields:
                unknown_fields.append(name)

        rows.append(row)

    return {
        "rows": rows,
        "unknown_fields": unknown_fields,
        "problems": problems,
    }


# ============================================================
# 内部工具函数
# ============================================================


def _clean_value(standard: str, value, problems: list):
    """按标准字段的类型清洗值。

    text  -> 去掉首尾空格；空值变 ""
    money -> 变成 float；空值当 0
    """
    if FIELD_TYPES.get(standard) == "money":
        return _to_number(standard, value, problems)

    if _is_blank(value):
        return ""
    return str(value).strip()


def _is_blank(value) -> bool:
    """空值判断：None 或者去空格后为空字符串。"""
    return value is None or str(value).strip() == ""


def _to_number(field: str, value, problems: list) -> float:
    """把金额洗成 float。

    空值当 0（奖金、扣款空着很正常）；
    洗不出来的脏数据不抛异常，而是记一笔账到 problems 里按 0 处理 ——
    一个金额填错，不该让整张表读不出来。
    """
    if _is_blank(value):
        return 0.0

    # bool 是 int 的子类，不先挡掉的话 True 会静默变成 1.0 混进合计
    if isinstance(value, bool):
        problems.append(f"{field} 的值是布尔值 {value!r}，已按 0 处理")
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    for junk in NUMBER_NOISE:
        text = text.replace(junk, "")

    if text == "":
        return 0.0

    try:
        return float(text)
    except ValueError:
        problems.append(f"{field} 的值 {value!r} 不是数字，已按 0 处理")
        return 0.0


if __name__ == "__main__":
    # 验证用：python extraction.py
    import json

    sample = [
        # 1. 你给的例子：不同叫法 -> 同一标准字段
        {"姓名": "张三", "固定薪资": 12000, "绩效奖金": 3000, "扣款": 500},
        # 2. 认不出的字段：不猜、不丢
        {"姓名": "李四", "底薪": 8000, "奖金": 2000, "餐补": 300},
        # 3. 金额带噪音 + 空值
        {"姓名": "王五", "基本工资": "¥20,000", "奖金": None, "扣款": "800元"},
        # 4. 洗不出来的脏数据
        {"姓名": "赵六", "基本工资": "面议", "奖金": 1000},
    ]
    print(json.dumps(extract_rows(sample), ensure_ascii=False, indent=2))
