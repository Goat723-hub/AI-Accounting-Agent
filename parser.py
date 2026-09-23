"""parser.py —— 文件处理员。

只做一件事：把工资表 xlsx **原样**读成 Python 的嵌套结构。

输出里行的键，就是表里原本的列名（"基本工资" 就是 "基本工资"），
值也是单元格里原本的值，不做任何转换。

刻意不做的：
    不认字段含义、不做中英映射、不清洗金额、不算工资、不校验、不调用 LLM。

    认识「固定薪资 其实等于 base_salary」，是 extraction.py 的事。
    Parser 只负责把格子搬出来，不负责理解它是什么。
"""

from pathlib import Path

from openpyxl import load_workbook


# 只用来「定位表头行」——判断哪一行长得像表头。
#
# 注意：这张表不决定任何列的含义，只用来找位置。
# 一个客户表里哪怕全是没见过的列名，只要还有几列眼熟，就能定位到表头。
HEADER_HINTS = {
    "工号", "员工编号", "员工号", "编号",
    "姓名", "员工姓名", "名字",
    "部门", "所属部门", "所在部门",
    "基本工资", "固定薪资", "底薪", "固定工资",
    "奖金", "绩效奖金", "绩效", "津贴",
    "扣款", "扣除", "罚款",
    "实发工资", "实发", "到手工资",
}

# 表头不一定在第 1 行（上面常有标题行、空行），往下扫这么多行去找
MAX_HEADER_SCAN_ROWS = 10

# 表头行至少命中这么多列，才算真的找到了表头
MIN_HEADER_HITS = 3


def read_payroll_excel(file_path: str) -> dict:
    """读工资表 xlsx，原样返回。

    约定：这个函数永远不抛异常。
    文件不存在、不是合法 xlsx、表头对不上，都返回 ok=False + error。

    返回：
        {
            "ok":       bool,
            "error":    str | None,
            "file":     str,
            "sheet":    str | None,
            "columns":  [列名, ...],            # 表头原文，按列顺序
            "rows":     [ {列名: 原始值}, ... ], # 键是表里原本的列名，值是原值
            "problems": [str, ...],
        }
    """
    path = Path(file_path)

    if not path.exists():
        return _fail(file_path, f"文件不存在：{file_path}，请确认路径或补充文件")

    try:
        # data_only=True 不能省：
        # 不加它，带公式的单元格返回的是字符串 "=SUM(D4:F4)"。
        wb = load_workbook(path, data_only=True)
    except Exception as e:
        return _fail(file_path, f"无法打开文件（可能不是 xlsx 或已损坏）：{e}")

    try:
        sheet = wb.active
        # values_only=True 直接给单元格的值，不给 Cell 对象
        all_rows = list(sheet.iter_rows(values_only=True))

        if not all_rows:
            return _fail(file_path, "工作表是空的")

        # ---- 1. 找到正确的表头 ----
        header_index = _find_header_row(all_rows)
        if header_index is None:
            return _fail(
                file_path,
                f"找不到表头行：前 {MAX_HEADER_SCAN_ROWS} 行里没有识别到像表头的一行",
            )

        headers = [_clean_cell(c) for c in all_rows[header_index]]

        # ---- 2. 读取每一行（原样搬，不转换）----
        rows, problems = [], []

        # 从表头下一行开始。offset 是 Excel 里的真实行号，方便写进 problems。
        for offset, raw_row in enumerate(
            all_rows[header_index + 1:], start=header_index + 2
        ):
            # 整行都是空的 -> 空行或表尾，直接跳过。
            # 注意这里判断的是「整行都空」，不是「没有姓名」——
            # 哪一列是姓名、缺了姓名算不算问题，都是 extraction.py 的判断。
            if all(_is_blank(c) for c in raw_row):
                continue

            record = {}
            for i, name in enumerate(headers):
                # 这一行比表头短时，缺的格子补 None
                record[name] = raw_row[i] if i < len(raw_row) else None

            if len(raw_row) < len(headers):
                problems.append(f"第 {offset} 行比表头少 {len(headers) - len(raw_row)} 列")

            rows.append(record)

        if not rows:
            return _fail(file_path, "表头找到了，但一行数据都没有")

        return {
            "ok": True,
            "error": None,
            "file": str(path),
            "sheet": sheet.title,
            "columns": headers,
            "rows": rows,
            "problems": problems,
        }

    finally:
        # Windows 上不关文件句柄，文件会一直被占用，
        # 你在 Excel 里改完会存不回去。
        wb.close()


# ============================================================
# 内部工具函数
# ============================================================


def _fail(file_path: str, message: str) -> dict:
    """所有失败走同一个出口，保证返回结构一致。"""
    return {
        "ok": False,
        "error": message,
        "file": str(file_path),
        "sheet": None,
        "columns": [],
        "rows": [],
        "problems": [],
    }


def _is_blank(value) -> bool:
    """空值判断：None 或者去空格后为空字符串。"""
    return value is None or str(value).strip() == ""


def _clean_cell(value) -> str:
    """把表头单元格洗成干净字符串。

    全角空格 \\u3000 和 BOM \\ufeff 肉眼完全看不出来，
    但会让 "工号" == "工号 " 判成 False。

    注意：这里只洗「表头」，不洗数据 —— 数据原样搬给 extraction.py。
    """
    if value is None:
        return ""
    text = str(value)
    text = text.replace("﻿", "")   # BOM
    text = text.replace("　", "")   # 全角空格
    return text.strip()


def _find_header_row(all_rows: list) -> int | None:
    """在前若干行里找表头行：命中 HEADER_HINTS 最多的那一行。

    用打分而不是硬取第 1 行，因为真实工资表第 1 行往往是标题。

    找不到返回 None，让上层明确报「表头对不上」，
    而不是硬着头皮把标题行当表头用。
    """
    best_index, best_hits = None, 0

    for i, row in enumerate(all_rows[:MAX_HEADER_SCAN_ROWS]):
        hits = sum(1 for cell in row if _clean_cell(cell) in HEADER_HINTS)
        if hits > best_hits:
            best_index, best_hits = i, hits

    # 命中太少说明这张表根本不是工资表，宁可报错也不要猜
    return best_index if best_hits >= MIN_HEADER_HITS else None


if __name__ == "__main__":
    # 单独验证用：python parser.py data/payroll_demo.xlsx
    import json
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/payroll_demo.xlsx"
    print(json.dumps(read_payroll_excel(target), ensure_ascii=False, indent=2))
