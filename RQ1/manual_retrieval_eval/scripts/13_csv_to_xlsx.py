"""
Convert Files 1 + 2 from CSV to multi-sheet XLSX with annotator-friendly
formatting:
  - rows sorted by query, then unit_id (units for same query are grouped)
  - text-wrap on question / unit_text / primary_reason
  - alternating row background per query (visually separates queries)
  - conditional formatting on primary_label (green=True, red=False)
  - dropdown validation for annotator2_label (True / False / "")
  - frozen header + question column

File 1 -> 01_usefulness_pipelines.xlsx (one tab per pipeline)
File 2 -> 02_usefulness_thresholds.xlsx (one tab per threshold)
"""
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, PatternFill, Font, Border, Side
from openpyxl.formatting.rule import CellIsRule
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.abspath(os.path.join(HERE, "..", "manual_evaluation"))

QUERY_BG_A = PatternFill("solid", fgColor="FFFFFFFF")  # white
QUERY_BG_B = PatternFill("solid", fgColor="FFE9F2FA")  # very light blue
HEADER_FILL = PatternFill("solid", fgColor="FFB7CEE2")  # blue-grey
PRIMARY_HIGHLIGHT = PatternFill("solid", fgColor="FFFFF2C5")  # light yellow
ANN2_HIGHLIGHT = PatternFill("solid", fgColor="FFFFD7D2")    # light pink
TRUE_FILL = PatternFill("solid", fgColor="FFC6EFCE")  # green
FALSE_FILL = PatternFill("solid", fgColor="FFFFC7CE")  # red
THIN = Side(style="thin", color="FFCCCCCC")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def write_sheet(wb, sheet_name, df, group_key="test_idx",
                wrap_cols=("question", "unit_text", "primary_reason"),
                widths=None):
    ws = wb.create_sheet(title=sheet_name[:31])
    df = df.reset_index(drop=True)

    # Header
    for j, col in enumerate(df.columns, 1):
        cell = ws.cell(row=1, column=j, value=col)
        cell.font = Font(bold=True, color="FF222222")
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER

    # Data rows with alternating per-query fill
    last_q = None
    use_a = True
    for i, row in df.iterrows():
        excel_row = i + 2
        cur_q = row[group_key] if group_key in df.columns else None
        if cur_q != last_q:
            use_a = not use_a
            last_q = cur_q
        bg = QUERY_BG_A if use_a else QUERY_BG_B
        for j, col in enumerate(df.columns, 1):
            v = row[col]
            cell = ws.cell(row=excel_row, column=j, value=("" if pd.isna(v) else v))
            cell.fill = bg
            cell.border = BORDER
            wrap = col in wrap_cols
            cell.alignment = Alignment(
                wrap_text=wrap,
                vertical="top",
                horizontal="left" if wrap else "center")

    # Highlight primary_label and annotator2_label columns
    cols = list(df.columns)
    if "primary_label" in cols:
        col = cols.index("primary_label") + 1
        for r in range(2, len(df) + 2):
            ws.cell(row=r, column=col).font = Font(bold=True)
        # conditional formatting True -> green, False -> red
        rng = f"{get_column_letter(col)}2:{get_column_letter(col)}{len(df)+1}"
        ws.conditional_formatting.add(
            rng, CellIsRule(operator="equal", formula=['"True"'], fill=TRUE_FILL))
        ws.conditional_formatting.add(
            rng, CellIsRule(operator="equal", formula=['"False"'], fill=FALSE_FILL))
    if "annotator2_label" in cols:
        col = cols.index("annotator2_label") + 1
        rng = f"{get_column_letter(col)}2:{get_column_letter(col)}{len(df)+1}"
        # Data validation: True / False / blank
        dv = DataValidation(type="list", formula1='"True,False"', allow_blank=True)
        dv.error = "Use True or False"
        ws.add_data_validation(dv)
        dv.add(rng)
        # conditional formatting
        ws.conditional_formatting.add(
            rng, CellIsRule(operator="equal", formula=['"True"'], fill=TRUE_FILL))
        ws.conditional_formatting.add(
            rng, CellIsRule(operator="equal", formula=['"False"'], fill=FALSE_FILL))
        for r in range(2, len(df) + 2):
            ws.cell(row=r, column=col).fill = ANN2_HIGHLIGHT
            ws.cell(row=r, column=col).border = BORDER

    # Column widths
    if widths is None:
        widths = {}
    default_widths = {
        "question": 50, "method": 12, "threshold": 9, "unit_id": 6,
        "unit_text": 70, "primary_label": 11, "primary_reason": 45,
        "annotator2_label": 14, "annotator2_notes": 30,
        "test_idx": 8, "from_cache": 8, "model": 18,
        "generated_answer": 70, "Catalog": 8, "query_idx": 8,
    }
    default_widths.update(widths)
    for j, col in enumerate(df.columns, 1):
        ws.column_dimensions[get_column_letter(j)].width = default_widths.get(col, 14)

    # Default row height (give text wrap room)
    for r in range(2, len(df) + 2):
        ws.row_dimensions[r].height = 60

    # Freeze: header row + first column (question)
    ws.freeze_panes = "B2"


def build_file1():
    src = os.path.join(EVAL, "01_usefulness_pipelines.csv")
    dst = os.path.join(EVAL, "01_usefulness_pipelines.xlsx")
    df = pd.read_csv(src)
    df = df.sort_values(["test_idx", "method", "unit_id"]).reset_index(drop=True)
    methods = sorted(df["method"].unique())
    wb = Workbook()
    wb.remove(wb.active)
    for m in methods:
        sub = df[df["method"] == m].drop(columns=["method"]).reset_index(drop=True)
        write_sheet(wb, sheet_name=m, df=sub, group_key="test_idx")
    wb.save(dst)
    print(f"[File 1] wrote {dst}  sheets={methods}")


def build_file2():
    src = os.path.join(EVAL, "02_usefulness_thresholds.csv")
    dst = os.path.join(EVAL, "02_usefulness_thresholds.xlsx")
    df = pd.read_csv(src)
    df = df.sort_values(["threshold", "test_idx", "unit_id"]).reset_index(drop=True)
    thresholds = sorted(df["threshold"].unique())
    wb = Workbook()
    wb.remove(wb.active)
    for t in thresholds:
        sub = df[df["threshold"] == t].drop(columns=["threshold"]).reset_index(drop=True)
        write_sheet(wb, sheet_name=f"thr_{t:.1f}", df=sub, group_key="test_idx")
    wb.save(dst)
    print(f"[File 2] wrote {dst}  sheets={thresholds}")


def build_file3():
    src = os.path.join(EVAL, "03_final_answer_usefulness.csv")
    dst = os.path.join(EVAL, "03_final_answer_usefulness.xlsx")
    if not os.path.exists(src):
        print(f"[File 3] source not found: {src}"); return
    df = pd.read_csv(src)
    df = df.sort_values(["model", "query_idx"]).reset_index(drop=True)
    models = sorted(df["model"].unique())
    wb = Workbook()
    wb.remove(wb.active)
    for m in models:
        sub = df[df["model"] == m].drop(columns=["model"]).reset_index(drop=True)
        # Use query_idx as group key for File 3
        write_sheet(wb, sheet_name=m, df=sub, group_key="query_idx",
                    wrap_cols=("question", "generated_answer", "primary_reason"))
    wb.save(dst)
    print(f"[File 3] wrote {dst}  sheets={models}")


def build_file4():
    src = os.path.join(EVAL, "04_hallucination.csv")
    dst = os.path.join(EVAL, "04_hallucination.xlsx")
    if not os.path.exists(src):
        print(f"[File 4] source not found yet")
        return
    df = pd.read_csv(src)
    df = df.sort_values(["model", "query_idx"]).reset_index(drop=True)
    wb = Workbook()
    wb.remove(wb.active)
    write_sheet(wb, sheet_name="alignment_sample", df=df, group_key="query_idx",
                wrap_cols=("question", "generated_answer", "accepted_answer",
                           "primary_reason", "primary_claims_json"))
    wb.save(dst)
    print(f"[File 4] wrote {dst}  rows={len(df)}")


def main():
    build_file1()
    build_file2()
    build_file3()
    build_file4()


if __name__ == "__main__":
    main()
