"""
Google Sheets 讀寫（推播績效追蹤用）

需要 GitHub Actions secrets：
  GOOGLE_SERVICE_ACCOUNT_JSON  服務帳戶金鑰 JSON 全文
  PUSH_SHEET_ID                試算表 ID（網址 /d/ 與 /edit 之間那串）
試算表要「共用」給金鑰裡的 client_email（編輯者）。

沒設定時 configured() 回傳 False，呼叫端改用 repo 內 CSV，不會出錯。
欄位以 [(程式欄名, 試算表表頭), ...] 描述；試算表顯示中文表頭，讀回來轉成程式欄名。
空白值寫成「—」，讀回來再轉回空白。
"""
import os
import json

EMPTY = "—"


def configured() -> bool:
    return bool(os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
                and os.environ.get("PUSH_SHEET_ID", "").strip())


_SH = None


def _book():
    global _SH
    if _SH is None:
        import gspread
        info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
        gc = gspread.service_account_from_dict(info)
        _SH = gc.open_by_key(os.environ["PUSH_SHEET_ID"].strip())
    return _SH


def _color(hex_):
    h = hex_.lstrip("#")
    return {"red": int(h[0:2], 16) / 255, "green": int(h[2:4], 16) / 255, "blue": int(h[4:6], 16) / 255}


def worksheet(title, cols, pct_keys=(), up="#C0392B", down="#1E7E5A"):
    """取得分頁；不存在就建立，加上粗體凍結表頭、大字，並把報酬欄設成漲跌顏色（台股紅漲綠跌、美股綠漲紅跌）。"""
    import gspread
    sh = _book()
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        pass
    ws = sh.add_worksheet(title=title, rows=1000, cols=max(len(cols), 8))
    ws.update(values=[[label for _, label in cols]], range_name="A1")
    sid = ws.id
    reqs = [
        {"updateSheetProperties": {"properties": {"sheetId": sid, "gridProperties": {"frozenRowCount": 1}},
                                   "fields": "gridProperties.frozenRowCount"}},
        {"repeatCell": {"range": {"sheetId": sid},
                        "cell": {"userEnteredFormat": {"textFormat": {"fontSize": 12}}},
                        "fields": "userEnteredFormat.textFormat.fontSize"}},
        {"repeatCell": {"range": {"sheetId": sid, "startRowIndex": 0, "endRowIndex": 1},
                        "cell": {"userEnteredFormat": {"textFormat": {"bold": True, "fontSize": 12},
                                                       "backgroundColor": _color("#E2E8F0")}},
                        "fields": "userEnteredFormat(textFormat,backgroundColor)"}},
    ]
    keys = [k for k, _ in cols]
    for k in pct_keys:
        if k not in keys:
            continue
        i = keys.index(k)
        rng = {"sheetId": sid, "startRowIndex": 1, "startColumnIndex": i, "endColumnIndex": i + 1}
        for op, col in (("NUMBER_GREATER", up), ("NUMBER_LESS", down)):
            reqs.append({"addConditionalFormatRule": {"index": 0, "rule": {
                "ranges": [rng],
                "booleanRule": {"condition": {"type": op, "values": [{"userEnteredValue": "0"}]},
                                "format": {"textFormat": {"bold": True, "foregroundColor": _color(col)}}}}}})
    sh.batch_update({"requests": reqs})
    return ws


def _cell(v, num=False):
    try:
        import numpy as np
        if isinstance(v, np.generic):
            v = v.item()
    except ImportError:
        pass
    if num and isinstance(v, str) and v.strip() not in ("", "nan", "None"):
        try:
            v = float(v)
        except ValueError:
            pass
    if v is None:
        return EMPTY
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float) and v != v:
        return EMPTY
    if isinstance(v, str) and v.strip() in ("", "nan", "None"):
        return EMPTY
    return v


def _values(cols, rows, num_keys):
    return [[_cell(r.get(k), k in num_keys) for k, _ in cols] for r in rows]


# RAW：字串原樣保存（股票代號 0050、日期不會被試算表改格式），數字以數字保存（顏色規則才會生效）
def append_rows(title, cols, rows, num_keys=(), **kw):
    if not rows:
        return
    ws = worksheet(title, cols, **kw)
    ws.append_rows(_values(cols, rows, num_keys), value_input_option="RAW")


def replace_rows(title, cols, rows, num_keys=(), **kw):
    """整個分頁改寫成 rows（保留表頭與格式）。"""
    ws = worksheet(title, cols, **kw)
    values = [[label for _, label in cols]] + _values(cols, rows, num_keys)
    ws.batch_clear([f"A2:{_col_letter(len(cols))}{max(ws.row_count, 2)}"])
    if ws.row_count < len(values) + 10:
        ws.add_rows(len(values) + 100 - ws.row_count)
    ws.update(values=values, range_name="A1", value_input_option="RAW")


def read_rows(title, cols):
    """讀回 list[dict]（程式欄名）；分頁不存在回傳 []。表頭對不上的欄位留空。"""
    import gspread
    try:
        ws = _book().worksheet(title)
    except gspread.WorksheetNotFound:
        return []
    vals = ws.get_all_values()
    if len(vals) < 2:
        return []
    head = vals[0]
    lab2key = {label: k for k, label in cols}
    out = []
    for row in vals[1:]:
        if not any(str(x).strip() for x in row):
            continue
        d = {k: "" for k, _ in cols}
        for i, h in enumerate(head):
            if h in lab2key and i < len(row):
                v = row[i]
                d[lab2key[h]] = "" if v == EMPTY else {"是": "True", "否": "False"}.get(v, v)
        out.append(d)
    return out


def _col_letter(n):
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s
