import os
import json
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ระบบชี้เป้าโรคเฝ้าระวังสัปดาห์หน้า | สคร.2 จ.พิษณุโลก",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# ฟังก์ชันพื้นฐานและตัวช่วยจัดรูปแบบ
# =========================================================
def render_html(html_str):
    cleaned = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def to_int(val, default=0):
    try:
        if pd.isna(val) or str(val).strip() == "":
            return default
        return int(round(float(val)))
    except Exception:
        return default


def fmt_int(val):
    try:
        if pd.isna(val) or str(val).strip() == "":
            return "-"
        return f"{int(round(float(val))):,}"
    except Exception:
        return "-"


def normalize_dcode(val):
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return digits.zfill(2) if len(digits) <= 2 else digits
    return s


def calc_threshold(series):
    valid = series.dropna()
    if len(valid) == 0:
        return 1
    th = int(round(float(np.percentile(valid, 75))))
    if th == 0:
        th = max(1, int(round(float(np.percentile(valid, 90))))) if valid.max() > 0 else 1
    return th


def risk_sort_key(d):
    fc_i, th_i = to_int(d["forecast"]), to_int(d["threshold"])
    ratio = (fc_i / th_i) if th_i > 0 else float(fc_i)
    return (1 if d["is_flagged"] else 0, ratio, fc_i - th_i, fc_i)


def ratio_text(num, den):
    return f"{num}/{den} ({num/den:.0%})" if den > 0 else f"{num}/0 (ไม่มีตัวหาร)"


def clean_str(val):
    return "" if pd.isna(val) or str(val).strip().lower() == "nan" else str(val).strip()


DEFAULT_DISEASE_NAMES = {
    "02": "โรคอุจจาระร่วงเฉียบพลัน",
    "15": "โรคไข้หวัดใหญ่",
    "31": "โรคปอดอักเสบ"
}

ODPC2_PROVINCES = {
    "65": "พิษณุโลก",
    "53": "อุตรดิตถ์",
    "63": "ตาก",
    "64": "สุโขทัย",
    "67": "เพชรบูรณ์",
}

PROVINCE_THEME = {
    "65": {"color": "#0369a1", "bg": "#e0f2fe", "border": "#bae6fd"},
    "53": {"color": "#6d28d9", "bg": "#ede9fe", "border": "#ddd6fe"},
    "63": {"color": "#c2410c", "bg": "#ffedd5", "border": "#fed7aa"},
    "64": {"color": "#047857", "bg": "#d1fae5", "border": "#a7f3d0"},
    "67": {"color": "#a16207", "bg": "#fef9c3", "border": "#fde047"},
}


def get_prov_theme(prov_code):
    return PROVINCE_THEME.get(str(prov_code).strip(), {"color": "#475569", "bg": "#f1f5f9", "border": "#e2e8f0"})


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_data_path(target_name):
    for folder in [BASE_DIR, os.getcwd()]:
        if os.path.exists(folder):
            for fname in os.listdir(folder):
                if fname.lower() == target_name.lower():
                    return os.path.join(folder, fname)
    return os.path.join(BASE_DIR, target_name)


WEEKLY_FILE = resolve_data_path("weekly_summary (1).csv")
JSON_FILE = resolve_data_path("dashboard_data (1).json")
CSV_REGISTRY = os.path.join(BASE_DIR, "tracking_registry (1).csv")

# =========================================================
# สไตล์ Enterprise Web Application (Clean White & Green + Google Sans)
# =========================================================
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&display=swap');

:root {
    --brand: #006837;
    --brand-dark: #004d28;
    --brand-light: #ecfdf5;
    --brand-border: #a7f3d0;
    --bg-canvas: #f8fafc;
    --surface: #ffffff;
    --border: #e2e8f0;
    --text-main: #0f172a;
    --text-sub: #475569;
    --text-muted: #64748b;
}

html, body, [class*="css"] {
    font-family: 'Google Sans', 'Noto Sans Thai', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 14px !important;
    color: var(--text-main);
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background-color: var(--bg-canvas);
}

/* ซ่อนแถบเครื่องมือ Streamlit แต่คงปุ่มเปิด-ปิดแถบข้าง */
[data-testid="stDecoration"] { display: none !important; }
.stDeployButton, [data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {
    visibility: visible !important;
    display: flex !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 1.8rem !important;
}
#MainMenu, footer { visibility: hidden !important; }

.block-container {
    padding-top: 0.7rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1320px !important;
}

/* ================= SIDEBAR (NAVIGATION RAIL) ================= */
[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.1rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

.sb-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 16px;
    background: linear-gradient(135deg, #00522b 0%, #00733d 100%);
    border-radius: 12px;
    margin-bottom: 18px;
    box-shadow: 0 4px 12px rgba(0, 104, 55, 0.14);
}
.sb-brand-icon {
    width: 36px;
    height: 36px;
    border-radius: 9px;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.28);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    flex-shrink: 0;
}
.sb-brand-title { font-size: 15px; font-weight: 700; color: #ffffff; line-height: 1.25; }
.sb-brand-sub { font-size: 11.5px; color: #d1fae5; margin-top: 2px; }

.sb-sec-tag {
    font-size: 11.5px;
    font-weight: 700;
    color: var(--brand-dark);
    background: var(--brand-light);
    border: 1px solid var(--brand-border);
    border-radius: 8px;
    padding: 6px 11px;
    margin: 16px 0 8px 0;
}

[data-testid="stSidebar"] .stCheckbox {
    padding: 3px 6px;
    border-radius: 8px;
    margin-bottom: -2px;
}
[data-testid="stSidebar"] .stCheckbox:hover {
    background-color: #f1f5f9;
}

.sb-guide {
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 15px;
    margin-top: 18px;
}
.sb-guide-hdr {
    font-size: 12.5px;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 10px;
    padding-bottom: 7px;
    border-bottom: 1px solid var(--border);
}
.sb-guide-step {
    display: flex;
    gap: 9px;
    align-items: flex-start;
    margin-bottom: 9px;
    font-size: 12px;
    color: var(--text-sub);
    line-height: 1.5;
}
.sb-guide-step:last-child { margin-bottom: 0; }
.sb-step-badge {
    background: var(--brand);
    color: #ffffff;
    font-size: 10.5px;
    font-weight: 700;
    min-width: 19px;
    height: 19px;
    border-radius: 5px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 2px;
}

/* ================= TOP APPLICATION HEADER ================= */
.top-shell {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 22px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
}
.top-org-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    padding-bottom: 10px;
    margin-bottom: 10px;
    border-bottom: 1px solid #f1f5f9;
}
.org-identity {
    display: flex;
    align-items: center;
    gap: 10px;
}
.org-pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 0 3px #d1fae5;
}
.org-main { font-size: 13.5px; font-weight: 700; color: var(--brand); }
.org-unit { font-size: 12px; color: var(--text-muted); border-left: 1px solid #cbd5e1; padding-left: 10px; }

.app-main-title {
    font-size: 20px;
    font-weight: 700;
    color: var(--text-main);
    margin: 0 0 10px 0;
    line-height: 1.35;
}
.meta-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}
.chip-green { background: var(--brand); color: #ffffff; padding: 4px 12px; border-radius: 7px; font-size: 12px; font-weight: 600; }
.chip-light { background: #f8fafc; color: var(--text-sub); border: 1px solid var(--border); padding: 4px 11px; border-radius: 7px; font-size: 12px; }
.chip-amber { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; padding: 4px 11px; border-radius: 7px; font-size: 12px; font-weight: 600; }

/* ================= NATIVE METRIC CARDS & CONTAINERS ================= */
[data-testid="stMetric"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: 3.5px solid var(--brand) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 12.5px !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-size: 23px !important;
    font-weight: 700 !important;
    color: var(--text-main) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 20px 24px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03) !important;
    margin-bottom: 14px !important;
}

.card-hdr {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 6px;
}
.card-h3 { font-size: 15.5px; font-weight: 700; color: var(--text-main); margin: 0; }
.card-desc { font-size: 12.5px; color: var(--text-muted); margin: 3px 0 0 0; }

.prov-tag {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
}

/* ================= PRIORITY ALERT PANEL ================= */
.alert-box {
    background: var(--surface);
    border: 1px solid #fecaca;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(220, 38, 38, 0.05);
    margin-bottom: 14px;
}
.alert-box-hdr {
    background: #fef2f2;
    border-bottom: 1px solid #fecaca;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14.5px;
    font-weight: 700;
    color: #991b1b;
}
.alert-item { padding: 14px 20px; border-bottom: 1px solid #f1f5f9; }
.alert-item:last-child { border-bottom: none; }
.alert-item:first-of-type { background: #fffcfc; }
.alert-item-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 6px; }
.alert-item-title { font-size: 14.5px; font-weight: 700; color: var(--text-main); }
.r-tag-1 { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; font-size: 11.5px; font-weight: 700; padding: 2px 9px; border-radius: 6px; }
.r-tag-2 { background: #ffedd5; color: #c2410c; border: 1px solid #fdba74; font-size: 11.5px; font-weight: 700; padding: 2px 9px; border-radius: 6px; }
.r-tag-3 { background: #fef9c3; color: #a16207; border: 1px solid #fde047; font-size: 11.5px; font-weight: 700; padding: 2px 9px; border-radius: 6px; }
.alert-ul { margin: 0; padding-left: 18px; color: var(--text-sub); font-size: 13px; line-height: 1.65; }

.safe-box {
    background: var(--surface);
    border: 1px solid #a7f3d0;
    border-left: 5px solid var(--brand);
    border-radius: 14px;
    padding: 18px 22px;
    color: #065f46;
    font-size: 13.5px;
    line-height: 1.65;
    margin-bottom: 14px;
}

.sec-hdr {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 18px 0 10px 0;
}
.sec-bar { width: 4px; height: 18px; background: var(--brand); border-radius: 3px; }
.sec-txt { font-size: 15.5px; font-weight: 700; color: var(--text-main); }

.legend-inline {
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 8px;
    font-size: 12.5px;
    color: var(--text-sub);
    line-height: 1.7;
}

.rec-bar {
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 10px 14px;
    margin: 8px 0 14px 0;
    font-size: 13px;
    color: var(--text-sub);
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    align-items: center;
}

/* ปุ่มกดและ Popover */
.stButton > button[kind="primary"], .stFormSubmitButton > button {
    background-color: var(--brand) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {
    background-color: var(--brand-dark) !important;
}
.stButton > button[kind="secondary"], [data-testid="stPopover"] > button {
    background-color: var(--surface) !important;
    color: var(--brand) !important;
    border: 1px solid var(--brand-border) !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}
.stButton > button[kind="secondary"]:hover, [data-testid="stPopover"] > button:hover {
    background-color: var(--brand-light) !important;
    border-color: var(--brand) !important;
}
</style>
""")


# =========================================================
# หน้าต่างป๊อปอัปแสดงระเบียบวิธีและเกณฑ์ทางสถิติ (@st.dialog)
# =========================================================
@st.dialog("ระเบียบวิธีและเกณฑ์ทางระบาดวิทยา (Methodology)")
def show_methodology_modal():
    st.markdown("""
    **1. แหล่งข้อมูลและขอบเขตการเฝ้าระวัง**
    * ใช้ข้อมูลรายงานผู้ป่วยรายสัปดาห์จากระบบเฝ้าระวังโรคดิจิทัล (**DDS**) ครอบคลุมพื้นที่ **5 จังหวัดเขตสุขภาพที่ 2** (พิษณุโลก, อุตรดิตถ์, ตาก, สุโขทัย, เพชรบูรณ์) โดยผ่านกระบวนการตัดข้อมูลส่วนบุคคล (De-identification) ตามมาตรฐาน PDPA เรียบร้อยแล้ว

    **2. เกณฑ์เตือนปกติ (Alert Threshold — P75)**
    * คำนวณแยกตามรายจังหวัดและรายโรค จากค่า **เปอร์เซ็นไทล์ที่ 75 (P75)** ของจำนวนผู้ป่วยรายสัปดาห์ย้อนหลังสูงสุด 104 สัปดาห์ (2 ปี)
    * หากยอดคาดการณ์สัปดาห์หน้าสูงเท่ากับหรือมากกว่าค่า P75 ระบบจะติดป้าย **"เฝ้าระวัง"** เพื่อให้ทีมระบาดวิทยาตรวจสอบเป็นลำดับแรก

    **3. แบบจำลองพยากรณ์ล่วงหน้า 1 สัปดาห์**
    * ใช้แบบจำลองอนุกรมเวลา **Simple Exponential Smoothing** พร้อมคำนวณช่วงความเชื่อมั่น **95% Prediction Interval** เพื่อประมาณการแนวโน้มสัปดาห์ถัดไป

    **4. ข้อควรระวังเรื่องความครบถ้วนของข้อมูล**
    * ข้อมูลช่วง **1–2 สัปดาห์ล่าสุด** เป็นข้อมูลเบื้องต้นที่หน่วยบริการกำลังทยอยบันทึกเข้าระบบ (Reporting Lag) ควรตรวจสอบร่วมกับผลสอบสวนโรคภาคสนามก่อนสรุปผล
    """)


# =========================================================
# 1. ระบบจัดเก็บทะเบียนติดตามผล (PostgreSQL / CSV)
# =========================================================
@st.cache_resource
def get_db_engine():
    try:
        db_url = st.secrets.get("DATABASE_URL", "")
    except Exception:
        db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return None
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return create_engine(db_url, pool_pre_ping=True)


def ensure_table(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tracking_registry (
                run_id TEXT, province_code TEXT, province_name TEXT,
                disease_code TEXT, disease_name TEXT, target_week TEXT,
                frozen_forecast DOUBLE PRECISION, frozen_threshold DOUBLE PRECISION,
                is_flagged BOOLEAN, alert_sent_time TEXT,
                check_start_time TEXT, field_result TEXT, action_taken TEXT,
                settled_actual_cases DOUBLE PRECISION,
                PRIMARY KEY (run_id, province_code, disease_code, target_week)
            );
        """))


def sync_to_registry(payload):
    rows = []
    cutoff = payload["meta"]["cutoff_date"]
    for p in payload["provinces"]:
        for d in p["all_diseases"]:
            rows.append({
                "run_id": d["run_id"], "province_code": str(d["province_code"]),
                "province_name": str(d["province_name"]), "disease_code": str(d["disease_code"]),
                "disease_name": str(d["disease_name"]), "target_week": str(d["target_week"]),
                "frozen_forecast": to_int(d["forecast"]), "frozen_threshold": to_int(d["threshold"]),
                "is_flagged": bool(d["is_flagged"]), "alert_sent_time": f"{cutoff} 09:00",
                "check_start_time": "", "field_result": "", "action_taken": "", "settled_actual_cases": None
            })

    engine = get_db_engine()
    if engine:
        ensure_table(engine)
        with engine.begin() as conn:
            for r in rows:
                conn.execute(text("""
                    INSERT INTO tracking_registry VALUES (
                        :run_id, :province_code, :province_name, :disease_code, :disease_name,
                        :target_week, :frozen_forecast, :frozen_threshold, :is_flagged,
                        :alert_sent_time, :check_start_time, :field_result, :action_taken, :settled_actual_cases
                    ) ON CONFLICT (run_id, province_code, disease_code, target_week) DO NOTHING;
                """), r)
    else:
        new_df = pd.DataFrame(rows)
        if os.path.exists(CSV_REGISTRY):
            old_df = pd.read_csv(CSV_REGISTRY, dtype={"province_code": str, "disease_code": str})
            combined = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(
                subset=["run_id", "province_code", "disease_code", "target_week"], keep="first"
            )
        else:
            combined = new_df
        combined.to_csv(CSV_REGISTRY, index=False, encoding="utf-8-sig")


def load_registry(prov_code=None, disease_code_filter=None):
    engine = get_db_engine()
    if engine:
        ensure_table(engine)
        df_r = pd.read_sql("SELECT * FROM tracking_registry ORDER BY target_week DESC, province_code ASC, disease_code ASC", engine)
    elif os.path.exists(CSV_REGISTRY):
        df_r = pd.read_csv(CSV_REGISTRY, dtype={"province_code": str, "disease_code": str})
    else:
        return pd.DataFrame()
    if df_r.empty:
        return df_r
    df_r["province_code"] = df_r["province_code"].astype(str).str.strip()
    df_r["disease_code"] = df_r["disease_code"].apply(normalize_dcode)
    if prov_code is not None:
        p_list = [str(x) for x in prov_code] if isinstance(prov_code, list) else [str(prov_code)]
        df_r = df_r[df_r["province_code"].isin(p_list)].copy()
    if disease_code_filter is not None:
        d_list = [normalize_dcode(x) for x in disease_code_filter] if isinstance(disease_code_filter, list) else [normalize_dcode(disease_code_filter)]
        df_r = df_r[df_r["disease_code"].isin(d_list)].copy()
    return df_r


def update_registry(run_id, prov_code, disease_code, target_week, check_time, field_res, action_txt, actual_val):
    engine = get_db_engine()
    if engine:
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE tracking_registry
                SET check_start_time=:ct, field_result=:fr, action_taken=:at, settled_actual_cases=:ac
                WHERE run_id=:rid AND province_code=:pc AND disease_code=:dc AND target_week=:tw;
            """), {"ct": check_time, "fr": field_res, "at": action_txt, "ac": actual_val,
                   "rid": run_id, "pc": str(prov_code), "dc": normalize_dcode(disease_code), "tw": str(target_week)})
    else:
        df_all = pd.read_csv(CSV_REGISTRY, dtype={"province_code": str, "disease_code": str})
        m = (
            (df_all["run_id"] == run_id) &
            (df_all["province_code"] == str(prov_code)) &
            (df_all["disease_code"].apply(normalize_dcode) == normalize_dcode(disease_code)) &
            (df_all["target_week"] == str(target_week))
        )
        df_all.loc[m, ["check_start_time", "field_result", "action_taken", "settled_actual_cases"]] = [
            check_time, field_res, action_txt, actual_val
        ]
        df_all.to_csv(CSV_REGISTRY, index=False, encoding="utf-8-sig")


# =========================================================
# 2. โหลดข้อมูลและคำนวณพยากรณ์ (รองรับทุกโรคอัตโนมัติ)
# =========================================================
@st.cache_data
def load_files():
    if not os.path.exists(JSON_FILE) or not os.path.exists(WEEKLY_FILE):
        files_here = os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else []
        st.error(f"ไม่พบไฟล์ dashboard_data.json หรือ weekly_summary.csv (ไฟล์ที่ระบบมองเห็นในโฟลเดอร์คือ: {files_here})")
        st.stop()
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        latest_json = json.load(f)
    weekly = pd.read_csv(WEEKLY_FILE, dtype={"province_code": str, "disease_code": str})
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    weekly["disease_code"] = weekly["disease_code"].apply(normalize_dcode)
    if "disease_name" not in weekly.columns:
        weekly["disease_name"] = weekly["disease_code"].map(DEFAULT_DISEASE_NAMES).fillna(weekly["disease_code"])
    else:
        weekly["disease_name"] = weekly["disease_name"].fillna(weekly["disease_code"].map(DEFAULT_DISEASE_NAMES)).fillna(weekly["disease_code"])
    return latest_json, weekly


@st.cache_data
def predict_for_date(weekly_df, cutoff_str):
    df_all = weekly_df.sort_values(["province_code", "disease_code", "week_start"]).copy()
    cutoff_wk = pd.to_datetime(cutoff_str).to_period("W-SAT").start_time
    df_cut = df_all[df_all["week_start"] <= cutoff_wk].copy()
    if df_cut.empty:
        df_cut = df_all[df_all["week_start"] == df_all["week_start"].min()].copy()

    last_wk = df_cut["week_start"].max()
    target_week = (last_wk + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d")
    run_id = f"RUN-{cutoff_str}"

    prelim_start = last_wk - pd.Timedelta(weeks=1)
    df_cut.loc[df_cut["data_status"] != "ข้อมูลขาด", "data_status"] = "ครบถ้วน"
    df_cut.loc[(df_cut["week_start"] >= prelim_start) & (df_cut["data_status"] != "ข้อมูลขาด"), "data_status"] = "เบื้องต้น (รอยอดนิ่ง)"

    provinces = []
    for (p_code, p_name), p_grp in df_cut.groupby(["province_code", "province_name"]):
        diseases = []
        for d_code, grp in p_grp.groupby("disease_code"):
            d_name_val = clean_str(grp["disease_name"].iloc[-1]) if "disease_name" in grp.columns else ""
            if not d_name_val or d_name_val == d_code:
                d_name_val = DEFAULT_DISEASE_NAMES.get(d_code, f"กลุ่มโรค {d_code}")

            s = grp.set_index("week_start")["cases"].astype(float).interpolate(method="linear").ffill().bfill().fillna(0)
            th = calc_threshold(s.iloc[:min(104, len(s))])
            try:
                m = ExponentialSmoothing(s, trend=None, seasonal=None).fit()
                fc_raw = max(0.0, float(m.forecast(1).iloc[0]))
                std = float(np.std(m.resid))
            except Exception:
                fc_raw, std = float(s.iloc[-1]), float(np.std(s))

            fc = int(round(fc_raw))
            pi_low = min(fc, int(round(max(0.0, fc_raw - 1.96 * std))))
            pi_high = max(fc, int(round(fc_raw + 1.96 * std)))
            flagged = fc >= th

            diseases.append({
                "run_id": run_id, "province_code": p_code, "province_name": p_name,
                "disease_code": d_code, "disease_name": d_name_val,
                "target_week": target_week, "latest_cases": int(round(float(s.iloc[-1]))),
                "forecast": fc, "pi_low": pi_low, "pi_high": pi_high,
                "threshold": th, "is_flagged": flagged,
                "excess_ratio": round(fc / th, 2) if th > 0 else 0.0,
                "history": [
                    {"week": r["week_start"].strftime("%Y-%m-%d"), "cases": None if pd.isna(r["cases"]) else int(round(float(r["cases"]))), "status": r["data_status"]}
                    for _, r in grp.tail(52).iterrows()
                ]
            })

        top_alerts = sorted([d for d in diseases if d["is_flagged"]], key=risk_sort_key, reverse=True)[:3]
        provinces.append({
            "province_code": p_code, "province_name": p_name,
            "alert_count": len(top_alerts), "alerts": top_alerts, "all_diseases": diseases
        })

    return {
        "meta": {"run_id": run_id, "cutoff_date": cutoff_str, "target_week": target_week, "data_status": df_cut["data_status"].iloc[-1]},
        "provinces": provinces
    }


# =========================================================
# 3. โหลดข้อมูลหลัก + แถบควบคุมด้านซ้าย
# =========================================================
latest_json, weekly_df = load_files()

min_date = pd.to_datetime("2023-11-23").date()
max_data_dt = weekly_df["week_start"].max().date()
max_date = max(pd.Timestamp.today().date(), max_data_dt)
latest_cutoff = pd.to_datetime(latest_json["meta"]["cutoff_date"]).date()
default_date = min(max(latest_cutoff, min_date), max_date)

with st.sidebar:
    render_html("""
    <div class="sb-brand">
        <div class="sb-brand-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="M9 12l2 2 4-4"/>
            </svg>
        </div>
        <div>
            <div class="sb-brand-title">สคร.2 จ.พิษณุโลก</div>
            <div class="sb-brand-sub">ระบบเฝ้าระวังและชี้เป้าโรค (DDS)</div>
        </div>
    </div>
    """)

    render_html('<div class="sb-sec-tag">1. ใช้ข้อมูลถึงวันที่ (ทำนายสัปดาห์หน้า)</div>')
    picked_date = st.date_input(
        "วันตัดข้อมูล",
        value=default_date, min_value=min_date, max_value=max_date,
        label_visibility="collapsed"
    )
    picked_str = picked_date.strftime("%Y-%m-%d")

    payload = latest_json if picked_str == latest_json["meta"]["cutoff_date"] else predict_for_date(weekly_df, picked_str)
    sync_to_registry(payload)

    render_html('<div class="sb-sec-tag">2. เลือกจังหวัด (สคร.2)</div>')
    all_p_codes = [p["province_code"] for p in payload["provinces"]]

    if "chk_p_all" not in st.session_state:
        st.session_state["chk_p_all"] = True
    for pc in all_p_codes:
        if f"chk_p_{pc}" not in st.session_state:
            st.session_state[f"chk_p_{pc}"] = True

    def _toggle_all_p():
        val = st.session_state["chk_p_all"]
        for pc in all_p_codes:
            st.session_state[f"chk_p_{pc}"] = val

    def _sync_all_p():
        st.session_state["chk_p_all"] = all(st.session_state[f"chk_p_{pc}"] for pc in all_p_codes)

    st.checkbox("**เลือกทุกจังหวัด (ทั้ง 5 จว.)**", key="chk_p_all", on_change=_toggle_all_p)

    selected_provinces_objs = []
    for p in payload["provinces"]:
        chk_lbl = f"จ.{p['province_name']} (เตือน {p['alert_count']} โรค)"
        if st.checkbox(chk_lbl, key=f"chk_p_{p['province_code']}", on_change=_sync_all_p):
            selected_provinces_objs.append(p)

    render_html('<div class="sb-sec-tag">3. เลือกกลุ่มโรคเฝ้าระวัง</div>')
    dynamic_diseases = {}
    for p in payload["provinces"]:
        for d in p["all_diseases"]:
            dynamic_diseases[d["disease_code"]] = d["disease_name"]
    all_d_codes = sorted(dynamic_diseases.keys())

    if "chk_d_all" not in st.session_state:
        st.session_state["chk_d_all"] = True
    for dc in all_d_codes:
        if f"chk_d_{dc}" not in st.session_state:
            st.session_state[f"chk_d_{dc}"] = st.session_state["chk_d_all"]

    def _toggle_all_d():
        val = st.session_state["chk_d_all"]
        for dc in all_d_codes:
            st.session_state[f"chk_d_{dc}"] = val

    def _sync_all_d():
        st.session_state["chk_d_all"] = all(st.session_state.get(f"chk_d_{dc}", True) for dc in all_d_codes)

    st.checkbox(f"**เลือกทุกโรค (ทั้งหมด {len(all_d_codes)} โรค)**", key="chk_d_all", on_change=_toggle_all_d)

    selected_disease_codes = []
    disease_box = st.container(height=210, border=False) if len(all_d_codes) > 6 else st.container()
    with disease_box:
        for d_code in all_d_codes:
            d_name = dynamic_diseases.get(d_code, d_code)
            if st.checkbox(f"{d_code} - {d_name}", key=f"chk_d_{d_code}", on_change=_sync_all_d):
                selected_disease_codes.append(d_code)

    render_html("""
    <div class="sb-guide">
        <div class="sb-guide-hdr">ขั้นตอนการใช้งานระบบ</div>
        <div class="sb-guide-step">
            <div class="sb-step-badge">1</div>
            <div><b>กำหนดตัวกรอง:</b> เลือกวันตัดข้อมูล จังหวัด และโรคที่ต้องการติดตาม</div>
        </div>
        <div class="sb-guide-step">
            <div class="sb-step-badge">2</div>
            <div><b>ตรวจสัญญาณเตือน:</b> ดูภาพรวมและตารางจัดอันดับในหน้าแรก</div>
        </div>
        <div class="sb-guide-step">
            <div class="sb-step-badge">3</div>
            <div><b>บันทึกผลภาคสนาม:</b> กรอกผลตรวจสอบและปิดยอดจริงในหน้าติดตามผล</div>
        </div>
    </div>
    """)

meta = payload["meta"]

if not selected_provinces_objs or not selected_disease_codes:
    st.warning("กรุณาเลือกอย่างน้อย 1 จังหวัด และ 1 กลุ่มโรคจากแถบควบคุมด้านซ้ายเพื่อแสดงข้อมูล")
    st.stop()

selected_prov_code = [p["province_code"] for p in selected_provinces_objs]
selected_disease_code = selected_disease_codes
is_all_prov = len(selected_provinces_objs) > 1
selected_prov_obj = selected_provinces_objs[0]

scope_all_items = [d for p in selected_provinces_objs for d in p["all_diseases"]]
view_items = sorted(
    [d for d in scope_all_items if d["disease_code"] in selected_disease_codes],
    key=risk_sort_key,
    reverse=True
)

if len(selected_disease_codes) == len(all_d_codes):
    disease_scope_text = f"ทุกโรคในขอบเขต ({len(all_d_codes)} โรค)"
elif len(selected_disease_codes) == 1:
    disease_scope_text = dynamic_diseases.get(selected_disease_codes[0], selected_disease_codes[0])
else:
    disease_scope_text = f"เลือก {len(selected_disease_codes)} จาก {len(all_d_codes)} โรค"

flagged_items = [d for d in view_items if d["is_flagged"]]
top_alerts = flagged_items[:3]

if len(selected_provinces_objs) == len(payload["provinces"]):
    area_title = f"ภาพรวมเขตสุขภาพที่ 2 ({len(payload['provinces'])} จังหวัด)"
elif len(selected_provinces_objs) == 1:
    area_title = f"จังหวัด{selected_prov_obj['province_name']}"
else:
    area_title = f"พื้นที่ที่เลือก ({len(selected_provinces_objs)} จังหวัด)"

# =========================================================
# 4. ส่วนหัวองค์กร + แถบสลับหน้าแบบ Web App Navigation
# =========================================================
status_chip_cls = "chip-amber" if "เบื้องต้น" in str(meta["data_status"]) else "chip-light"
render_html(f"""
<div class="top-shell">
    <div class="top-org-row">
        <div class="org-identity">
            <span class="org-pulse"></span>
            <span class="org-main">สำนักงานป้องกันควบคุมโรคที่ 2 จังหวัดพิษณุโลก</span>
            <span class="org-unit">กรมควบคุมโรค • กลุ่มระบาดวิทยาและตอบโต้ภาวะฉุกเฉินทางสาธารณสุข</span>
        </div>
        <div style="font-size:12px; color:#64748b;">DDS Weekly Surveillance System</div>
    </div>
    <div class="app-main-title">ระบบชี้เป้าโรคที่ควรเฝ้าระวังสัปดาห์หน้าจากข้อมูล DDS — {area_title}</div>
    <div class="meta-chips">
        <span class="chip-green">สัปดาห์เป้าหมาย: {meta['target_week']}</span>
        <span class="chip-light">ขอบเขตโรค: <b>{disease_scope_text}</b></span>
        <span class="chip-light">ตัดข้อมูลถึง: <b>{meta['cutoff_date']}</b></span>
        <span class="chip-light">รหัสรอบ: <b>{meta['run_id']}</b></span>
        <span class="{status_chip_cls}">สถานะข้อมูล: {meta['data_status']}</span>
    </div>
</div>
""")

nav_col, btn_col = st.columns([4.0, 1.15], gap="small")
NAV_PAGES = [
    "แดชบอร์ดชี้เป้าสัปดาห์หน้า",
    "วิเคราะห์แนวโน้มรายโรค",
    "บันทึกและติดตามผล"
]
with nav_col:
    if hasattr(st, "segmented_control"):
        active_page = st.segmented_control(
            "เมนูหลัก",
            NAV_PAGES,
            default=NAV_PAGES[0],
            label_visibility="collapsed"
        )
        if active_page is None:
            active_page = NAV_PAGES[0]
    else:
        active_page = st.radio("เมนูหลัก", NAV_PAGES, horizontal=True, label_visibility="collapsed")

with btn_col:
    if st.button("ระเบียบวิธี / เกณฑ์ P75", use_container_width=True):
        show_methodology_modal()

# =========================================================
# หน้า 1: แดชบอร์ดชี้เป้าสัปดาห์หน้า
# =========================================================
if active_page == NAV_PAGES[0]:
    total_fc = sum(to_int(d["forecast"]) for d in view_items)
    total_latest = sum(to_int(d["latest_cases"]) for d in view_items)
    diff_total = total_fc - total_latest

    if top_alerts:
        t1 = top_alerts[0]
        top_risk_name = f"{t1['disease_name']}"
        prov_prefix = f"จ.{t1['province_name']} • " if is_all_prov else ""
        top_risk_delta = f"{prov_prefix}คาด {to_int(t1['forecast']):,} ราย (เกณฑ์ {to_int(t1['threshold']):,})"
    else:
        top_risk_name, top_risk_delta = "ปกติทุกโรค", "ไม่พบรายการเกินเกณฑ์ P75"

    if is_all_prov:
        flagged_provs_cnt = len({d["province_code"] for d in flagged_items})
        kpi1_val = f"พบเตือน {flagged_provs_cnt} จังหวัด" if flagged_provs_cnt > 0 else "ปกติทุกจังหวัด"
        kpi1_delta = f"เกินเกณฑ์ {len(flagged_items)}/{len(view_items)} รายการ"
    else:
        kpi1_val = f"เกินเกณฑ์ {len(flagged_items)} โรค" if len(flagged_items) > 0 else "ปกติทุกโรค"
        kpi1_delta = f"เฝ้าระวังทั้งหมด {len(view_items)} โรค"

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    k1.metric(
        label="พื้นที่และโรคที่ต้องเฝ้าระวัง",
        value=kpi1_val,
        delta=kpi1_delta,
        delta_color="inverse" if len(flagged_items) > 0 else "off"
    )
    k2.metric(
        label="ต้องระวังเป็นอันดับที่ 1",
        value=top_risk_name,
        delta=top_risk_delta,
        delta_color="inverse" if top_alerts else "off"
    )
    k3.metric(
        label="ยอดคาดการณ์รวมสัปดาห์หน้า",
        value=f"{total_fc:,} ราย",
        delta=f"{diff_total:+,} ราย จากสัปดาห์ล่าสุด",
        delta_color="inverse" if diff_total > 0 else "normal"
    )
    k4.metric(
        label="ยอดผู้ป่วยจริงรวม (สัปดาห์ล่าสุด)",
        value=f"{total_latest:,} ราย",
        delta=f"ณ วันที่ {meta['cutoff_date']}",
        delta_color="off"
    )

    col_chart, col_action = st.columns([1.28, 1.0], gap="medium")

    with col_chart:
        with st.container(border=True):
            chart_slice = view_items[:6]
            chart_sub_extra = f" (แสดง {len(chart_slice)} อันดับเสี่ยงสูงสุด)" if len(view_items) > 6 else ""
            c_hdr_l, c_hdr_r = st.columns([3.5, 1.2])
            with c_hdr_l:
                render_html(f"""
                <div class="card-hdr">
                    <div>
                        <div class="card-h3">เปรียบเทียบยอดคาดการณ์สัปดาห์หน้ากับเกณฑ์เตือน</div>
                        <div class="card-desc">แท่งสีแดง = เกินเกณฑ์เตือนปกติ (P75) • แท่งสีฟ้า = อยู่ในเกณฑ์ปกติ{chart_sub_extra}</div>
                    </div>
                </div>
                """)
            with c_hdr_r:
                with st.popover("วิธีอ่านกราฟ", use_container_width=True):
                    st.markdown("""
                    * **แท่งสีเทา:** เกณฑ์เตือนปกติ (P75) ของพื้นที่และโรคนั้น
                    * **แท่งสีแดง:** ยอดคาดการณ์สัปดาห์หน้าสูงเกินเกณฑ์เตือน ควรตรวจสอบก่อน
                    * **แท่งสีฟ้า:** ยอดคาดการณ์ยังอยู่ในเกณฑ์ปกติ
                    """)

            chart_items = list(reversed(chart_slice))
            if is_all_prov and len(selected_disease_codes) > 1:
                y_names =
