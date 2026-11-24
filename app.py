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
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# ฟังก์ชันช่วยจัดรูปแบบและสีประจำ 5 จังหวัด สคร.2
# =========================================================
def render_html(html_str):
    """ล้างช่องว่างหน้าบรรทัด กันไม่ให้ Streamlit แสดงเป็นโค้ด HTML"""
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


DISEASE_NAMES = {
    "02": "โรคอุจจาระร่วงเฉียบพลัน",
    "15": "โรคไข้หวัดใหญ่",
    "31": "โรคปอดอักเสบ"
}

PROVINCE_THEME = {
    "65": {"dot": "🔵", "color": "#0284c7", "bg": "#e0f2fe"},  # พิษณุโลก
    "53": {"dot": "🟣", "color": "#7c3aed", "bg": "#ede9fe"},  # อุตรดิตถ์
    "63": {"dot": "🟠", "color": "#ea580c", "bg": "#ffedd5"},  # ตาก
    "64": {"dot": "🟢", "color": "#059669", "bg": "#d1fae5"},  # สุโขทัย
    "67": {"dot": "🟡", "color": "#ca8a04", "bg": "#fef9c3"},  # เพชรบูรณ์
}


def get_prov_theme(prov_code):
    return PROVINCE_THEME.get(str(prov_code).strip(), {"dot": "⚪", "color": "#475569", "bg": "#f1f5f9"})


WEEKLY_FILE = "weekly_summary (1).csv"
JSON_FILE = "dashboard_data (1).json"
CSV_REGISTRY = "tracking_registry (1).csv"

# =========================================================
# สไตล์เว็บไซต์พอร์ทัลสาธารณสุข (เขียว-ขาว) + ช่องติ๊กชมพูพาสเทล
# =========================================================
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@300;400;500;600;700&family=Prompt:wght@400;500;600;700&display=swap');

/* ================= 1. Base & Website Typography ================= */
html, body, [class*="css"] {
    font-family: 'Noto Sans Thai', sans-serif !important;
    font-size: 14px !important;
    color: #1f2937;
    -webkit-font-smoothing: antialiased;
}
h1, h2, h3, .top-title, .panel-title, .sec-label, .portal-org {
    font-family: 'Prompt', 'Noto Sans Thai', sans-serif !important;
}
.stApp { background-color: #f3f6f4; }

/* ซ่อนแถบสีด้านบนสุด, ปุ่ม Deploy และ Footer เดิมของ Streamlit ให้ดูเป็นเว็บไซต์จริง */
[data-testid="stDecoration"] { display: none !important; }
.stDeployButton, [data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {
    visibility: visible !important;
    display: flex !important;
}
header[data-testid="stHeader"] { background: transparent !important; height: 2.2rem !important; }
#MainMenu, footer { visibility: hidden !important; }

.block-container {
    padding-top: 0.6rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* ================= 2. Sidebar + ช่องติ๊กสีชมพูพาสเทล ================= */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ebf5ef 0%, #f8fafc 28%, #ffffff 100%);
    border-right: 1px solid #d1d5db;
}
.sidebar-brand {
    background: linear-gradient(135deg, #005a2f 0%, #007a40 100%);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 16px;
    box-shadow: 0 4px 10px rgba(0, 104, 55, 0.14);
}
.sidebar-brand-title { font-size: 15px; font-weight: 700; color: #ffffff; margin: 0; }
.sidebar-brand-sub { font-size: 11.5px; color: #d1fae5; margin: 3px 0 0 0; }

.sb-sec-hdr {
    background: #ffffff;
    border-left: 4px solid #006837;
    border-radius: 6px;
    padding: 7px 11px;
    font-size: 13px;
    font-weight: 700;
    color: #00522b;
    margin: 14px 0 8px 0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

/* เปลี่ยนสีช่องติ๊ก Checkbox เป็นสีน้ำเงินโทนอ่อน */
[data-testid="stSidebar"] .stCheckbox {
    padding: 2px 6px;
    border-radius: 6px;
    margin-bottom: -4px;
    transition: background-color 0.15s ease;
}
[data-testid="stSidebar"] .stCheckbox:hover {
    background-color: #eff6ff;
}
[data-testid="stSidebar"] label[data-baseweb="checkbox"] > span:first-of-type,
[data-testid="stSidebar"] label[data-baseweb="checkbox"] > div:first-of-type {
    border-radius: 5px !important;
    border: 1.5px solid #93c5fd !important;
    transition: all 0.15s ease;
}
[data-testid="stSidebar"] label[data-baseweb="checkbox"] input:checked + span,
[data-testid="stSidebar"] label[data-baseweb="checkbox"] input:checked + div,
[data-testid="stSidebar"] label[data-baseweb="checkbox"]:has(input:checked) > span:first-of-type,
[data-testid="stSidebar"] label[data-baseweb="checkbox"]:has(input:checked) > div:first-of-type {
    background-color: #60a5fa !important;
    border-color: #3b82f6 !important;
}
[data-testid="stSidebar"] label[data-baseweb="checkbox"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}

/* กล่องขั้นตอนการใช้งานเว็บนี้ */
.guide-card {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-top: 3.5px solid #006837;
    border-radius: 10px;
    padding: 13px 14px;
    margin-top: 18px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}
.guide-hdr {
    font-size: 13.5px;
    font-weight: 700;
    color: #00522b;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #e5e7eb;
}
.guide-step {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    margin-bottom: 9px;
    font-size: 12px;
    color: #374151;
    line-height: 1.5;
}
.guide-step:last-child { margin-bottom: 0; }
.guide-num {
    background: #e6f4ed;
    color: #006837;
    border: 1px solid #a7f3d0;
    font-weight: 700;
    font-size: 11px;
    min-width: 20px;
    height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 1px;
}

/* ================= 3. Website Top Navigation Bar ================= */
.portal-navbar {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    padding: 10px 18px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}
.portal-org {
    font-size: 13.5px;
    font-weight: 600;
    color: #006837;
    display: flex;
    align-items: center;
    gap: 8px;
}
.portal-dot {
    width: 9px;
    height: 9px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
}
.portal-sys {
    font-size: 12px;
    color: #4b5563;
    background: #f3f4f6;
    padding: 3px 10px;
    border-radius: 6px;
    font-weight: 500;
}

/* ================= 4. Hero Banner (เขียวสาธารณสุข-ขาว ไม่มีเส้นเหลือง) ================= */
.top-header {
    background: linear-gradient(120deg, #00522b 0%, #006837 55%, #059669 100%);
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 14px;
    box-shadow: 0 4px 12px rgba(0, 104, 55, 0.14);
}
.top-title {
    font-size: 20.5px;
    font-weight: 600;
    color: #ffffff;
    margin: 0 0 11px 0;
}
.meta-line { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.meta-tag {
    background: rgba(255, 255, 255, 0.15);
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.25);
    padding: 3px 11px;
    border-radius: 5px;
    font-size: 12px;
}
.meta-tag-blue {
    background: #ffffff;
    color: #006837;
    padding: 3px 12px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 700;
}
.meta-tag-warn {
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fcd34d;
    padding: 3px 11px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 600;
}

/* ================= 5. Website Navigation Tabs (เส้นเขียวเส้นเดียว) ================= */
.stTabs [data-baseweb="tab-list"] {
    background-color: #ffffff;
    padding: 0 16px;
    border-radius: 10px;
    border: 1px solid #d1d5db;
    gap: 22px;
    margin-bottom: 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}
.stTabs [data-baseweb="tab"] {
    padding: 12px 6px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #4b5563 !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #006837 !important;
}
.stTabs [aria-selected="true"] {
    color: #006837 !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-highlight"],
div[data-baseweb="tab-highlight"] {
    background-color: #006837 !important;
    height: 3px !important;
}

/* ================= 6. Buttons & Inputs ================= */
.stButton > button[kind="primary"], .stFormSubmitButton > button {
    background-color: #006837 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 7px !important;
    padding: 0.5rem 1.3rem !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(0, 104, 55, 0.2) !important;
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {
    background-color: #00522b !important;
}
div[data-baseweb="select"] > div,
.stTextInput > div > div > input,
.stDateInput > div > div > input {
    border-radius: 7px !important;
    border: 1px solid #cbd5e1 !important;
    background-color: #ffffff !important;
}

/* ================= 7. Cards & Panels ================= */
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 4px 0 18px 0; }
.kpi-card, .kpi-card-accent {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-top: 3.5px solid #006837;
    border-radius: 10px;
    padding: 15px 18px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.kpi-card:hover, .kpi-card-accent:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.06);
}
.kpi-lbl { font-size: 12.5px; color: #4b5563; font-weight: 500; }
.kpi-val { font-size: 21px; font-weight: 700; color: #111827; margin-top: 4px; }
.kpi-note { font-size: 12px; color: #6b7280; margin-top: 4px; }

.panel-box {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
}
.panel-title { font-size: 15px; font-weight: 600; color: #006837; margin: 0 0 4px 0; }
.panel-sub { font-size: 12.5px; color: #6b7280; margin: 0 0 8px 0; }

/* ================= 8. Top-3 Alert Box ================= */
.rank-box {
    background: #ffffff;
    border: 1px solid #fca5a5;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(220, 38, 38, 0.06);
    margin-bottom: 12px;
}
.rank-box-header {
    background-color: #fef2f2;
    border-bottom: 1px solid #fecaca;
    padding: 11px 16px;
    font-size: 14px;
    font-weight: 700;
    color: #991b1b;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.rank-item { padding: 12px 16px; border-bottom: 1px solid #f3f4f6; }
.rank-item:last-child { border-bottom: none; }
.rank-item:first-of-type { background-color: #fffbfb; }
.rank-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 5px; }
.rank-title { font-size: 14.5px; font-weight: 700; color: #111827; }
.rank-b1 { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; font-size: 11.5px; font-weight: 700; padding: 2px 9px; border-radius: 5px; white-space: nowrap; }
.rank-b2 { background: #ffedd5; color: #c2410c; border: 1px solid #fdba74; font-size: 11.5px; font-weight: 700; padding: 2px 9px; border-radius: 5px; white-space: nowrap; }
.rank-b3 { background: #fef9c3; color: #a16207; border: 1px solid #fde047; font-size: 11.5px; font-weight: 700; padding: 2px 9px; border-radius: 5px; white-space: nowrap; }
.rank-bullets { margin: 4px 0 0 0; padding-left: 18px; color: #374151; font-size: 13px; line-height: 1.65; }
.rank-bullets li { margin-bottom: 2px; }

/* ================= 9. Tables, Badges & Website Footer ================= */
.prov-pill { display: inline-block; padding: 2px 9px; border-radius: 5px; font-size: 12px; font-weight: 600; white-space: nowrap; }
.safe-card { background: #ecfdf5; border: 1px solid #a7f3d0; border-left: 4px solid #006837; border-radius: 10px; padding: 14px 18px; color: #065f46; font-size: 13.5px; line-height: 1.6; }
.rec-preview { background: #f9fafb; border: 1px solid #d1d5db; border-radius: 8px; padding: 10px 14px; margin: 8px 0 12px 0; font-size: 13px; color: #374151; display: flex; flex-wrap: wrap; gap: 16px; align-items: center; }

.tbl-wrap {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    overflow-x: auto;
    margin-top: 8px;
    margin-bottom: 18px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
}
.m-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.m-table th {
    background: #e6f4ed;
    color: #00522b;
    font-weight: 700;
    font-size: 12.5px;
    padding: 11px 14px;
    text-align: left;
    border-bottom: 2px solid #006837;
    white-space: nowrap;
}
.m-table td { padding: 10px 14px; border-bottom: 1px solid #e5e7eb; color: #1f2937; vertical-align: middle; }
.m-table tbody tr:nth-child(even) { background-color: #f9fafb; }
.m-table tbody tr:hover { background-color: #f0fdf4; }
.m-table tr:last-child td { border-bottom: none; }
.num { text-align: right !important; font-variant-numeric: tabular-nums; white-space: nowrap; }
.ctr { text-align: center !important; }
.muted { font-size: 12px; color: #6b7280; }

.badge-red { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; padding: 2px 9px; border-radius: 5px; font-size: 12px; font-weight: 600; white-space: nowrap; }
.badge-ok { background: #dcfce7; color: #15803d; border: 1px solid #86efac; padding: 2px 9px; border-radius: 5px; font-size: 12px; font-weight: 600; white-space: nowrap; }
.sec-label { font-size: 15px; font-weight: 600; color: #006837; margin: 16px 0 8px 0; border-left: 4px solid #006837; padding-left: 10px; }

.chart-caption {
    background: #f8fafc;
    border: 1px solid #d1d5db;
    border-top: 2px solid #e2e8f0;
    border-radius: 0 0 10px 10px;
    padding: 11px 16px;
    margin-top: -14px;
    margin-bottom: 14px;
    font-size: 12.5px;
    color: #374151;
    line-height: 1.65;
}
.chart-caption b { color: #00522b; }

.portal-footer {
    margin-top: 28px;
    padding: 14px 20px;
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 12px;
    color: #6b7280;
}
</style>
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
    df_r["disease_code"] = df_r["disease_code"].astype(str).str.strip().str.zfill(2)
    if prov_code is not None:
        p_list = [str(x) for x in prov_code] if isinstance(prov_code, list) else [str(prov_code)]
        df_r = df_r[df_r["province_code"].isin(p_list)].copy()
    if disease_code_filter is not None:
        d_list = [str(x).zfill(2) for x in disease_code_filter] if isinstance(disease_code_filter, list) else [str(disease_code_filter).zfill(2)]
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
                   "rid": run_id, "pc": str(prov_code), "dc": str(disease_code), "tw": str(target_week)})
    else:
        df_all = pd.read_csv(CSV_REGISTRY, dtype={"province_code": str, "disease_code": str})
        m = (
            (df_all["run_id"] == run_id) &
            (df_all["province_code"] == str(prov_code)) &
            (df_all["disease_code"].str.zfill(2) == str(disease_code).zfill(2)) &
            (df_all["target_week"] == str(target_week))
        )
        df_all.loc[m, ["check_start_time", "field_result", "action_taken", "settled_actual_cases"]] = [
            check_time, field_res, action_txt, actual_val
        ]
        df_all.to_csv(CSV_REGISTRY, index=False, encoding="utf-8-sig")


# =========================================================
# 2. โหลดข้อมูลยอดรวมและคำนวณพยากรณ์ตามวันที่เลือก
# =========================================================
@st.cache_data
def load_files():
    if not os.path.exists(JSON_FILE) or not os.path.exists(WEEKLY_FILE):
        st.error("ไม่พบไฟล์ dashboard_data.json หรือ weekly_summary.csv กรุณาอัปโหลดให้ครบใน GitHub")
        st.stop()
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        latest_json = json.load(f)
    weekly = pd.read_csv(WEEKLY_FILE, dtype={"province_code": str, "disease_code": str})
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    weekly["disease_code"] = weekly["disease_code"].str.strip().str.zfill(2)
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
                "disease_code": d_code, "disease_name": DISEASE_NAMES.get(d_code, d_code),
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
# 3. แถบเมนูด้านซ้าย (Sidebar)
# =========================================================
latest_json, weekly_df = load_files()
min_date = pd.to_datetime("2023-11-23").date()
max_date = pd.Timestamp.today().date()
latest_cutoff = pd.to_datetime(latest_json["meta"]["cutoff_date"]).date()
default_date = min(max(latest_cutoff, min_date), max_date)

with st.sidebar:
    render_html("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">สคร.2 จ.พิษณุโลก</div>
        <div class="sidebar-brand-sub">ระบบเฝ้าระวังและชี้เป้าโรค (DDS)</div>
    </div>
    """)

    # 1. เลือกวันตัดข้อมูล
    render_html('<div class="sb-sec-hdr">1. ใช้ข้อมูลถึงวันที่ (ทำนายสัปดาห์หน้า)</div>')
    picked_date = st.date_input(
        "วันตัดข้อมูล",
        value=default_date, min_value=min_date, max_value=max_date,
        label_visibility="collapsed"
    )
    picked_str = picked_date.strftime("%Y-%m-%d")

    payload = latest_json if picked_str == latest_json["meta"]["cutoff_date"] else predict_for_date(weekly_df, picked_str)
    sync_to_registry(payload)

    # 2. เลือกจังหวัดแบบติ๊ก (ชมพูพาสเทล + ปุ่มเลือกทั้งหมด)
    render_html('<div class="sb-sec-hdr">2. เลือกจังหวัด (สคร.2)</div>')
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
    
        chk_lbl = f" จ.{p['province_name']} (เตือน {p['alert_count']} โรค)"
        if st.checkbox(chk_lbl, key=f"chk_p_{p['province_code']}", on_change=_sync_all_p):
            selected_provinces_objs.append(p)

    # 3. เลือกโรคแบบติ๊ก (ชมพูพาสเทล + ปุ่มเลือกทั้งหมด)
    render_html('<div class="sb-sec-hdr">3. เลือกกลุ่มโรคเฝ้าระวัง</div>')
    all_d_codes = ["02", "15", "31"]

    if "chk_d_all" not in st.session_state:
        st.session_state["chk_d_all"] = True
    for dc in all_d_codes:
        if f"chk_d_{dc}" not in st.session_state:
            st.session_state[f"chk_d_{dc}"] = True

    def _toggle_all_d():
        val = st.session_state["chk_d_all"]
        for dc in all_d_codes:
            st.session_state[f"chk_d_{dc}"] = val

    def _sync_all_d():
        st.session_state["chk_d_all"] = all(st.session_state[f"chk_d_{dc}"] for dc in all_d_codes)

    st.checkbox("**เลือกทุกโรค (ทั้ง 3 โรค)**", key="chk_d_all", on_change=_toggle_all_d)

    selected_disease_codes = []
    for d_code in all_d_codes:
        d_name = DISEASE_NAMES.get(d_code, d_code)
        if st.checkbox(f"{d_code} - {d_name}", key=f"chk_d_{d_code}", on_change=_sync_all_d):
            selected_disease_codes.append(d_code)

    # 4. กล่องขั้นตอนการใช้งานเว็บนี้
    render_html("""
    <div class="guide-card">
        <div class="guide-hdr">ขั้นตอนการใช้งานเว็บนี้</div>
        <div class="guide-step">
            <div class="guide-num">1</div>
            <div><b>เลือกตัวกรองด้านบน</b> — กำหนดวันตัดข้อมูล ติ๊กเลือกจังหวัด และโรคที่ต้องการติดตาม</div>
        </div>
        <div class="guide-step">
            <div class="guide-num">2</div>
            <div><b>ดูผลชี้เป้าและกราฟ (แท็บ 1–2)</b> — โรคที่ <b style="color:#be123c;">เกินเกณฑ์เตือน</b> จะถูกจัดอันดับ #1–#3 ให้ตรวจสอบก่อน</div>
        </div>
        <div class="guide-step">
            <div class="guide-num">3</div>
            <div><b>บันทึกและปิดยอด (แท็บ 3)</b> — กรอกผลลงตรวจจริงและยอดจริงนิ่งแล้ว เพื่อวัดความแม่นยำรายเดือน</div>
        </div>
    </div>
    """)

meta = payload["meta"]

if not selected_provinces_objs or not selected_disease_codes:
    st.warning("กรุณาติ๊กเลือกอย่างน้อย 1 จังหวัด และ 1 โรค จากแถบเมนูด้านซ้ายเพื่อแสดงข้อมูลครับ")
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

if len(selected_disease_codes) == len(DISEASE_NAMES):
    disease_scope_text = "ทุกโรคในขอบเขต (3 โรค)"
elif len(selected_disease_codes) == 1:
    disease_scope_text = DISEASE_NAMES.get(selected_disease_codes[0], selected_disease_codes[0])
else:
    disease_scope_text = f"เลือก {len(selected_disease_codes)} โรค"

flagged_items = [d for d in view_items if d["is_flagged"]]
top_alerts = flagged_items[:3]

if len(selected_provinces_objs) == len(payload["provinces"]):
    area_title = f"ภาพรวม สคร.2 ({len(payload['provinces'])} จังหวัด)"
elif len(selected_provinces_objs) == 1:
    area_title = f"จังหวัด{selected_prov_obj['province_name']}"
else:
    area_title = f"พื้นที่ที่เลือก ({len(selected_provinces_objs)} จังหวัด)"

# =========================================================
# 4. แถบนำทางพอร์ทัล + หัวข้อใหญ่ + 3 แท็บหลัก
# =========================================================
render_html("""
<div class="portal-navbar">
    <div class="portal-org">
        <span class="portal-dot"></span>
        <span>กรมควบคุมโรค • สำนักงานป้องกันควบคุมโรคที่ 2 จังหวัดพิษณุโลก</span>
    </div>
    <div class="portal-sys">กลุ่มระบาดวิทยาและตอบโต้ภาวะฉุกเฉินทางสาธารณสุข</div>
</div>
""")

status_cls = "meta-tag-warn" if "เบื้องต้น" in str(meta["data_status"]) else "meta-tag"
render_html(f"""
<div class="top-header">
    <div class="top-title">การชี้เป้าโรคที่ควรเฝ้าระวังสัปดาห์หน้าจากข้อมูล DDS — {area_title}</div>
    <div class="meta-line">
        <span class="meta-tag-blue">ขอบเขตโรค: {disease_scope_text}</span>
        <span class="meta-tag">รหัสรอบ: <b>{meta['run_id']}</b></span>
        <span class="meta-tag">สัปดาห์เป้าหมาย: <b>{meta['target_week']}</b></span>
        <span class="meta-tag">ตัดข้อมูลถึง: <b>{meta['cutoff_date']}</b></span>
        <span class="{status_cls}">สถานะ: {meta['data_status']}</span>
    </div>
</div>
""")

tab1, tab2, tab3 = st.tabs([
    "แดชบอร์ดชี้เป้าสัปดาห์หน้า",
    "กราฟแนวโน้มรายโรค",
    "บันทึกและติดตามผล"
])

# =========================================================
# แท็บ 1: แดชบอร์ดชี้เป้าสัปดาห์หน้า
# =========================================================
with tab1:
    total_fc = sum(to_int(d["forecast"]) for d in view_items)
    total_latest = sum(to_int(d["latest_cases"]) for d in view_items)
    diff_total = total_fc - total_latest
    diff_total_txt = f"+{diff_total:,} ราย จากสัปดาห์ล่าสุด" if diff_total >= 0 else f"{diff_total:,} ราย จากสัปดาห์ล่าสุด"

    if top_alerts:
        t1 = top_alerts[0]
        top_risk_name = f"{t1['disease_name']}"
        prov_prefix = f"จ.{t1['province_name']} • " if is_all_prov else ""
        top_risk_note = f"{prov_prefix}คาด {to_int(t1['forecast']):,} ราย (เกณฑ์ {to_int(t1['threshold']):,} ราย)"
    else:
        top_risk_name, top_risk_note = "ยังไม่มีโรคเกินเกณฑ์", "สถานการณ์อยู่ในเกณฑ์ปกติ"

    alert_color = "#be123c" if len(flagged_items) > 0 else "#006837"

    if is_all_prov:
        flagged_provs_cnt = len({d["province_code"] for d in flagged_items})
        kpi1_val = f"พบเตือน {flagged_provs_cnt} จังหวัด" if flagged_provs_cnt > 0 else "ปกติทุกจังหวัด"
        kpi1_note = f"เกินเกณฑ์รวม {len(flagged_items)} จาก {len(view_items)} รายการ"
    else:
        kpi1_val = f"เกินเกณฑ์ {len(flagged_items)} โรค" if len(flagged_items) > 0 else "ปกติทุกโรค"
        kpi1_note = f"จากทั้งหมด {len(view_items)} โรคที่เฝ้าระวัง"

    render_html(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-lbl">พื้นที่และโรคที่ต้องเฝ้าระวัง</div>
            <div class="kpi-val" style="color:{alert_color};">{kpi1_val}</div>
            <div class="kpi-note">{kpi1_note}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">ต้องระวังเป็นอันดับที่ 1</div>
            <div class="kpi-val" style="font-size:17.5px; color:{alert_color};">{top_risk_name}</div>
            <div class="kpi-note">{top_risk_note}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">ยอดคาดการณ์รวมสัปดาห์หน้า</div>
            <div class="kpi-val">{total_fc:,} ราย</div>
            <div class="kpi-note">{diff_total_txt}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">ยอดผู้ป่วยจริงรวม (สัปดาห์ล่าสุด)</div>
            <div class="kpi-val">{total_latest:,} ราย</div>
            <div class="kpi-note">ข้อมูล ณ วันตัดรอบ {meta['cutoff_date']}</div>
        </div>
    </div>
    """)

    col_chart, col_action = st.columns([1.25, 1.0], gap="medium")

    with col_chart:
        chart_slice = view_items[:6]
        chart_sub_extra = f" • แสดง {len(chart_slice)} อันดับเสี่ยงสูงสุด" if len(view_items) > 6 else ""
        render_html(f"""
        <div class="panel-box" style="margin-bottom:0; border-bottom-left-radius:0; border-bottom-right-radius:0; border-bottom:none;">
            <div class="panel-title">เปรียบเทียบ "ยอดคาดสัปดาห์หน้า" กับ "เกณฑ์เตือน"</div>
            <div class="panel-sub">สีแดง = เกินเกณฑ์เตือน • สีฟ้า = ปกติ{chart_sub_extra}</div>
        </div>
        """)

        chart_items = list(reversed(chart_slice))
        if is_all_prov and len(selected_disease_codes) > 1:
            y_names = [f"จ.{d['province_name']} - {d['disease_name']}" for d in chart_items]
        elif is_all_prov:
            y_names = [f"จ.{d['province_name']} ({d['province_code']})" for d in chart_items]
        else:
            y_names = [f"{d['disease_name']} ({d['disease_code']})" for d in chart_items]

        fc_vals = [to_int(d["forecast"]) for d in chart_items]
        th_vals = [to_int(d["threshold"]) for d in chart_items]
        fc_colors = ["#f43f5e" if d["is_flagged"] else "#38bdf8" for d in chart_items]
        fc_texts = [f" {to_int(d['forecast']):,} ราย (เกินเกณฑ์)" if d["is_flagged"] else f" {to_int(d['forecast']):,} ราย" for d in chart_items]
        th_texts = [f" {to_int(d['threshold']):,} ราย" for d in chart_items]

        fig_ov = go.Figure()
        fig_ov.add_trace(go.Bar(
            y=y_names, x=th_vals, name="เกณฑ์เตือนปกติ", orientation="h",
            marker=dict(color="#cbd5e1"), text=th_texts, textposition="outside",
            textfont=dict(size=11.5, color="#64748b"), hovertemplate="%{y}<br>เกณฑ์เตือน: %{x:,} ราย<extra></extra>"
        ))
        fig_ov.add_trace(go.Bar(
            y=y_names, x=fc_vals, name="ยอดคาดสัปดาห์หน้า", orientation="h",
            marker=dict(color=fc_colors), text=fc_texts, textposition="outside",
            textfont=dict(size=12, color="#0f172a"), hovertemplate="%{y}<br>ยอดคาดสัปดาห์หน้า: %{x:,} ราย<extra></extra>"
        ))

        max_x = max(fc_vals + th_vals + [1])
        fig_ov.update_layout(
            barmode="group", bargap=0.28, bargroupgap=0.12, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0, font=dict(size=12)),
            font=dict(family="Noto Sans Thai, sans-serif", size=12, color="#334155"),
            margin=dict(l=10, r=30, t=42, b=28), height=max(220, 75 + len(chart_items) * 58)
        )
        fig_ov.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False, range=[0, max_x * 1.48], nticks=5, tickformat=",d", title_text="จำนวนผู้ป่วย (ราย)")
        fig_ov.update_yaxes(showgrid=False, automargin=True)
        st.plotly_chart(fig_ov, use_container_width=True)

    with col_action:
        if not top_alerts:
            render_html(f"""
            <div class="safe-card">
                <b>สัปดาห์เป้าหมาย {meta['target_week']} ยังไม่มีสัญญาณเตือน</b><br>
                ยอดคาดการณ์ของ <b>{disease_scope_text}</b> ใน <b>{area_title}</b> ยังอยู่ต่ำกว่าเกณฑ์เตือนทั้งหมด
            </div>
            """)
        else:
            badge_classes = ["rank-b1", "rank-b2", "rank-b3"]
            items_html = []
            for i, item in enumerate(top_alerts):
                fc_i, th_i, lt_i = to_int(item["forecast"]), to_int(item["threshold"]), to_int(item["latest_cases"])
                lo_i, hi_i, diff = to_int(item["pi_low"]), to_int(item["pi_high"]), fc_i - th_i
                pt = get_prov_theme(item["province_code"])
                prov_pill = f'<span class="prov-pill" style="background:{pt["bg"]}; color:{pt["color"]}; margin-right:6px;">จ.{item["province_name"]}</span>' if is_all_prov else ""
                b_cls = badge_classes[min(i, 2)]

                items_html.append(f"""
                <div class="rank-item">
                    <div class="rank-head">
                        <div>{prov_pill}<span class="rank-title">{item['disease_name']}</span></div>
                        <span class="{b_cls}">อันดับ #{i+1}</span>
                    </div>
                    <ul class="rank-bullets">
                        <li>ยอดคาดสัปดาห์หน้า: <b style="color:#be123c;">{fc_i:,} ราย</b> <span class="muted">(ช่วง 95%: {lo_i:,}–{hi_i:,} ราย)</span></li>
                        <li>เกณฑ์เตือนปกติ: <b>{th_i:,} ราย</b> <span style="color:#be123c; font-weight:600;">(เกินเกณฑ์ +{diff:,} ราย)</span></li>
                        <li>ยอดจริงสัปดาห์ล่าสุด: <b>{lt_i:,} ราย</b></li>
                    </ul>
                </div>
                """)

            render_html(f"""
            <div class="rank-box">
                <div class="rank-box-header">
                    <span>อันดับที่ควรเฝ้าระวังก่อน (สูงสุด 3 อันดับแรก)</span>
                </div>
                {"".join(items_html)}
            </div>
            """)

    render_html('<div class="sec-label">ตารางสรุปตัวเลขทั้งหมด (เรียงลำดับความเสี่ยงจากมากไปน้อย)</div>')
    rows_html = []
    for rank_idx, d in enumerate(view_items, start=1):
        fc_i, th_i, lt_i = to_int(d["forecast"]), to_int(d["threshold"]), to_int(d["latest_cases"])
        lo_i, hi_i, diff = to_int(d["pi_low"]), to_int(d["pi_high"]), fc_i - th_i
        badge = '<span class="badge-red">เฝ้าระวัง</span>' if d["is_flagged"] else '<span class="badge-ok">ปกติ</span>'
        diff_txt = f'<span style="color:#be123c; font-weight:600;">+{diff:,}</span>' if d["is_flagged"] else f'<span style="color:#15803d;">{diff:,}</span>'

        pt = get_prov_theme(d["province_code"])
        prov_td = f'<td><span class="prov-pill" style="background:{pt["bg"]}; color:{pt["color"]};">จ.{d["province_name"]}</span></td>' if is_all_prov else ""

        rows_html.append(f"""
        <tr>
            <td class="ctr"><b>{rank_idx}</b></td>
            {prov_td}
            <td><b>{d['disease_name']}</b> <span class="muted">({d['disease_code']})</span></td>
            <td class="num">{lt_i:,}</td>
            <td class="num"><b>{fc_i:,}</b></td>
            <td class="num muted">{lo_i:,}–{hi_i:,}</td>
            <td class="num">{th_i:,}</td>
            <td class="num">{diff_txt}</td>
            <td class="ctr">{badge}</td>
        </tr>
        """)
    prov_th = "<th>จังหวัด</th>" if is_all_prov else ""
    render_html(f"""
    <div class="tbl-wrap"><table class="m-table">
        <thead>
            <tr>
                <th class="ctr">#</th>{prov_th}<th>ชื่อโรค</th>
                <th class="num">ล่าสุด</th><th class="num">คาดสัปดาห์หน้า</th>
                <th class="num">ช่วง 95%</th><th class="num">เกณฑ์เตือน</th>
                <th class="num">ส่วนต่าง</th><th class="ctr">สถานะ</th>
            </tr>
        </thead>
        <tbody>{"".join(rows_html)}</tbody>
    </table></div>
    """)

# =========================================================
# แท็บ 2: กราฟแนวโน้มรายโรค (แยก 2 ช่องเลือก + กราฟเส้นสีน้ำเงิน)
# =========================================================
with tab2:
    if len(view_items) == 1:
        d = view_items[0]
    else:
        c_p, c_d = st.columns(2)

        avail_provs = {}
        for it in view_items:
            if it["province_code"] not in avail_provs:
                pt = get_prov_theme(it["province_code"])
                avail_provs[it["province_code"]] = f"{pt['dot']} จ.{it['province_name']}"

        if len(avail_provs) > 1:
            prov_lbl_to_code = {v: k for k, v in avail_provs.items()}
            sel_p_lbl = c_p.selectbox("เลือกจังหวัด", list(prov_lbl_to_code.keys()), key="tab2_prov")
            pick_p_code = prov_lbl_to_code[sel_p_lbl]
        else:
            pick_p_code = list(avail_provs.keys())[0]
            c_p.selectbox("จังหวัด", list(avail_provs.values()), disabled=True, key="tab2_prov_fixed")

        prov_items = [it for it in view_items if it["province_code"] == pick_p_code]
        d_map = {}
        for it in prov_items:
            st_tag = "🔴 เฝ้าระวัง" if it["is_flagged"] else "🟢 ปกติ"
            d_map[f"{it['disease_name']} ({it['disease_code']}) — {st_tag}"] = it

        sel_d_lbl = c_d.selectbox("เลือกโรค", list(d_map.keys()), key="tab2_dis")
        d = d_map[sel_d_lbl]

    pt_sel = get_prov_theme(d["province_code"])
    fc_i, th_i, lt_i, lo_i, hi_i = to_int(d["forecast"]), to_int(d["threshold"]), to_int(d["latest_cases"]), to_int(d["pi_low"]), to_int(d["pi_high"])
    st_color = "#be123c" if d["is_flagged"] else "#006837"
    st_label = "เกินเกณฑ์เตือน (ควรเฝ้าระวัง)" if d["is_flagged"] else "ต่ำกว่าเกณฑ์ (ปกติ)"

    render_html(f"""
    <div class="kpi-row">
        <div class="kpi-card"><div class="kpi-lbl">ยอดจริงสัปดาห์ล่าสุด</div><div class="kpi-val">{lt_i:,} ราย</div></div>
        <div class="kpi-card"><div class="kpi-lbl">พยากรณ์สัปดาห์เป้าหมาย ({d['target_week']})</div><div class="kpi-val" style="color:{st_color};">{fc_i:,} ราย</div><div class="kpi-note">ช่วง 95%: {lo_i:,} – {hi_i:,} ราย</div></div>
        <div class="kpi-card"><div class="kpi-lbl">เกณฑ์เตือนรายโรค</div><div class="kpi-val">{th_i:,} ราย</div></div>
        <div class="kpi-card"><div class="kpi-lbl">ผลการเทียบเกณฑ์</div><div class="kpi-val" style="font-size:15.5px; color:{st_color}; margin-top:6px;">{st_label}</div></div>
    </div>
    """)

    render_html(f"""
    <div class="panel-box" style="margin-bottom:0; border-bottom-left-radius:0; border-bottom-right-radius:0; border-bottom:none;">
        <div class="panel-title">
            <span class="prov-pill" style="background:{pt_sel['bg']}; color:{pt_sel['color']}; margin-right:6px;">จ.{d['province_name']}</span>
            แนวโน้มผู้ป่วยรายสัปดาห์ — {d['disease_name']}
        </div>
        <div class="panel-sub">แสดงยอดย้อนหลัง 1 ปี เทียบกับเส้นเกณฑ์เตือนและจุดพยากรณ์สัปดาห์หน้า (ลากคลุมเพื่อซูม ดับเบิลคลิกเพื่อรีเซ็ต)</div>
    </div>
    """)

    hist = pd.DataFrame(d["history"])
    hist["week_dt"] = pd.to_datetime(hist["week"])
    hist["cases_num"] = pd.to_numeric(hist["cases"], errors="coerce")
    target_dt, first_dt = pd.to_datetime(d["target_week"]), hist["week_dt"].iloc[0]

    fig = go.Figure()
    prelim = hist[hist["status"].str.contains("เบื้องต้น", na=False)]
    if not prelim.empty:
        fig.add_vrect(x0=prelim["week_dt"].iloc[0] - pd.Timedelta(days=3), x1=prelim["week_dt"].iloc[-1] + pd.Timedelta(days=3), fillcolor="rgba(245, 158, 11, 0.12)", line_width=0)

    fig.add_trace(go.Scatter(x=[first_dt, target_dt], y=[th_i, th_i], mode="lines", name=f"เกณฑ์เตือน ({th_i:,} ราย)", line=dict(color="#f59e0b", width=2.2, dash="dot"), hovertemplate=f"เกณฑ์เตือน: {th_i:,} ราย<extra></extra>"))
    fig.add_trace(go.Scatter(x=hist["week_dt"], y=hist["cases_num"], mode="lines+markers", name=f"ยอดจริง จ.{d['province_name']}", line=dict(color="#2563eb", width=2.4), marker=dict(size=6.5, color="#2563eb", line=dict(color="#ffffff", width=1)), hovertemplate="สัปดาห์ %{x|%d %b %Y}<br>ยอดจริง: %{y:,.0f} ราย<extra></extra>"))

    valid = hist.dropna(subset=["cases_num"])
    if not valid.empty:
        fig.add_trace(go.Scatter(x=[valid["week_dt"].iloc[-1], target_dt], y=[valid["cases_num"].iloc[-1], fc_i], mode="lines", line=dict(color="#f43f5e", width=2, dash="dash"), showlegend=False, hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=[target_dt], y=[fc_i], mode="markers", name=f"พยากรณ์สัปดาห์หน้า ({fc_i:,} ราย)", marker=dict(color="#f43f5e", size=14, symbol="star", line=dict(color="#ffffff", width=1)), hovertemplate=f"สัปดาห์เป้าหมาย {d['target_week']}<br>พยากรณ์: {fc_i:,} ราย<br>ช่วง 95%: {lo_i:,}–{hi_i:,} ราย<extra></extra>"))
    if not prelim.empty:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="แถบเหลือง = ข้อมูลเบื้องต้น", marker=dict(symbol="square", size=10, color="rgba(245, 158, 11, 0.35)", line=dict(color="#f59e0b", width=1)), hoverinfo="skip"))

    y_max = max(float(valid["cases_num"].max() if not valid.empty else 0), float(th_i), float(fc_i), 1.0)
    fig.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0, font=dict(size=12.5)),
        font=dict(family="Noto Sans Thai, sans-serif", size=12.5, color="#64748b"),
        margin=dict(l=45, r=25, t=65, b=45), height=410
    )
    fig.update_xaxes(showgrid=False, tickformat="%b %Y", nticks=7, zeroline=False, automargin=True, range=[first_dt - pd.Timedelta(days=7), target_dt + pd.Timedelta(days=18)])
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f6", zeroline=False, nticks=6, tickformat=",d", automargin=True, range=[-max(0.3, y_max * 0.08), y_max * 1.22], title_text="จำนวนผู้ป่วย (ราย)")
    st.plotly_chart(fig, use_container_width=True)
# =========================================================
    # กราฟที่ 2 (ต่อจากกราฟ 1 ปี): ซูมแนวโน้ม 12 สัปดาห์ล่าสุด + พยากรณ์
    # =========================================================
    hist_12 = hist.tail(12).copy()
    first_dt_12 = hist_12["week_dt"].iloc[0]

    diff_wk = fc_i - lt_i
    if diff_wk > 0:
        trend_badge = f'<span style="background:#fee2e2; color:#b91c1c; border:1px solid #fca5a5; padding:3px 10px; border-radius:6px; font-weight:700; font-size:12.5px;">📈 สัปดาห์หน้าคาดว่าเพิ่มขึ้น +{diff_wk:,} ราย</span>'
    elif diff_wk < 0:
        trend_badge = f'<span style="background:#dcfce7; color:#15803d; border:1px solid #86efac; padding:3px 10px; border-radius:6px; font-weight:700; font-size:12.5px;">📉 สัปดาห์หน้าคาดว่าลดลง {diff_wk:,} ราย</span>'
    else:
        trend_badge = '<span style="background:#f1f5f9; color:#334155; border:1px solid #cbd5e1; padding:3px 10px; border-radius:6px; font-weight:700; font-size:12.5px;">➡️ สัปดาห์หน้าคาดว่าทรงตัว (0 ราย)</span>'

    render_html(f"""
    <div class="panel-box" style="margin-top:14px; margin-bottom:0; border-bottom-left-radius:0; border-bottom-right-radius:0; border-bottom:none;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:4px;">
            <div class="panel-title" style="margin:0;">
                <span class="prov-pill" style="background:{pt_sel['bg']}; color:{pt_sel['color']}; margin-right:6px;">จ.{d['province_name']}</span>
                ซูมแนวโน้มระยะสั้น (12 สัปดาห์ล่าสุด + พยากรณ์สัปดาห์หน้า) — {d['disease_name']}
            </div>
            {trend_badge}
        </div>
        <div class="panel-sub">แสดงตัวเลขรายสัปดาห์ช่วง 3 เดือนล่าสุด เพื่อดูทิศทางขึ้น–ลงก่อนเข้าสู่สัปดาห์เป้าหมาย ({d['target_week']})</div>
    </div>
    """)

    fig_12w = go.Figure()
    prelim_12 = hist_12[hist_12["status"].str.contains("เบื้องต้น", na=False)]
    if not prelim_12.empty:
        fig_12w.add_vrect(
            x0=prelim_12["week_dt"].iloc[0] - pd.Timedelta(days=3),
            x1=prelim_12["week_dt"].iloc[-1] + pd.Timedelta(days=3),
            fillcolor="rgba(245, 158, 11, 0.12)", line_width=0
        )

    fig_12w.add_trace(go.Scatter(
        x=[first_dt_12, target_dt], y=[th_i, th_i], mode="lines",
        name=f"เกณฑ์เตือน ({th_i:,} ราย)",
        line=dict(color="#f59e0b", width=2.2, dash="dot"),
        hovertemplate=f"เกณฑ์เตือน: {th_i:,} ราย<extra></extra>"
    ))

    pt_texts_12 = [f"{int(v):,}" if pd.notna(v) else "" for v in hist_12["cases_num"]]
    fig_12w.add_trace(go.Scatter(
        x=hist_12["week_dt"], y=hist_12["cases_num"],
        mode="lines+markers+text", text=pt_texts_12, textposition="top center",
        textfont=dict(size=11, color="#1e40af"),
        name=f"ยอดจริง จ.{d['province_name']}",
        line=dict(color="#2563eb", width=2.5),
        marker=dict(size=7.5, color="#2563eb", line=dict(color="#ffffff", width=1)),
        hovertemplate="สัปดาห์ %{x|%d %b %Y}<br>ยอดจริง: %{y:,.0f} ราย<extra></extra>"
    ))

    valid_12 = hist_12.dropna(subset=["cases_num"])
    if not valid_12.empty:
        fig_12w.add_trace(go.Scatter(
            x=[valid_12["week_dt"].iloc[-1], target_dt],
            y=[valid_12["cases_num"].iloc[-1], fc_i],
            mode="lines", line=dict(color="#f43f5e", width=2.4, dash="dash"),
            showlegend=False, hoverinfo="skip"
        ))

    fig_12w.add_trace(go.Scatter(
        x=[target_dt], y=[fc_i], mode="markers+text",
        text=[f"  {fc_i:,} ราย"], textposition="middle right",
        textfont=dict(size=12, color="#be123c", family="Prompt, Noto Sans Thai, sans-serif"),
        name=f"พยากรณ์สัปดาห์หน้า ({fc_i:,} ราย)",
        marker=dict(color="#f43f5e", size=14, symbol="star", line=dict(color="#ffffff", width=1)),
        hovertemplate=f"สัปดาห์เป้าหมาย {d['target_week']}<br>พยากรณ์: {fc_i:,} ราย<br>ช่วง 95%: {lo_i:,}–{hi_i:,} ราย<extra></extra>"
    ))

    if not prelim_12.empty:
        fig_12w.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name="แถบเหลือง = ข้อมูลเบื้องต้น",
            marker=dict(symbol="square", size=10, color="rgba(245, 158, 11, 0.35)", line=dict(color="#f59e0b", width=1)),
            hoverinfo="skip"
        ))

    y_max_12 = max(float(valid_12["cases_num"].max() if not valid_12.empty else 0), float(th_i), float(fc_i), 1.0)
    fig_12w.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0, font=dict(size=12.5)),
        font=dict(family="Noto Sans Thai, sans-serif", size=12.5, color="#64748b"),
        margin=dict(l=45, r=45, t=65, b=45), height=360
    )
    fig_12w.update_xaxes(
        showgrid=False, tickformat="%d %b %y", nticks=7, zeroline=False, automargin=True,
        range=[first_dt_12 - pd.Timedelta(days=4), target_dt + pd.Timedelta(days=14)]
    )
    fig_12w.update_yaxes(
        showgrid=True, gridcolor="#eef2f6", zeroline=False, nticks=6, tickformat=",d", automargin=True,
        range=[-max(0.3, y_max_12 * 0.08), y_max_12 * 1.25], title_text="จำนวนผู้ป่วย (ราย)"
    )
    st.plotly_chart(fig_12w, use_container_width=True, key="chart_12w")
    
    

# =========================================================
# แท็บ 3: บันทึกและติดตามผล + ตารางประวัติพับได้ + โหลด CSV
# =========================================================
with tab3:
    reg_df = load_registry(selected_prov_code, selected_disease_code)

    if reg_df.empty:
        st.info("ยังไม่มีรายการในทะเบียนตามจังหวัดและโรคที่เลือก")
    else:
        reg_df["month"] = reg_df["target_week"].astype(str).str.slice(0, 7)

        c_title, c_month = st.columns([3, 1])
        with c_title:
            render_html(f'<div class="sec-label" style="margin-top:0;">สรุปผลประเมินความแม่นยำรายเดือน — {area_title}</div>')
        sel_m = c_month.selectbox("เลือกเดือนประเมิน", ["ทุกเดือน"] + sorted(reg_df["month"].unique().tolist(), reverse=True), label_visibility="collapsed")

        ev = reg_df if sel_m == "ทุกเดือน" else reg_df[reg_df["month"] == sel_m]
        ev["settled_num"] = pd.to_numeric(ev["settled_actual_cases"], errors="coerce")
        closed = ev[ev["settled_num"].notna()].copy()

        if closed.empty:
            m1_val, m2_val, m3_val, m4_val = "รอปิดยอดจริง", "รอปิดยอดจริง", "0 ครั้ง", "รอระบุเวลา"
            m3_note = "คำนวณอัตโนมัติเมื่อกรอกยอดจริงนิ่งแล้ว"
        else:
            flagged = closed["is_flagged"].astype(str).str.lower() == "true"
            actual_met = closed["settled_num"] >= pd.to_numeric(closed["frozen_threshold"], errors="coerce")
            tp, fp, fn = int((flagged & actual_met).sum()), int((flagged & ~actual_met).sum()), int((~flagged & actual_met).sum())
            avg_hr = ((pd.to_datetime(closed["check_start_time"], errors="coerce") - pd.to_datetime(closed["alert_sent_time"], errors="coerce")).dt.total_seconds() / 3600).dropna().mean()
            m1_val, m2_val = ratio_text(tp, tp + fp), ratio_text(tp, tp + fn)
            m3_val, m3_note = f"รวม {tp + fp} (เกิน {fp})", f"เตือนทั้งหมด {tp + fp} ครั้ง / เตือนเกิน {fp} ครั้ง"
            m4_val = f"{avg_hr:.1f} ชม." if pd.notna(avg_hr) else "ยังไม่ระบุเวลา"

        render_html(f"""
        <div class="kpi-row">
            <div class="kpi-card-accent"><div class="kpi-lbl">1. ความแม่นของสัญญาณ</div><div class="kpi-val">{m1_val}</div><div class="kpi-note">เตือนแล้วเข้าเกณฑ์จริง ÷ เตือนทั้งหมด</div></div>
            <div class="kpi-card-accent"><div class="kpi-lbl">2. ความครอบคลุม</div><div class="kpi-val">{m2_val}</div><div class="kpi-note">จับเหตุการณ์ที่พุ่งเกินเกณฑ์จริงได้</div></div>
            <div class="kpi-card-accent"><div class="kpi-lbl">3. ภาระตรวจสอบ</div><div class="kpi-val">{m3_val}</div><div class="kpi-note">{m3_note}</div></div>
            <div class="kpi-card-accent"><div class="kpi-lbl">4. ความเร็วเริ่มตรวจ</div><div class="kpi-val">{m4_val}</div><div class="kpi-note">เฉลี่ยจากเวลาส่งสัญญาณถึงเริ่มตรวจ</div></div>
        </div>
        """)

        with st.container(border=True):
            render_html('<div class="panel-title" style="margin-bottom:8px;">บันทึกผลการตรวจสอบภาคสนาม / ปิดยอดจริง</div>')
            reg_df["label"] = (
                reg_df["run_id"] + "  |  จ." + reg_df["province_name"].astype(str) +
                "  |  " + reg_df["disease_name"] + "  |  สัปดาห์ " + reg_df["target_week"]
            )
            rec = reg_df[reg_df["label"] == st.selectbox("เลือกรอบรายงานที่ต้องการบันทึก", reg_df["label"].tolist())].iloc[0]

            rec_status = '<span class="badge-red">ชี้เป้าเฝ้าระวัง</span>' if str(rec["is_flagged"]).lower() == "true" else '<span class="badge-ok">ปกติ</span>'
            render_html(f"""
            <div class="rec-preview">
                <span><b>สถานะรอบนี้:</b> {rec_status}</span>
                <span><b>ยอดคาดการณ์เดิม:</b> {fmt_int(rec['frozen_forecast'])} ราย</span>
                <span><b>เกณฑ์เตือนเดิม:</b> {fmt_int(rec['frozen_threshold'])} ราย</span>
                <span><b>เวลาส่งสัญญาณ:</b> {clean_str(rec['alert_sent_time']) or '-'}</span>
            </div>
            """)

            with st.form("track_form", border=False):
                c1, c2 = st.columns(2)
                in_time = c1.text_input("1. เวลาเริ่มตรวจ (ใช้คิดความเร็ว)", value=clean_str(rec["check_start_time"]), placeholder="เช่น 2026-11-23 11:30")
                f_opts = ["", "พบกลุ่มก้อน", "พบการส่งย้อนหลัง", "ยังสรุปไม่ได้", "ปกติ"]
                cur_f = clean_str(rec["field_result"])
                in_field = c2.selectbox("2. ผลตรวจสอบภาคสนาม", f_opts, index=f_opts.index(cur_f) if cur_f in f_opts else 0)

                c3, c4 = st.columns(2)
                in_action = c3.text_input("3. การดำเนินการ", value=clean_str(rec["action_taken"]), placeholder="เช่น แจ้งทีม SRRT พื้นที่ตรวจสอบซ้ำ")
                raw_act = clean_str(rec["settled_actual_cases"])
                in_actual = c4.text_input("4. ยอดจริงนิ่งแล้ว (กรอกเมื่อครบ 1–2 สัปดาห์)", value=str(to_int(raw_act)) if raw_act != "" else "", placeholder="จำนวนเต็ม (เว้นว่างได้หากยอดยังไม่นิ่ง)")

                if st.form_submit_button("บันทึกผลและอัปเดตตัวชี้วัด", type="primary"):
                    actual_val = None
                    if in_actual.strip() != "":
                        try:
                            actual_val = int(round(float(in_actual.strip())))
                            if actual_val < 0:
                                st.error("ยอดผู้ป่วยจริงต้องไม่ติดลบ")
                                st.stop()
                        except ValueError:
                            st.error("กรุณากรอกยอดจริงเป็นตัวเลขจำนวนเต็ม")
                            st.stop()
                    update_registry(rec["run_id"], rec["province_code"], rec["disease_code"], rec["target_week"], in_time.strip(), in_field, in_action.strip(), actual_val)
                    st.success("บันทึกเรียบร้อยแล้ว")
                    st.rerun()

        with st.expander(f"ดูทะเบียนประวัติการติดตามผลทั้งหมด ({len(reg_df):,} รายการ)", expanded=False):
            c_flt, c_dl = st.columns([2.5, 1.2])
            with c_flt:
                tbl_filter = st.radio(
                    "กรองทะเบียน",
                    ["ทั้งหมด", "เฉพาะที่ชี้เป้า", "รอปิดยอดจริง"],
                    horizontal=True,
                    label_visibility="collapsed"
                )

            show_reg = reg_df.copy()
            if tbl_filter == "เฉพาะที่ชี้เป้า":
                show_reg = show_reg[show_reg["is_flagged"].astype(str).str.lower() == "true"]
            elif tbl_filter == "รอปิดยอดจริง":
                show_reg = show_reg[
                    show_reg["settled_actual_cases"].isna() |
                    (show_reg["settled_actual_cases"].astype(str).str.strip() == "")
                ]

            with c_dl:
                export_cols = [
                    "run_id", "target_week", "province_code", "province_name",
                    "disease_code", "disease_name", "frozen_forecast", "frozen_threshold",
                    "is_flagged", "alert_sent_time", "check_start_time",
                    "field_result", "action_taken", "settled_actual_cases"
                ]
                valid_cols = [c for c in export_cols if c in show_reg.columns]
                csv_bytes = show_reg[valid_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    label="ดาวน์โหลดทะเบียน (CSV)",
                    data=csv_bytes,
                    file_name=f"tracking_registry_{meta['cutoff_date']}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            if show_reg.empty:
                st.info("ไม่พบรายการตามตัวกรองที่เลือก")
            else:
                reg_rows = []
                for _, r in show_reg.iterrows():
                    f_badge = '<span class="badge-red">ชี้เป้า</span>' if str(r["is_flagged"]).lower() == "true" else '<span class="badge-ok">ปกติ</span>'
                    act_s = clean_str(r["settled_actual_cases"])
                    act_txt = f"<b>{fmt_int(act_s)} ราย</b>" if act_s != "" else '<span class="muted">รอยอดนิ่ง</span>'
                    pt = get_prov_theme(r["province_code"])
                    prov_td = f'<td><span class="prov-pill" style="background:{pt["bg"]}; color:{pt["color"]};">จ.{r["province_name"]}</span></td>' if is_all_prov else ""
                    reg_rows.append(f"""
                    <tr>
                        <td><b>{r['target_week']}</b><div class="muted">{r['run_id']}</div></td>
                        {prov_td}
                        <td><b>{r['disease_name']}</b></td>
                        <td class="num"><b>{fmt_int(r['frozen_forecast'])}</b> / <span class="muted">{fmt_int(r['frozen_threshold'])}</span></td>
                        <td class="ctr">{f_badge}</td>
                        <td>{clean_str(r['field_result']) or '<span class="muted">-</span>'}</td>
                        <td>{clean_str(r['action_taken']) or '<span class="muted">-</span>'}</td>
                        <td class="num">{act_txt}</td>
                    </tr>
                    """)
                prov_th = "<th>จังหวัด</th>" if is_all_prov else ""
                render_html(f"""
                <div class="tbl-wrap"><table class="m-table">
                    <thead>
                        <tr>
                            <th>สัปดาห์เป้าหมาย</th>{prov_th}<th>โรค</th>
                            <th class="num">คาด / เกณฑ์</th><th class="ctr">สถานะ</th>
                            <th>ผลภาคสนาม</th><th>การดำเนินการ</th><th class="num">ยอดจริงนิ่งแล้ว</th>
                        </tr>
                    </thead>
                    <tbody>{"".join(reg_rows)}</tbody>
                </table></div>
                """)

# =========================================================
# 5. แถบท้ายเว็บไซต์ (Portal Footer)
# =========================================================
render_html("""
<div class="portal-footer">
    <span>© 2026 สำนักงานป้องกันควบคุมโรคที่ 2 จังหวัดพิษณุโลก • กรมควบคุมโรค กระทรวงสาธารณสุข</span>
    <span>ระบบสนับสนุนการตัดสินใจเฝ้าระวังโรค (DDS Weekly Surveillance Forecast)</span>
</div>
""")
