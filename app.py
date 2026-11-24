import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# สร้างไอคอนโล่สีเขียว-ขาวสำหรับแสดงบนแถบแท็บของเบราว์เซอร์ (Favicon)
FAVICON_PATH = "/tmp/odpc2_shield_icon.svg"
with open(FAVICON_PATH, "w", encoding="utf-8") as _f:
    _f.write("""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="16" fill="#00522b"/>
  <rect x="6" y="6" width="52" height="52" rx="12" fill="#ffffff" fill-opacity="0.14" stroke="#ffffff" stroke-opacity="0.32" stroke-width="1.5"/>
  <path d="M32 16L18 22v11c0 9.5 6 18 14 21 8-3 14-11.5 14-21V22L32 16z" fill="none" stroke="#ffffff" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M26 32.5l4.2 4.2 8.3-8.7" fill="none" stroke="#ffffff" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round"/>
</svg>""")

st.set_page_config(
    page_title="ระบบชี้เป้าโรคเฝ้าระวังสัปดาห์หน้า | สคร.2 จ.พิษณุโลก",
    page_icon=FAVICON_PATH,
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


def normalize_pcode(val):
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits.zfill(2) if digits else ""


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


# รายชื่อ 8 คอลัมน์ที่อนุญาตให้นำเข้าตามมาตรฐาน PDPA (Whitelist)
ALLOWED_DDS_COLUMNS = [
    "viewcode",
    "รหัสจังหวัดขณะป่วย",
    "จังหวัดขณะป่วย",
    "epidem_report_group_code",
    "epidem_report_group_name",
    "วันที่เริ่มมีอาการ",
    "วันที่เริ่มรักษา",
    "วันที่วินิจฉัยโรค",
]

DEFAULT_DISEASE_NAMES = {
    "02": "โรคอุจจาระร่วงเฉียบพลัน",
    "15": "โรคไข้หวัดใหญ่",
    "31": "โรคปอดอักเสบ"
}

# ตั้งค่าขอบเขตกลุ่มโรค: True = กรองเฉพาะ 3 โรคหลัก | False = เปิดรับทุกโรคในไฟล์ DDS อัตโนมัติ
RESTRICT_TO_CORE_DISEASES = True
CORE_DISEASE_CODES = {"02", "15", "31"}

ODPC2_PROVINCES = {
    "65": "พิษณุโลก",
    "53": "อุตรดิตถ์",
    "63": "ตาก",
    "64": "สุโขทัย",
    "67": "เพชรบูรณ์",
}

PROV_NAME_TO_CODE = {v: k for k, v in ODPC2_PROVINCES.items()}

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
# สไตล์ Executive Command Center (Vibrant Public Health UI)
# =========================================================
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&display=swap');

:root {
    --brand: #006837;
    --brand-dark: #004d28;
    --brand-mid: #059669;
    --brand-light: #e6f4ee;
    --brand-border: #9fd3bc;
    --bg-canvas: #e6eeeb;
    --surface: #ffffff;
    --border: #d5e0dc;
    --text-main: #0f172a;
    --text-sub: #475569;
    --text-muted: #64748b;
    --shadow-card: 0 6px 20px rgba(15, 23, 42, 0.05), 0 1px 3px rgba(15, 23, 42, 0.03);
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
    padding-top: 0.8rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1320px !important;
}

/* ================= SIDEBAR (NAVIGATION RAIL) ================= */
[data-testid="stSidebar"] {
    background-color: #f8fbf9;
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
    padding: 15px 16px;
    background: linear-gradient(135deg, #004d28 0%, #059669 100%);
    border-radius: 18px;
    margin-bottom: 14px;
    box-shadow: 0 8px 20px rgba(0, 104, 55, 0.22);
}
.sb-brand-icon {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.35);
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
    border-radius: 12px;
    padding: 6px 12px;
    margin: 14px 0 8px 0;
}

.pdpa-pill {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
    font-size: 11px;
    font-weight: 600;
    padding: 6px 10px;
    border-radius: 10px;
    margin-bottom: 8px;
    line-height: 1.45;
}

[data-testid="stSidebar"] .stCheckbox {
    padding: 3px 8px;
    border-radius: 10px;
    margin-bottom: -2px;
}
[data-testid="stSidebar"] .stCheckbox:hover {
    background-color: #e6f4ee;
}

.sb-guide {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 15px 16px;
    margin-top: 18px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
}
.sb-guide-hdr {
    font-size: 12.5px;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 10px;
    padding-bottom: 7px;
    border-bottom: 1px solid #eef3f1;
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
    background: linear-gradient(135deg, #006837 0%, #059669 100%);
    color: #ffffff;
    font-size: 10.5px;
    font-weight: 700;
    min-width: 20px;
    height: 20px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 2px;
}

/* ================= EXECUTIVE EMERALD HERO BANNER ================= */
.exec-hero {
    background: linear-gradient(125deg, #003d20 0%, #006837 52%, #059669 100%);
    border-radius: 24px;
    padding: 20px 26px;
    margin-bottom: 16px;
    box-shadow: 0 10px 28px rgba(0, 82, 43, 0.18);
    color: #ffffff;
    position: relative;
    overflow: hidden;
}
.exec-hero::after {
    content: "";
    position: absolute;
    right: -40px;
    top: -40px;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0) 70%);
    pointer-events: none;
}
.exec-org-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    padding-bottom: 11px;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.16);
}
.exec-org-left {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}
.exec-live-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 0 4px rgba(52, 211, 153, 0.25);
}
.exec-org-title { font-size: 13.5px; font-weight: 700; color: #ffffff; }
.exec-org-dept { font-size: 12px; color: #a7f3d0; border-left: 1px solid rgba(255, 255, 255, 0.25); padding-left: 10px; }
.exec-sys-tag {
    font-size: 11.5px;
    font-weight: 600;
    color: #ecfdf5;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.22);
    padding: 3px 11px;
    border-radius: 999px;
}
.exec-main-title { font-size: 21px; font-weight: 700; color: #ffffff; margin: 0 0 14px 0; line-height: 1.35; }
.exec-chips { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.echip-gold {
    background: #fbbf24;
    color: #1e293b;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12);
}
.echip-glass {
    background: rgba(255, 255, 255, 0.13);
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.22);
    padding: 5px 13px;
    border-radius: 999px;
    font-size: 12px;
}
.echip-warn {
    background: #fef3c7;
    color: #92400e;
    padding: 5px 13px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
}

/* ================= COLORFUL EXECUTIVE KPI CARDS ================= */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 15px;
    margin: 4px 0 18px 0;
}
.kpi-tile {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 18px 20px;
    box-shadow: var(--shadow-card);
    display: flex;
    align-items: flex-start;
    gap: 14px;
    position: relative;
    overflow: hidden;
}
.kpi-tile::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; }
.kpi-rose::before { background: linear-gradient(180deg, #e11d48 0%, #fb7185 100%); }
.kpi-amber::before { background: linear-gradient(180deg, #d97706 0%, #fbbf24 100%); }
.kpi-emerald::before { background: linear-gradient(180deg, #006837 0%, #10b981 100%); }
.kpi-sky::before { background: linear-gradient(180deg, #0284c7 0%, #38bdf8 100%); }

.kpi-icon-box {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}
.kicon-rose { background: #ffe4e6; color: #be123c; }
.kicon-amber { background: #fef3c7; color: #b45309; }
.kicon-emerald { background: #d1fae5; color: #047857; }
.kicon-sky { background: #e0f2fe; color: #0369a1; }

.kpi-body { flex: 1; min-width: 0; }
.kpi-lbl { font-size: 12px; font-weight: 600; color: var(--text-muted); }
.kpi-val {
    font-size: 21px;
    font-weight: 700;
    color: var(--text-main);
    margin: 4px 0 3px 0;
    font-variant-numeric: tabular-nums;
    line-height: 1.25;
}
.kpi-sub { font-size: 12px; color: var(--text-sub); }

/* ================= 5-PROVINCE REGIONAL PULSE STRIP ================= */
.prov-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 18px;
}
.prov-mini-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 12px 15px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
}
.prov-mini-alert { background: linear-gradient(135deg, #ffffff 0%, #fff1f2 100%); border-color: #fecdd3; }
.prov-mini-ok { background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%); border-color: #bbf7d0; }
.prov-mini-name { font-size: 13.5px; font-weight: 700; color: var(--text-main); }
.prov-mini-sub { font-size: 11.5px; color: var(--text-muted); margin-top: 2px; }
.pbadge-alert { background: #be123c; color: #ffffff; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 999px; white-space: nowrap; }
.pbadge-ok { background: #047857; color: #ffffff; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 999px; white-space: nowrap; }

/* ================= CONTAINERS & DATAFRAME ================= */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 22px !important;
    padding: 22px 26px !important;
    box-shadow: var(--shadow-card) !important;
    margin-bottom: 16px !important;
}

[data-testid="stDataFrame"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-card) !important;
}

.card-hdr { display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px; margin-bottom: 6px; }
.card-h3 { font-size: 15.5px; font-weight: 700; color: var(--text-main); margin: 0; }
.card-desc { font-size: 12.5px; color: var(--text-muted); margin: 3px 0 0 0; }

.prov-tag { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; white-space: nowrap; }

/* ================= PRIORITY ALERT PANEL ================= */
.alert-box {
    background: var(--surface);
    border: 1px solid #fecaca;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: var(--shadow-card);
    margin-bottom: 14px;
}
.alert-box-hdr {
    background: linear-gradient(90deg, #991b1b 0%, #be123c 100%);
    padding: 12px 18px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13.5px;
    font-weight: 700;
    color: #ffffff;
}
.alert-item { padding: 12px 18px; border-bottom: 1px solid #f1f5f9; }
.alert-item:last-child { border-bottom: none; }
.alert-item:first-of-type { background: #fff5f5; }
.alert-item-top { display: flex; justify-content: space-between; align-items: center; gap: 6px; margin-bottom: 4px; }
.alert-item-title { font-size: 13.5px; font-weight: 700; color: var(--text-main); }
.r-tag-1 { background: #be123c; color: #ffffff; font-size: 11px; font-weight: 700; padding: 2px 9px; border-radius: 999px; white-space: nowrap; }
.r-tag-2 { background: #ea580c; color: #ffffff; font-size: 11px; font-weight: 700; padding: 2px 9px; border-radius: 999px; white-space: nowrap; }
.r-tag-3 { background: #d97706; color: #ffffff; font-size: 11px; font-weight: 700; padding: 2px 9px; border-radius: 999px; white-space: nowrap; }
.alert-meta-line { font-size: 12px; color: var(--text-sub); line-height: 1.5; }
.alert-ci-sub { font-size: 11.5px; color: var(--text-muted); margin-top: 2px; }

.safe-box {
    background: linear-gradient(135deg, #ffffff 0%, #ecfdf5 100%);
    border: 1px solid var(--brand-border);
    border-left: 5px solid var(--brand);
    border-radius: 20px;
    padding: 18px 22px;
    color: #065f46;
    font-size: 13.5px;
    line-height: 1.65;
    box-shadow: var(--shadow-card);
    margin-bottom: 14px;
}

.sec-hdr { display: flex; align-items: center; gap: 10px; margin: 20px 0 12px 0; }
.sec-bar { width: 5px; height: 20px; background: linear-gradient(180deg, #006837 0%, #10b981 100%); border-radius: 999px; }
.sec-txt { font-size: 15.5px; font-weight: 700; color: var(--text-main); }

.legend-inline {
    background: #f4f8f6;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 13px 18px;
    margin-top: 10px;
    font-size: 12.5px;
    color: var(--text-sub);
    line-height: 1.7;
}

.rec-bar {
    background: #f4f8f6;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 11px 16px;
    margin: 8px 0 14px 0;
    font-size: 13px;
    color: var(--text-sub);
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    align-items: center;
}

/* ================= PILL NAVIGATION, INPUTS & BUTTONS ================= */
[data-testid="stSegmentedControl"] > div {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    padding: 5px !important;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04) !important;
}
[data-testid="stSegmentedControl"] button { border-radius: 999px !important; font-weight: 600 !important; }
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
    background-color: var(--brand) !important;
    color: #ffffff !important;
}

[data-baseweb="select"] > div, [data-baseweb="input"] > div {
    border-radius: 12px !important;
    border-color: var(--border) !important;
}

.stButton > button[kind="primary"], .stFormSubmitButton > button {
    background: linear-gradient(135deg, #006837 0%, #059669 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.52rem 1.5rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(0, 104, 55, 0.22) !important;
}
.stButton > button[kind="secondary"], [data-testid="stPopover"] > button, [data-testid="stDownloadButton"] > button {
    background-color: var(--surface) !important;
    color: var(--brand) !important;
    border: 1px solid var(--brand-border) !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
}
.stButton > button[kind="secondary"]:hover, [data-testid="stPopover"] > button:hover, [data-testid="stDownloadButton"] > button:hover {
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
    **1. การคุ้มครองข้อมูลส่วนบุคคล 100% (PDPA Whitelist)**
    * เมื่อนำเข้าไฟล์ข้อมูลดิบจากระบบ **DDS** ระบบจะประมวลผลในหน่วยความจำชั่วคราว (RAM) และอ่านเข้าโปรแกรม **เฉพาะ 8 คอลัมน์ที่จำเป็น** ได้แก่ `viewcode`, `รหัสจังหวัดขณะป่วย`, `จังหวัดขณะป่วย`, `epidem_report_group_code`, `epidem_report_group_name`, `วันที่เริ่มมีอาการ`, `วันที่เริ่มรักษา` และ `วันที่วินิจฉัยโรค`
    * ข้อมูลส่วนบุคคลอื่น ๆ จะไม่ถูกโหลดเข้าสู่ระบบ และรหัส `viewcode` จะถูกทำลายทิ้งทันทีเมื่อรวมยอดรายสัปดาห์แล้ว

    **2. เกณฑ์เตือนปกติ (Alert Threshold — P75)**
    * คำนวณแยกตามรายจังหวัดและรายโรค จากค่า **เปอร์เซ็นไทล์ที่ 75 (P75)** ของจำนวนผู้ป่วยรายสัปดาห์ย้อนหลังสูงสุด 104 สัปดาห์ (2 ปี)
    * หากยอดคาดการณ์สัปดาห์หน้าสูงเท่ากับหรือมากกว่าค่า P75 ระบบจะติดป้าย **"เฝ้าระวัง"** เพื่อให้ทีมระบาดวิทยาตรวจสอบเป็นลำดับแรก

    **3. แบบจำลองพยากรณ์ล่วงหน้า 1 สัปดาห์**
    * ใช้แบบจำลองอนุกรมเวลา **Simple Exponential Smoothing** พร้อมคำนวณช่วงความเชื่อมั่น **95% Prediction Interval** เพื่อประมาณการแนวโน้มสัปดาห์ถัดไป

    **4. ข้อควรระวังเรื่องความครบถ้วนของข้อมูล**
    * ข้อมูลช่วง **1–2 สัปดาห์ล่าสุด** เป็นข้อมูลเบื้องต้นที่หน่วยบริการกำลังทยอยบันทึกเข้าระบบ (Reporting Lag) ควรตรวจสอบร่วมกับผลสอบสวนโรคภาคสนามก่อนสรุปผล
    """)


# =========================================================
# 1. ระบบฐานข้อมูล PostgreSQL / CSV (ทะเบียนติดตามผล & ยอดรวมรายสัปดาห์)
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
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS weekly_surveillance_agg (
                province_code TEXT, province_name TEXT,
                disease_code TEXT, disease_name TEXT,
                week_start TEXT, cases INTEGER, data_status TEXT,
                PRIMARY KEY (province_code, disease_code, week_start)
            );
        """))


def save_weekly_agg_store(df_weekly):
    """บันทึกเฉพาะยอดสรุปรวมระดับจังหวัด-โรค-สัปดาห์ (ไม่มีข้อมูลส่วนบุคคล 100%)"""
    out_df = df_weekly.copy()
    out_df["week_start"] = pd.to_datetime(out_df["week_start"]).dt.strftime("%Y-%m-%d")
    out_df["cases"] = pd.to_numeric(out_df["cases"], errors="coerce").fillna(0).astype(int)

    engine = get_db_engine()
    if engine:
        ensure_table(engine)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM weekly_surveillance_agg;"))
            out_df[["province_code", "province_name", "disease_code", "disease_name", "week_start", "cases", "data_status"]].to_sql(
                "weekly_surveillance_agg", conn, if_exists="append", index=False
            )
    try:
        out_df.to_csv(WEEKLY_FILE, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def load_weekly_agg_from_db():
    engine = get_db_engine()
    if not engine:
        return None
    try:
        ensure_table(engine)
        df_db = pd.read_sql("SELECT * FROM weekly_surveillance_agg", engine)
        if not df_db.empty:
            df_db["week_start"] = pd.to_datetime(df_db["week_start"])
            df_db["province_code"] = df_db["province_code"].astype(str).str.strip()
            df_db["disease_code"] = df_db["disease_code"].apply(normalize_dcode)
            return df_db
    except Exception:
        return None
    return None


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
def split_check_datetime(dt_str):
    """แยกค่า YYYY-MM-DD HH:MM จากฐานข้อมูลออกเป็น (วว/ดด/ปปปป, HH:MM) เพื่อแสดงใน 2 กล่อง"""
    s = clean_str(dt_str)
    if not s:
        return "", ""
    ts = pd.to_datetime(s, errors="coerce")
    if pd.notna(ts):
        return ts.strftime("%d/%m/%Y"), ts.strftime("%H:%M")
    parts = s.split()
    return (parts[0], parts[1] if len(parts) > 1 else "")


def parse_check_datetime(date_txt, time_txt):
    """แปลงวันที่ (วว/ดด/ปปปป รองรับทั้ง พ.ศ. และ ค.ศ.) + เวลา (HH:MM) เป็นมาตรฐาน YYYY-MM-DD HH:MM"""
    now_th = pd.Timestamp.now(tz="Asia/Bangkok")
    d_raw = str(date_txt).strip()
    t_raw = str(time_txt).strip()

    # หากเว้นว่างทั้งคู่ ให้ตั้งค่าวันและเวลาปัจจุบันให้อัตโนมัติ
    if d_raw == "" and t_raw == "":
        return now_th.strftime("%Y-%m-%d %H:%M")

    # จัดการส่วนวันที่ (หากกรอกเฉพาะเวลา ให้ใช้วันที่ปัจจุบัน)
    if d_raw == "":
        y, m, d = now_th.year, now_th.month, now_th.day
    else:
        m_dmy = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$", d_raw)
        m_ymd = re.match(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$", d_raw)
        if m_dmy:
            d, m, y = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
        elif m_ymd:
            y, m, d = int(m_ymd.group(1)), int(m_ymd.group(2)), int(m_ymd.group(3))
        else:
            raise ValueError("รูปแบบวันที่ไม่ถูกต้อง กรุณากรอกเป็น วว/ดด/ปปปป (เช่น 24/11/2569)")
        if y >= 2400:
            y -= 543
        pd.Timestamp(year=y, month=m, day=d)  # ตรวจสอบความถูกต้องของปฏิทิน

    # จัดการส่วนเวลา (หากกรอกเฉพาะวันที่ ให้ใช้เวลาปัจจุบัน)
    if t_raw == "":
        hh, mm = now_th.hour, now_th.minute
    else:
        t_clean = t_raw.replace(".", ":")
        if re.match(r"^\d{4}$", t_clean):
            t_clean = f"{t_clean[:2]}:{t_clean[2:]}"
        m_tm = re.match(r"^(\d{1,2}):(\d{2})$", t_clean)
        if not m_tm:
            raise ValueError("รูปแบบเวลาไม่ถูกต้อง กรุณากรอกเป็นตัวเลข เช่น 11:30")
        hh, mm = int(m_tm.group(1)), int(m_tm.group(2))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError("เวลาต้องอยู่ระหว่าง 00:00 ถึง 23:59")

    return f"{y:04d}-{m:02d}-{d:02d} {hh:02d}:{mm:02d}"


def auto_settle_from_weekly(weekly_df):
    """ดึงยอดผู้ป่วยจริงที่นิ่งแล้ว (สถานะ 'ครบถ้วน') มาปิดยอดในทะเบียนติดตามผลให้อัตโนมัติ"""
    if weekly_df is None or weekly_df.empty:
        return 0

    w_copy = weekly_df.copy()
    w_copy["week_str"] = pd.to_datetime(w_copy["week_start"]).dt.strftime("%Y-%m-%d")
    w_copy["province_code"] = w_copy["province_code"].astype(str).str.strip()
    w_copy["disease_code"] = w_copy["disease_code"].apply(normalize_dcode)

    settled_w = w_copy[w_copy["data_status"] == "ครบถ้วน"]
    settled_map = {
        (r["province_code"], r["disease_code"], r["week_str"]): int(round(float(r["cases"])))
        for _, r in settled_w.iterrows()
    }
    if not settled_map:
        return 0

    engine = get_db_engine()
    updated_cnt = 0
    if engine:
        ensure_table(engine)
        df_open = pd.read_sql(
            "SELECT run_id, province_code, disease_code, target_week FROM tracking_registry WHERE settled_actual_cases IS NULL",
            engine
        )
        with engine.begin() as conn:
            for _, r in df_open.iterrows():
                key = (str(r["province_code"]).strip(), normalize_dcode(r["disease_code"]), str(r["target_week"]).strip())
                if key in settled_map:
                    conn.execute(text("""
                        UPDATE tracking_registry
                        SET settled_actual_cases = :ac
                        WHERE run_id = :rid AND province_code = :pc AND disease_code = :dc AND target_week = :tw;
                    """), {
                        "ac": settled_map[key],
                        "rid": r["run_id"],
                        "pc": str(r["province_code"]).strip(),
                        "dc": normalize_dcode(r["disease_code"]),
                        "tw": str(r["target_week"]).strip()
                    })
                    updated_cnt += 1
    elif os.path.exists(CSV_REGISTRY):
        df_all = pd.read_csv(CSV_REGISTRY, dtype={"province_code": str, "disease_code": str})
        for idx, r in df_all.iterrows():
            if pd.isna(r["settled_actual_cases"]) or str(r["settled_actual_cases"]).strip() == "":
                key = (str(r["province_code"]).strip(), normalize_dcode(r["disease_code"]), str(r["target_week"]).strip())
                if key in settled_map:
                    df_all.at[idx, "settled_actual_cases"] = settled_map[key]
                    updated_cnt += 1
        if updated_cnt > 0:
            df_all.to_csv(CSV_REGISTRY, index=False, encoding="utf-8-sig")

    return updated_cnt

# =========================================================
# 2. ระบบนำเข้าและคลีนข้อมูลดิบ DDS แบบ Real-Time (Strict 8 Columns PDPA)
# =========================================================
def parse_thai_date_series(series):
    """แปลงวันที่ทั้งรูปแบบ ค.ศ. และ พ.ศ. (เช่น 2568-11-20 หรือ 20/11/2568) เป็น datetime"""
    if pd.api.types.is_datetime64_any_dtype(series):
        s_dt = pd.to_datetime(series, errors="coerce")
        mask_be = s_dt.dt.year >= 2400
        if mask_be.any():
            s_dt.loc[mask_be] = s_dt.loc[mask_be] - pd.DateOffset(years=543)
        return s_dt

    s_str = series.astype(str).str.strip()
    s_str = s_str.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaT": np.nan, "0000-00-00": np.nan})

    def _conv_be_str(val):
        if pd.isna(val):
            return val
        txt = str(val).strip()
        # แปลงปี พ.ศ. 24xx - 26xx ให้เป็น ค.ศ. ก่อนแปลงเป็น datetime เพื่อป้องกันปัญหาปีอธิกสุรทิน
        m_iso = re.match(r"^(2[4-6]\d{2})([-/]\d{1,2}[-/]\d{1,2}.*)$", txt)
        if m_iso:
            return f"{int(m_iso.group(1)) - 543}{m_iso.group(2)}"
        m_dmy = re.match(r"^(\d{1,2}[-/]\d{1,2}[-/])(2[4-6]\d{2})(.*)$", txt)
        if m_dmy:
            return f"{m_dmy.group(1)}{int(m_dmy.group(2)) - 543}{m_dmy.group(3)}"
        return txt

    s_ce = s_str.map(_conv_be_str)
    parsed = pd.to_datetime(s_ce, errors="coerce", dayfirst=True)
    mask_be2 = parsed.dt.year >= 2400
    if mask_be2.any():
        parsed.loc[mask_be2] = parsed.loc[mask_be2] - pd.DateOffset(years=543)
    return parsed


def process_raw_dds_upload(uploaded_files, base_weekly_df=None):
    """
    อ่านเฉพาะ 8 คอลัมน์ที่อนุญาตในหน่วยความจำ -> คลีนข้อมูล -> รวมยอดรายสัปดาห์ -> ทำลาย viewcode ทันที
    """
    allowed_set = set(ALLOWED_DDS_COLUMNS)
    raw_frames = []

    for uf in uploaded_files:
        fname_low = uf.name.lower()
        if fname_low.endswith(".csv"):
            df_part = None
            for enc in ["utf-8-sig", "utf-8", "cp874", "tis-620"]:
                try:
                    uf.seek(0)
                    df_part = pd.read_csv(
                        uf,
                        encoding=enc,
                        usecols=lambda c: str(c).strip() in allowed_set,
                        dtype=str
                    )
                    break
                except Exception:
                    continue
            if df_part is not None and not df_part.empty:
                df_part.columns = [str(c).strip() for c in df_part.columns]
                raw_frames.append(df_part)
        else:
            uf.seek(0)
            sheets_dict = pd.read_excel(
                uf,
                sheet_name=None,
                usecols=lambda c: str(c).strip() in allowed_set,
                dtype=str
            )
            for _, df_sh in sheets_dict.items():
                if df_sh is not None and not df_sh.empty:
                    df_sh.columns = [str(c).strip() for c in df_sh.columns]
                    raw_frames.append(df_sh)

    if not raw_frames:
        raise ValueError("ไม่พบคอลัมน์มาตรฐาน DDS ในไฟล์ที่อัปโหลด (กรุณาตรวจสอบว่ามีคอลัมน์ epidem_report_group_code และวันที่เริ่มมีอาการ)")

    df_raw = pd.concat(raw_frames, ignore_index=True)
    # บังคับโครงสร้างให้เหลือเฉพาะ 8 คอลัมน์ที่อนุญาต 100%
    df_raw = df_raw.reindex(columns=ALLOWED_DDS_COLUMNS)
    total_rows_read = len(df_raw)

    # 1. ตัดรายการซ้ำด้วย viewcode (ถ้ามี)
    vc = df_raw["viewcode"].fillna("").astype(str).str.strip()
    has_vc = vc != ""
    if has_vc.any():
        df_with_vc = df_raw[has_vc].drop_duplicates(subset=["viewcode"], keep="first")
        df_no_vc = df_raw[~has_vc]
        df_clean = pd.concat([df_with_vc, df_no_vc], ignore_index=True)
    else:
        df_clean = df_raw.copy()

    # 2. จัดมาตรฐานรหัสจังหวัดและกรองเฉพาะ 5 จังหวัด เขตสุขภาพที่ 2
    df_clean["province_code"] = df_clean["รหัสจังหวัดขณะป่วย"].apply(normalize_pcode)
    prov_txt = df_clean["จังหวัดขณะป่วย"].fillna("").astype(str).str.replace("จังหวัด", "", regex=False).str.replace("จ.", "", regex=False).str.strip()
    missing_pcode = df_clean["province_code"] == ""
    df_clean.loc[missing_pcode, "province_code"] = prov_txt[missing_pcode].map(PROV_NAME_TO_CODE).fillna("")

    df_clean = df_clean[df_clean["province_code"].isin(ODPC2_PROVINCES.keys())].copy()
    df_clean["province_name"] = df_clean["province_code"].map(ODPC2_PROVINCES)

    # 3. จัดมาตรฐานรหัสกลุ่มโรคเฝ้าระวัง
    df_clean["disease_code"] = df_clean["epidem_report_group_code"].apply(normalize_dcode)
    df_clean = df_clean[df_clean["disease_code"] != ""].copy()
    if RESTRICT_TO_CORE_DISEASES:
        df_clean = df_clean[df_clean["disease_code"].isin(CORE_DISEASE_CODES)].copy()

    dname_raw = df_clean["epidem_report_group_name"].fillna("").astype(str).str.strip()
    df_clean["disease_name"] = np.where(
        (dname_raw != "") & (dname_raw.str.lower() != "nan"),
        dname_raw,
        df_clean["disease_code"].map(DEFAULT_DISEASE_NAMES).fillna("กลุ่มโรค " + df_clean["disease_code"])
    )

    # 4. จัดการวันที่อ้างอิง (วันที่เริ่มมีอาการ -> วันที่เริ่มรักษา -> วันที่วินิจฉัยโรค)
    dt_onset = parse_thai_date_series(df_clean["วันที่เริ่มมีอาการ"])
    dt_treat = parse_thai_date_series(df_clean["วันที่เริ่มรักษา"])
    dt_diag = parse_thai_date_series(df_clean["วันที่วินิจฉัยโรค"])

    event_dt = dt_onset.fillna(dt_treat).fillna(dt_diag)
    df_clean["event_dt"] = event_dt
    df_clean = df_clean[
        df_clean["event_dt"].notna() &
        (df_clean["event_dt"] >= pd.Timestamp("2020-01-01")) &
        (df_clean["event_dt"] <= pd.Timestamp.today() + pd.Timedelta(days=7))
    ].copy()

    if df_clean.empty:
        raise ValueError("ไม่พบรายการที่อยู่ในพื้นที่ 5 จังหวัด สคร.2 และมีวันที่ถูกต้อง")

    valid_cases_count = len(df_clean)

    # แปลงเป็นสัปดาห์ทางระบาดวิทยา (วันอาทิตย์ - วันเสาร์)
    df_clean["week_start"] = df_clean["event_dt"].dt.to_period("W-SAT").dt.start_time

    # 5. ยุบรวมข้อมูลเชิงสถิติ (ทำลาย viewcode ทิ้ง 100% ในขั้นตอนนี้)
    agg = (
        df_clean.groupby(["province_code", "province_name", "disease_code", "disease_name", "week_start"], as_index=False)
        .size()
        .rename(columns={"size": "cases"})
    )

    # ลบตัวแปรระดับรายบุคคลออกจากหน่วยความจำทันที
    del df_raw, df_clean

    # 6. ผสานกับฐานข้อมูลสัปดาห์เดิม (หากมี) และเติมสัปดาห์ที่ไม่มีผู้ป่วยให้ครบ (0 ราย)
    if base_weekly_df is not None and not base_weekly_df.empty:
        old_w = base_weekly_df.copy()
        old_w["week_start"] = pd.to_datetime(old_w["week_start"])
        old_w["province_code"] = old_w["province_code"].astype(str).str.strip()
        old_w["disease_code"] = old_w["disease_code"].apply(normalize_dcode)
        combined = pd.concat([old_w[["province_code", "province_name", "disease_code", "disease_name", "week_start", "cases"]], agg], ignore_index=True)
        # หากสัปดาห์ใดมีข้อมูลใหม่ให้อัปเดตทับด้วยยอดล่าสุด
        combined = combined.drop_duplicates(subset=["province_code", "disease_code", "week_start"], keep="last")
    else:
        combined = agg

    min_wk = combined["week_start"].min()
    max_wk = combined["week_start"].max()
    all_weeks = pd.date_range(min_wk, max_wk, freq="W-SUN")

    pairs = combined[["province_code", "province_name", "disease_code", "disease_name"]].drop_duplicates(
        subset=["province_code", "disease_code"], keep="last"
    )
    grid = pairs.assign(_k=1).merge(pd.DataFrame({"week_start": all_weeks, "_k": 1}), on="_k").drop(columns="_k")

    final_weekly = grid.merge(
        combined[["province_code", "disease_code", "week_start", "cases"]],
        on=["province_code", "disease_code", "week_start"],
        how="left"
    )
    final_weekly["cases"] = final_weekly["cases"].fillna(0).astype(int)

    prelim_start = max_wk - pd.Timedelta(weeks=1)
    final_weekly["data_status"] = np.where(
        final_weekly["week_start"] >= prelim_start,
        "เบื้องต้น (รอยอดนิ่ง)",
        "ครบถ้วน"
    )
    final_weekly = final_weekly.sort_values(["province_code", "disease_code", "week_start"]).reset_index(drop=True)

    stats_info = {
        "rows_read": total_rows_read,
        "valid_cases": valid_cases_count,
        "max_week": max_wk.strftime("%Y-%m-%d"),
        "num_diseases": int(final_weekly["disease_code"].nunique())
    }
    return final_weekly, stats_info


# =========================================================
# 3. โหลดข้อมูลและคำนวณพยากรณ์ (รองรับ Real-Time & Static)
# =========================================================
def load_files():
    latest_json = None
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            latest_json = json.load(f)

    # ลำดับการดึงข้อมูล: 1. ข้อมูลอัปโหลดสดใน Session -> 2. ฐานข้อมูล PostgreSQL -> 3. ไฟล์ CSV บน GitHub
    if "live_weekly_df" in st.session_state and st.session_state["live_weekly_df"] is not None:
        weekly = st.session_state["live_weekly_df"].copy()
    else:
        db_weekly = load_weekly_agg_from_db()
        if db_weekly is not None and not db_weekly.empty:
            weekly = db_weekly
        elif os.path.exists(WEEKLY_FILE):
            weekly = pd.read_csv(WEEKLY_FILE, dtype={"province_code": str, "disease_code": str})
        else:
            st.warning("ยังไม่มีข้อมูลตั้งต้น กรุณาอัปโหลดไฟล์ข้อมูลจากระบบ DDS ที่แถบเมนูด้านซ้าย")
            st.stop()

    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    weekly["province_code"] = weekly["province_code"].astype(str).str.strip()
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
# 4. โหลดข้อมูลหลัก + แถบควบคุมด้านซ้าย (พร้อมช่องอัปโหลด DDS สด)
# =========================================================
latest_json, weekly_df = load_files()

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

    with st.expander("อัปเดตข้อมูลดิบ DDS (Real-time)", expanded=False):
        render_html("""
        <div class="pdpa-pill">
            <b>PDPA Safe 100%:</b> ระบบอ่านเฉพาะ 8 คอลัมน์ที่จำเป็นใน RAM และทำลายรหัส <code>viewcode</code> ทันทีเมื่อรวมยอด ไม่เก็บข้อมูลส่วนบุคคล
        </div>
        <div style="background:#fffbeb; border:1px solid #fde68a; color:#92400e; font-size:11px; padding:7px 10px; border-radius:10px; margin-bottom:8px; line-height:1.5;">
            * <b>ข้อเสนอแนะ :</b> อัปโหลดทีละ <b>2–3 เดือน (ต่อช่วง)</b> / อัพไฟล์ <code>.csv</code> จะใช้ RAM น้อยกว่า <code>.xlsx</code> เพื่อให้ใช้งานได้อย่างต่อเนื่อง
        </div>
        """)
        uploaded_dds = st.file_uploader(
            "วางไฟล์ Excel/CSV จากระบบ DDS",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_dds:
            sig = tuple((f.name, f.size) for f in uploaded_dds)
            if st.session_state.get("last_dds_sig") != sig:
                try:
                    with st.spinner("กำลังคัดกรอง 8 คอลัมน์และคำนวณพยากรณ์สด..."):
                        new_weekly, up_stats = process_raw_dds_upload(uploaded_dds, base_weekly_df=weekly_df)
                        save_weekly_agg_store(new_weekly)
                        st.session_state["live_weekly_df"] = new_weekly
                        st.session_state["last_dds_sig"] = sig
                        st.session_state["last_dds_stats"] = up_stats
                        st.session_state["force_latest_date"] = True
                        st.cache_data.clear()
                        st.rerun()
                except Exception as e:
                    st.error(f"ไม่สามารถนำเข้าไฟล์ได้: {e}")

        if "last_dds_stats" in st.session_state:
            st_i = st.session_state["last_dds_stats"]
            st.success(f"อัปเดตสดแล้ว ({st_i['valid_cases']:,} ราย • สัปดาห์ล่าสุด {st_i['max_week']})")

    min_date = min(pd.to_datetime("2023-11-23").date(), weekly_df["week_start"].min().date())
    max_data_dt = weekly_df["week_start"].max().date()
    max_date = max(pd.Timestamp.today().date(), max_data_dt)

    if st.session_state.pop("force_latest_date", False):
        default_date = max_data_dt
        st.session_state["sidebar_cutoff_date"] = max_data_dt
    elif "live_weekly_df" in st.session_state and st.session_state["live_weekly_df"] is not None:
        default_date = max_data_dt
    elif latest_json is not None:
        latest_cutoff = pd.to_datetime(latest_json["meta"]["cutoff_date"]).date()
        default_date = min(max(latest_cutoff, min_date), max_date)
    else:
        default_date = max_data_dt

    render_html('<div class="sb-sec-tag">1. ใช้ข้อมูลถึงวันที่ (ทำนายสัปดาห์หน้า)</div>')
    picked_date = st.date_input(
        "วันตัดข้อมูล",
        value=default_date, min_value=min_date, max_value=max_date,
        key="sidebar_cutoff_date",
        label_visibility="collapsed"
    )
    picked_str = picked_date.strftime("%Y-%m-%d")

    use_static_json = (
        latest_json is not None
        and "live_weekly_df" not in st.session_state
        and picked_str == latest_json["meta"]["cutoff_date"]
    )
    payload = latest_json if use_static_json else predict_for_date(weekly_df, picked_str)
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
            <div><b>อัปเดตหรือกรองข้อมูล:</b> อัปโหลดไฟล์ DDS สด หรือเลือกวันที่และพื้นที่</div>
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
# 5. แบนเนอร์หัวเรื่องผู้บริหาร + แถบสลับหน้า
# =========================================================
status_chip_cls = "echip-warn" if "เบื้องต้น" in str(meta["data_status"]) else "echip-glass"
live_badge_txt = "LIVE DDS • PDPA Protected" if "live_weekly_df" in st.session_state else "DDS Weekly Surveillance"

render_html(f"""
<div class="exec-hero">
    <div class="exec-org-bar">
        <div class="exec-org-left">
            <span class="exec-live-dot"></span>
            <span class="exec-org-title">สำนักงานป้องกันควบคุมโรคที่ 2 จังหวัดพิษณุโลก</span>
            <span class="exec-org-dept">กรมควบคุมโรค • กลุ่มระบาดวิทยาและตอบโต้ภาวะฉุกเฉินทางสาธารณสุข</span>
        </div>
        <span class="exec-sys-tag">{live_badge_txt}</span>
    </div>
    <div class="exec-main-title">ระบบชี้เป้าโรคที่ควรเฝ้าระวังสัปดาห์หน้าจากข้อมูล DDS — {area_title}</div>
    <div class="exec-chips">
        <span class="echip-gold">สัปดาห์เป้าหมาย: {meta['target_week']}</span>
        <span class="echip-glass">ขอบเขตโรค: <b>{disease_scope_text}</b></span>
        <span class="echip-glass">ตัดข้อมูลถึง: <b>{meta['cutoff_date']}</b></span>
        <span class="echip-glass">รหัสรอบ: <b>{meta['run_id']}</b></span>
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
    diff_txt = f"+{diff_total:,} ราย จากสัปดาห์ล่าสุด" if diff_total >= 0 else f"{diff_total:,} ราย จากสัปดาห์ล่าสุด"

    if top_alerts:
        t1 = top_alerts[0]
        top_risk_name = f"{t1['disease_name']}"
        prov_prefix = f"จ.{t1['province_name']} • " if is_all_prov else ""
        top_risk_sub = f"{prov_prefix}คาด {to_int(t1['forecast']):,} ราย (เกณฑ์ {to_int(t1['threshold']):,})"
    else:
        top_risk_name, top_risk_sub = "ปกติทุกโรค", "ไม่พบรายการเกินเกณฑ์ P75"

    if is_all_prov:
        flagged_provs_cnt = len({d["province_code"] for d in flagged_items})
        kpi1_val = f"พบเตือน {flagged_provs_cnt} จังหวัด" if flagged_provs_cnt > 0 else "ปกติทุกจังหวัด"
        kpi1_sub = f"เกินเกณฑ์ {len(flagged_items)} จาก {len(view_items)} รายการ"
    else:
        kpi1_val = f"เกินเกณฑ์ {len(flagged_items)} โรค" if len(flagged_items) > 0 else "ปกติทุกโรค"
        kpi1_sub = f"เฝ้าระวังทั้งหมด {len(view_items)} โรค"

    k1_cls = "kpi-rose" if len(flagged_items) > 0 else "kpi-emerald"
    k1_ic = "kicon-rose" if len(flagged_items) > 0 else "kicon-emerald"
    k1_val_color = "#be123c" if len(flagged_items) > 0 else "#047857"

    render_html(f"""
    <div class="kpi-row">
        <div class="kpi-tile {k1_cls}">
            <div class="kpi-icon-box {k1_ic}">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/>
                </svg>
            </div>
            <div class="kpi-body">
                <div class="kpi-lbl">พื้นที่และโรคที่ต้องเฝ้าระวัง</div>
                <div class="kpi-val" style="color:{k1_val_color};">{kpi1_val}</div>
                <div class="kpi-sub">{kpi1_sub}</div>
            </div>
        </div>
        <div class="kpi-tile kpi-amber">
            <div class="kpi-icon-box kicon-amber">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
            </div>
            <div class="kpi-body">
                <div class="kpi-lbl">ต้องระวังเป็นอันดับที่ 1</div>
                <div class="kpi-val" style="font-size:17.5px; color:#b45309;">{top_risk_name}</div>
                <div class="kpi-sub">{top_risk_sub}</div>
            </div>
        </div>
        <div class="kpi-tile kpi-emerald">
            <div class="kpi-icon-box kicon-emerald">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>
                </svg>
            </div>
            <div class="kpi-body">
                <div class="kpi-lbl">ยอดคาดการณ์รวมสัปดาห์หน้า</div>
                <div class="kpi-val">{total_fc:,} ราย</div>
                <div class="kpi-sub">{diff_txt}</div>
            </div>
        </div>
        <div class="kpi-tile kpi-sky">
            <div class="kpi-icon-box kicon-sky">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>
                </svg>
            </div>
            <div class="kpi-body">
                <div class="kpi-lbl">ยอดผู้ป่วยจริงรวม (สัปดาห์ล่าสุด)</div>
                <div class="kpi-val">{total_latest:,} ราย</div>
                <div class="kpi-sub">ข้อมูล ณ วันที่ {meta['cutoff_date']}</div>
            </div>
        </div>
    </div>
    """)

    if is_all_prov:
        prov_cards_html = []
        for p in selected_provinces_objs:
            p_items = [d for d in p["all_diseases"] if d["disease_code"] in selected_disease_codes]
            p_flagged = sorted([d for d in p_items if d["is_flagged"]], key=risk_sort_key, reverse=True)
            if p_flagged:
                top_p = p_flagged[0]
                card_cls = "prov-mini-card prov-mini-alert"
                badge_html = f'<span class="pbadge-alert">เตือน {len(p_flagged)} โรค</span>'
                sub_txt = f"สูงสุด: {top_p['disease_name']}"
            else:
                card_cls = "prov-mini-card prov-mini-ok"
                badge_html = '<span class="pbadge-ok">ปกติ</span>'
                sub_txt = "ต่ำกว่าเกณฑ์ทุกรายการ"
            prov_cards_html.append(f"""
            <div class="{card_cls}">
                <div>
                    <div class="prov-mini-name">จ.{p['province_name']}</div>
                    <div class="prov-mini-sub">{sub_txt}</div>
                </div>
                {badge_html}
            </div>
            """)
        render_html(f'<div class="prov-strip">{"".join(prov_cards_html)}</div>')

    col_chart, col_action = st.columns([1.48, 1.0], gap="medium")

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
                y_names = [f"จ.{d['province_name']} - {d['disease_name']}" for d in chart_items]
            elif is_all_prov:
                y_names = [f"จ.{d['province_name']} ({d['province_code']})" for d in chart_items]
            else:
                y_names = [f"{d['disease_name']} ({d['disease_code']})" for d in chart_items]

            fc_vals = [to_int(d["forecast"]) for d in chart_items]
            th_vals = [to_int(d["threshold"]) for d in chart_items]
            fc_colors = ["#e11d48" if d["is_flagged"] else "#0284c7" for d in chart_items]
            fc_texts = [f" {to_int(d['forecast']):,} ราย (เกินเกณฑ์)" if d["is_flagged"] else f" {to_int(d['forecast']):,} ราย" for d in chart_items]
            th_texts = [f" {to_int(d['threshold']):,} ราย" for d in chart_items]

            fig_ov = go.Figure()
            fig_ov.add_trace(go.Bar(
                y=y_names, x=th_vals, name="เกณฑ์เตือนปกติ (P75)", orientation="h",
                marker=dict(color="#cbd5e1"), text=th_texts, textposition="outside",
                textfont=dict(size=11.5, color="#64748b"), hovertemplate="%{y}<br>เกณฑ์เตือน: %{x:,} ราย<extra></extra>"
            ))
            fig_ov.add_trace(go.Bar(
                y=y_names, x=fc_vals, name="ยอดคาดการณ์สัปดาห์หน้า", orientation="h",
                marker=dict(color=fc_colors), text=fc_texts, textposition="outside",
                textfont=dict(size=12, color="#0f172a"), hovertemplate="%{y}<br>ยอดคาดสัปดาห์หน้า: %{x:,} ราย<extra></extra>"
            ))

            max_x = max(fc_vals + th_vals + [1])
            fig_ov.update_layout(
                barmode="group", bargap=0.28, bargroupgap=0.12, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, font=dict(size=12)),
                font=dict(family="Google Sans, Noto Sans Thai, sans-serif", size=12, color="#334155"),
                margin=dict(l=4, r=24, t=36, b=16), height=max(220, 72 + len(chart_items) * 56)
            )
            fig_ov.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False, range=[0, max_x * 1.48], nticks=5, tickformat=",d", title_text="จำนวนผู้ป่วย (ราย)")
            fig_ov.update_yaxes(showgrid=False, automargin=True)
            st.plotly_chart(fig_ov, use_container_width=True)

    with col_action:
        if not top_alerts:
            render_html(f"""
            <div class="safe-box">
                <b>สัปดาห์เป้าหมาย {meta['target_week']} ยังไม่มีสัญญาณเตือนภัย</b><br>
                ยอดคาดการณ์ของ <b>{disease_scope_text}</b> ใน <b>{area_title}</b> ยังอยู่ต่ำกว่าเกณฑ์เตือนปกติทั้งหมด สามารถติดตามตามรอบปกติได้
            </div>
            """)
        else:
            badge_classes = ["r-tag-1", "r-tag-2", "r-tag-3"]
            items_html = []
            for i, item in enumerate(top_alerts):
                fc_i, th_i, lt_i = to_int(item["forecast"]), to_int(item["threshold"]), to_int(item["latest_cases"])
                lo_i, hi_i, diff = to_int(item["pi_low"]), to_int(item["pi_high"]), fc_i - th_i
                pt = get_prov_theme(item["province_code"])
                prov_pill = f'<span class="prov-tag" style="background:{pt["bg"]}; color:{pt["color"]}; border:1px solid {pt["border"]}; margin-right:5px; font-size:11px; padding:1px 7px;">จ.{item["province_name"]}</span>' if is_all_prov else ""
                b_cls = badge_classes[min(i, 2)]

                items_html.append(f"""
                <div class="alert-item">
                    <div class="alert-item-top">
                        <div>{prov_pill}<span class="alert-item-title">{item['disease_name']}</span></div>
                        <span class="{b_cls}">อันดับ #{i+1}</span>
                    </div>
                    <div class="alert-meta-line">
                        คาดสัปดาห์หน้า <b style="color:#be123c;">{fc_i:,} ราย</b> <span style="color:#be123c; font-weight:600;">(+{diff:,})</span> • เกณฑ์ <b>{th_i:,}</b> • ล่าสุด <b>{lt_i:,}</b>
                    </div>
                    <div class="alert-ci-sub">ช่วงคาดการณ์ 95%: {lo_i:,}–{hi_i:,} ราย</div>
                </div>
                """)

            render_html(f"""
            <div class="alert-box">
                <div class="alert-box-hdr">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/>
                        <line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                    <span>ลำดับพื้นที่และโรคที่ควรเฝ้าระวังก่อน (Top 3)</span>
                </div>
                {"".join(items_html)}
            </div>
            """)

    render_html("""
    <div class="sec-hdr">
        <div class="sec-bar"></div>
        <div class="sec-txt">ตารางสรุปตัวเลขทั้งหมด (คลิกหัวคอลัมน์เพื่อเรียงลำดับ หรือค้นหาได้ทันที)</div>
    </div>
    """)

    table_records = []
    for rank_idx, d in enumerate(view_items, start=1):
        fc_i, th_i, lt_i = to_int(d["forecast"]), to_int(d["threshold"]), to_int(d["latest_cases"])

        row_dict = {
            "อันดับ": rank_idx,
            "จังหวัด": f"จ.{d['province_name']}",
            "กลุ่มโรคเฝ้าระวัง": f"{d['disease_name']} ({d['disease_code']})",
            "ยอดล่าสุด": lt_i,
            "คาดสัปดาห์หน้า": fc_i,
            "เกณฑ์เตือน (P75)": th_i,
            "สถานะ": "🔴 เฝ้าระวัง" if d["is_flagged"] else "🟢 ปกติ"
        }
        if not is_all_prov:
            del row_dict["จังหวัด"]
        table_records.append(row_dict)

    df_summary_grid = pd.DataFrame(table_records)

    col_cfg = {
        "อันดับ": st.column_config.NumberColumn("อันดับ", width="small", format="%d"),
        "จังหวัด": st.column_config.TextColumn("จังหวัด", width="small"),
        "กลุ่มโรคเฝ้าระวัง": st.column_config.TextColumn("กลุ่มโรคเฝ้าระวัง", width="medium"),
        "ยอดล่าสุด": st.column_config.NumberColumn("ยอดล่าสุด", format="%d ราย"),
        "คาดสัปดาห์หน้า": st.column_config.NumberColumn("คาดสัปดาห์หน้า", format="%d ราย"),
        "เกณฑ์เตือน (P75)": st.column_config.NumberColumn("เกณฑ์เตือน (P75)", format="%d ราย"),
        "สถานะ": st.column_config.TextColumn("สถานะ", width="small"),
    }

    st.dataframe(
        df_summary_grid,
        column_config=col_cfg,
        use_container_width=True,
        hide_index=True,
        height=min(480, 48 + len(df_summary_grid) * 38)
    )

# =========================================================
# หน้า 2: วิเคราะห์แนวโน้มรายโรค
# =========================================================
elif active_page == NAV_PAGES[1]:
    if len(view_items) == 1:
        d = view_items[0]
    else:
        with st.container(border=True):
            c_p, c_d = st.columns(2, gap="medium")

            avail_provs = {}
            for it in view_items:
                if it["province_code"] not in avail_provs:
                    avail_provs[it["province_code"]] = f"จ.{it['province_name']}"

            if len(avail_provs) > 1:
                prov_lbl_to_code = {v: k for k, v in avail_provs.items()}
                sel_p_lbl = c_p.selectbox("เลือกจังหวัดที่ต้องการดูแนวโน้ม", list(prov_lbl_to_code.keys()), key="tab2_prov")
                pick_p_code = prov_lbl_to_code[sel_p_lbl]
            else:
                pick_p_code = list(avail_provs.keys())[0]
                c_p.selectbox("จังหวัดที่เลือก", list(avail_provs.values()), disabled=True, key="tab2_prov_fixed")

            prov_items = [it for it in view_items if it["province_code"] == pick_p_code]
            d_map = {}
            for it in prov_items:
                st_tag = "เกินเกณฑ์เตือน" if it["is_flagged"] else "ปกติ"
                d_map[f"{it['disease_name']} ({it['disease_code']}) • {st_tag}"] = it

            sel_d_lbl = c_d.selectbox("เลือกกลุ่มโรคที่ต้องการดูแนวโน้ม", list(d_map.keys()), key="tab2_dis")
            d = d_map[sel_d_lbl]

    pt_sel = get_prov_theme(d["province_code"])
    fc_i, th_i, lt_i, lo_i, hi_i = to_int(d["forecast"]), to_int(d["threshold"]), to_int(d["latest_cases"]), to_int(d["pi_low"]), to_int(d["pi_high"])
    diff_wk = fc_i - lt_i

    st_cls = "kpi-rose" if d["is_flagged"] else "kpi-emerald"
    st_ic = "kicon-rose" if d["is_flagged"] else "kicon-emerald"
    st_val_c = "#be123c" if d["is_flagged"] else "#047857"
    st_txt = "เกินเกณฑ์เตือน" if d["is_flagged"] else "ต่ำกว่าเกณฑ์ (ปกติ)"

    render_html(f"""
    <div class="kpi-row">
        <div class="kpi-tile kpi-sky">
            <div class="kpi-icon-box kicon-sky">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            </div>
            <div class="kpi-body">
                <div class="kpi-lbl">ยอดจริงสัปดาห์ล่าสุด</div>
                <div class="kpi-val">{lt_i:,} ราย</div>
                <div class="kpi-sub">ฐานข้อมูลปัจจุบัน</div>
            </div>
        </div>
        <div class="kpi-tile {st_cls}">
            <div class="kpi-icon-box {st_ic}">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
            </div>
            <div class="kpi-body">
                <div class="kpi-lbl">พยากรณ์สัปดาห์เป้าหมาย ({d['target_week']})</div>
                <div class="kpi-val" style="color:{st_val_c};">{fc_i:,} ราย</div>
                <div class="kpi-sub">ช่วง 95%: {lo_i:,} – {hi_i:,} ราย</div>
            </div>
        </div>
        <div class="kpi-tile kpi-amber">
            <div class="kpi-icon-box kicon-amber">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="9" x2="20" y2="9"/><line x1="4" y1="15" x2="20" y2="15"/><line x1="10" y1="3" x2="8" y2="21"/><line x1="16" y1="3" x2="14" y2="21"/></svg>
            </div>
            <div class="kpi-body">
                <div class="kpi-lbl">เกณฑ์เตือนรายโรค (P75)</div>
                <div class="kpi-val">{th_i:,} ราย</div>
                <div class="kpi-sub">เส้นมาตรฐานเฝ้าระวัง</div>
            </div>
        </div>
        <div class="kpi-tile {st_cls}">
            <div class="kpi-icon-box {st_ic}">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            </div>
            <div class="kpi-body">
                <div class="kpi-lbl">ผลการเทียบเกณฑ์</div>
                <div class="kpi-val" style="font-size:18px; color:{st_val_c};">{st_txt}</div>
                <div class="kpi-sub">{diff_wk:+,} ราย จากสัปดาห์ล่าสุด</div>
            </div>
        </div>
    </div>
    """)

    hist = pd.DataFrame(d["history"])
    hist["week_dt"] = pd.to_datetime(hist["week"])
    hist["cases_num"] = pd.to_numeric(hist["cases"], errors="coerce")
    target_dt, first_dt = pd.to_datetime(d["target_week"]), hist["week_dt"].iloc[0]

    with st.container(border=True):
        render_html(f"""
        <div class="card-hdr">
            <div>
                <div style="display:flex; align-items:center; flex-wrap:wrap; gap:8px;">
                    <span class="prov-tag" style="background:{pt_sel['bg']}; color:{pt_sel['color']}; border:1px solid {pt_sel['border']};">จ.{d['province_name']}</span>
                    <span class="card-h3">แนวโน้มผู้ป่วยรายสัปดาห์ (ย้อนหลัง 1 ปี) — {d['disease_name']}</span>
                </div>
                <div class="card-desc">แสดงรอบฤดูกาลย้อนหลัง 52 สัปดาห์ เทียบกับเส้นเกณฑ์เตือนและจุดพยากรณ์สัปดาห์หน้า</div>
            </div>
        </div>
        """)

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
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, font=dict(size=12.5)),
            font=dict(family="Google Sans, Noto Sans Thai, sans-serif", size=12.5, color="#64748b"),
            margin=dict(l=40, r=20, t=45, b=35), height=365
        )
        fig.update_xaxes(showgrid=False, tickformat="%b %Y", nticks=7, zeroline=False, automargin=True, range=[first_dt - pd.Timedelta(days=7), target_dt + pd.Timedelta(days=18)])
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False, nticks=6, tickformat=",d", automargin=True, range=[-max(0.3, y_max * 0.08), y_max * 1.22], title_text="จำนวนผู้ป่วย (ราย)")
        st.plotly_chart(fig, use_container_width=True, key="chart_52w")

        render_html("""
        <div class="legend-inline">
            <b style="color:#004d28;">วิธีอ่านกราฟ (ภาพรวม 1 ปี):</b><br>
            • <b style="color:#2563eb;">เส้นทึบสีน้ำเงิน:</b> ยอดผู้ป่วยจริงรายสัปดาห์ย้อนหลัง 52 สัปดาห์ ใช้ดูรอบฤดูกาลของโรค<br>
            • <b style="color:#d97706;">เส้นประสีเหลือง:</b> เกณฑ์เตือนปกติ (P75) หากยอดผู้ป่วยสูงเกินเส้นนี้ถือว่าสูงกว่าเกณฑ์ปกติของพื้นที่<br>
            • <b style="color:#be123c;">รูปดาวสีแดง (ขวาสุด):</b> ยอดคาดการณ์สัปดาห์หน้า หากดาวลอยอยู่เหนือเส้นประสีเหลือง แสดงว่าสัปดาห์หน้ามีแนวโน้มเกินเกณฑ์เตือน <i>(หมายเหตุ: แถบพื้นหลังสีเหลืองอ่อนช่วง 1–2 สัปดาห์ล่าสุด คือข้อมูลเบื้องต้นที่หน่วยบริการยังทยอยบันทึกเข้าระบบ)</i>
        </div>
        """)

    with st.container(border=True):
        hist_12 = hist.tail(12).copy()
        first_dt_12 = hist_12["week_dt"].iloc[0]

        if diff_wk > 0:
            trend_pill = f'<span style="background:#fef2f2; color:#b91c1c; border:1px solid #fecaca; padding:5px 13px; border-radius:999px; font-weight:700; font-size:12.5px;">สัปดาห์หน้าคาดว่าเพิ่มขึ้น +{diff_wk:,} ราย</span>'
        elif diff_wk < 0:
            trend_pill = f'<span style="background:#ecfdf5; color:#047857; border:1px solid #a7f3d0; padding:5px 13px; border-radius:999px; font-weight:700; font-size:12.5px;">สัปดาห์หน้าคาดว่าลดลง {diff_wk:,} ราย</span>'
        else:
            trend_pill = '<span style="background:#f1f5f9; color:#334155; border:1px solid #cbd5e1; padding:5px 13px; border-radius:999px; font-weight:700; font-size:12.5px;">สัปดาห์หน้าคาดว่าทรงตัว (0 ราย)</span>'

        render_html(f"""
        <div class="card-hdr">
            <div>
                <div style="display:flex; align-items:center; flex-wrap:wrap; gap:8px;">
                    <span class="prov-tag" style="background:{pt_sel['bg']}; color:{pt_sel['color']}; border:1px solid {pt_sel['border']};">จ.{d['province_name']}</span>
                    <span class="card-h3">ซูมแนวโน้มระยะสั้น (12 สัปดาห์ล่าสุด + พยากรณ์สัปดาห์หน้า) — {d['disease_name']}</span>
                </div>
                <div class="card-desc">แสดงตัวเลขรายสัปดาห์ช่วง 3 เดือนล่าสุด เพื่อดูทิศทางขึ้น–ลงก่อนเข้าสู่สัปดาห์เป้าหมาย ({d['target_week']})</div>
            </div>
            {trend_pill}
        </div>
        """)

        fig_12w = go.Figure()
        prelim_12 = hist_12[hist_12["status"].str.contains("เบื้องต้น", na=False)]
        if not prelim_12.empty:
            fig_12w.add_vrect(x0=prelim_12["week_dt"].iloc[0] - pd.Timedelta(days=3), x1=prelim_12["week_dt"].iloc[-1] + pd.Timedelta(days=3), fillcolor="rgba(245, 158, 11, 0.12)", line_width=0)

        fig_12w.add_trace(go.Scatter(x=[first_dt_12, target_dt], y=[th_i, th_i], mode="lines", name=f"เกณฑ์เตือน ({th_i:,} ราย)", line=dict(color="#f59e0b", width=2.2, dash="dot"), hovertemplate=f"เกณฑ์เตือน: {th_i:,} ราย<extra></extra>"))

        pt_texts_12 = [f"{int(v):,}" if pd.notna(v) else "" for v in hist_12["cases_num"]]
        fig_12w.add_trace(go.Scatter(
            x=hist_12["week_dt"], y=hist_12["cases_num"],
            mode="lines+markers+text", text=pt_texts_12, textposition="top center",
            textfont=dict(size=11.5, color="#1d4ed8"),
            name=f"ยอดจริง จ.{d['province_name']}",
            line=dict(color="#2563eb", width=2.5),
            marker=dict(size=8, color="#2563eb", line=dict(color="#ffffff", width=1.5)),
            hovertemplate="สัปดาห์ %{x|%d %b %Y}<br>ยอดจริง: %{y:,.0f} ราย<extra></extra>"
        ))

        valid_12 = hist_12.dropna(subset=["cases_num"])
        if not valid_12.empty:
            fig_12w.add_trace(go.Scatter(x=[valid_12["week_dt"].iloc[-1], target_dt], y=[valid_12["cases_num"].iloc[-1], fc_i], mode="lines", line=dict(color="#f43f5e", width=2.4, dash="dash"), showlegend=False, hoverinfo="skip"))

        fig_12w.add_trace(go.Scatter(
            x=[target_dt], y=[fc_i], mode="markers+text",
            text=[f"  {fc_i:,} ราย"], textposition="middle right",
            textfont=dict(size=12.5, color="#be123c", family="Google Sans, Noto Sans Thai, sans-serif"),
            name=f"พยากรณ์สัปดาห์หน้า ({fc_i:,} ราย)",
            marker=dict(color="#f43f5e", size=14, symbol="star", line=dict(color="#ffffff", width=1)),
            hovertemplate=f"สัปดาห์เป้าหมาย {d['target_week']}<br>พยากรณ์: {fc_i:,} ราย<br>ช่วง 95%: {lo_i:,}–{hi_i:,} ราย<extra></extra>"
        ))

        y_max_12 = max(float(valid_12["cases_num"].max() if not valid_12.empty else 0), float(th_i), float(fc_i), 1.0)
        fig_12w.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, font=dict(size=12.5)),
            font=dict(family="Google Sans, Noto Sans Thai, sans-serif", size=12.5, color="#64748b"),
            margin=dict(l=40, r=40, t=45, b=35), height=355
        )
        fig_12w.update_xaxes(showgrid=False, tickformat="%d %b %y", nticks=7, zeroline=False, automargin=True, range=[first_dt_12 - pd.Timedelta(days=4), target_dt + pd.Timedelta(days=14)])
        fig_12w.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False, nticks=6, tickformat=",d", automargin=True, range=[-max(0.3, y_max_12 * 0.08), y_max_12 * 1.25], title_text="จำนวนผู้ป่วย (ราย)")
        st.plotly_chart(fig_12w, use_container_width=True, key="chart_12w")

        render_html("""
        <div class="legend-inline">
            <b style="color:#004d28;">วิธีอ่านกราฟ (ซูมระยะสั้น 3 เดือนล่าสุด):</b><br>
            • <b style="color:#2563eb;">ตัวเลขบนจุดสีน้ำเงิน:</b> จำนวนผู้ป่วยจริงในแต่ละสัปดาห์ช่วง 12 สัปดาห์ล่าสุด<br>
            • <b style="color:#be123c;">เส้นประสีแดงและรูปดาว:</b> แสดงทิศทางและความชันจากสัปดาห์ล่าสุดไปยังสัปดาห์เป้าหมาย เพื่อประเมินว่าสถานการณ์กำลัง <b style="color:#be123c;">ไต่ระดับเพิ่มขึ้น</b> <b style="color:#be123c;">ทรงตัว</b> หรือ <b style="color:#be123c;">เริ่มลดลง</b> เมื่อเทียบกับเส้นเกณฑ์เตือนสีเหลือง
        </div>
        """)

# =========================================================
# หน้า 3: บันทึกและติดตามผล + ตาราง Interactive Data Grid
# =========================================================
elif active_page == NAV_PAGES[2]:
    reg_df = load_registry(selected_prov_code, selected_disease_code)

    if reg_df.empty:
        st.info("ยังไม่มีรายการในทะเบียนตามจังหวัดและโรคที่เลือก")
    else:
        reg_df["month"] = reg_df["target_week"].astype(str).str.slice(0, 7)

        c_title, c_auto, c_month = st.columns([2.4, 1.35, 1.0], gap="small")
        with c_title:
            render_html(f"""
            <div class="sec-hdr" style="margin-top:0;">
                <div class="sec-bar"></div>
                <div class="sec-txt">สรุปตัวชี้วัดประเมินความแม่นยำรายเดือน — {area_title}</div>
            </div>
            """)
        with c_auto:
            if st.button("ดึงยอดจริงมาปิดยอดอัตโนมัติ", type="secondary", use_container_width=True):
                cnt = auto_settle_from_weekly(weekly_df)
                if cnt > 0:
                    st.session_state["auto_settle_msg"] = ("success", f"ดึงยอดจริงที่นิ่งแล้วมาปิดยอดอัตโนมัติสำเร็จ {cnt:,} รายการ")
                    st.rerun()
                else:
                    st.session_state["auto_settle_msg"] = ("info", "ยังไม่มีรายการใหม่ที่ครบกำหนดยอดนิ่ง (ข้อมูลต้องผ่านช่วงเบื้องต้น 2 สัปดาห์แล้ว หรืออัปโหลดไฟล์ DDS สัปดาห์ถัดไปเพิ่ม)")

        sel_m = c_month.selectbox("เลือกเดือนประเมิน", ["ทุกเดือน"] + sorted(reg_df["month"].unique().tolist(), reverse=True), label_visibility="collapsed")

        if "auto_settle_msg" in st.session_state:
            msg_type, msg_txt = st.session_state.pop("auto_settle_msg")
            if msg_type == "success":
                st.success(msg_txt)
            else:
                st.info(msg_txt)

        ev = reg_df if sel_m == "ทุกเดือน" else reg_df[reg_df["month"] == sel_m]
        ev["settled_num"] = pd.to_numeric(ev["settled_actual_cases"], errors="coerce")
        closed = ev[ev["settled_num"].notna()].copy()

        if closed.empty:
            m1_val, m2_val, m3_val, m4_val = "รอปิดยอดจริง", "รอปิดยอดจริง", "0 ครั้ง", "รอระบุเวลา"
            m3_note = "คำนวณเมื่อกรอกยอดจริง"
        else:
            flagged = closed["is_flagged"].astype(str).str.lower() == "true"
            actual_met = closed["settled_num"] >= pd.to_numeric(closed["frozen_threshold"], errors="coerce")
            tp, fp, fn = int((flagged & actual_met).sum()), int((flagged & ~actual_met).sum()), int((~flagged & actual_met).sum())
            hr_diff = ((pd.to_datetime(closed["check_start_time"], errors="coerce") - pd.to_datetime(closed["alert_sent_time"], errors="coerce")).dt.total_seconds() / 3600).dropna().clip(lower=0)
            avg_hr = hr_diff.mean()
            m1_val, m2_val = ratio_text(tp, tp + fp), ratio_text(tp, tp + fn)
            m3_val, m3_note = f"รวม {tp + fp} (เกิน {fp})", f"เตือนเกิน {fp} ครั้ง"
            m4_val = f"{avg_hr:.1f} ชม." if pd.notna(avg_hr) else "ยังไม่ระบุเวลา"

        render_html(f"""
        <div class="kpi-row">
            <div class="kpi-tile kpi-emerald">
                <div class="kpi-icon-box kicon-emerald">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
                </div>
                <div class="kpi-body">
                    <div class="kpi-lbl">1. ความแม่นของสัญญาณ (Precision)</div>
                    <div class="kpi-val">{m1_val}</div>
                    <div class="kpi-sub">เตือนแล้วเข้าเกณฑ์จริง</div>
                </div>
            </div>
            <div class="kpi-tile kpi-sky">
                <div class="kpi-icon-box kicon-sky">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                </div>
                <div class="kpi-body">
                    <div class="kpi-lbl">2. ความครอบคลุม (Recall)</div>
                    <div class="kpi-val">{m2_val}</div>
                    <div class="kpi-sub">จับเหตุการณ์พุ่งเกินเกณฑ์</div>
                </div>
            </div>
            <div class="kpi-tile kpi-amber">
                <div class="kpi-icon-box kicon-amber">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                </div>
                <div class="kpi-body">
                    <div class="kpi-lbl">3. ภาระการตรวจสอบ</div>
                    <div class="kpi-val">{m3_val}</div>
                    <div class="kpi-sub">{m3_note}</div>
                </div>
            </div>
            <div class="kpi-tile kpi-rose">
                <div class="kpi-icon-box kicon-rose">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                </div>
                <div class="kpi-body">
                    <div class="kpi-lbl">4. ความเร็วในการเริ่มตรวจ</div>
                    <div class="kpi-val">{m4_val}</div>
                    <div class="kpi-sub">จากเวลาส่งสัญญาณ</div>
                </div>
            </div>
        </div>
        """)

        with st.container(border=True):
            render_html("""
            <div class="card-hdr">
                <div>
                    <div class="card-h3">บันทึกผลการตรวจสอบภาคสนาม และปิดยอดผู้ป่วยจริง</div>
                    <div class="card-desc">เลือกรอบรายงานเพื่อบันทึกการลงสอบสวนโรคและอัปเดตยอดจริงเมื่อข้อมูลนิ่ง (หลังครบ 1–2 สัปดาห์)</div>
                </div>
            </div>
            """)
            reg_df["label"] = (
                reg_df["run_id"] + "  |  จ." + reg_df["province_name"].astype(str) +
                "  |  " + reg_df["disease_name"] + "  |  สัปดาห์ " + reg_df["target_week"]
            )
            rec = reg_df[reg_df["label"] == st.selectbox("เลือกรอบรายงานที่ต้องการบันทึก", reg_df["label"].tolist())].iloc[0]

            rec_status_txt = "🔴 ชี้เป้าเฝ้าระวัง" if str(rec["is_flagged"]).lower() == "true" else "🟢 ปกติ"
            render_html(f"""
            <div class="rec-bar">
                <span><b>สถานะรอบนี้:</b> {rec_status_txt}</span>
                <span><b>ยอดคาดการณ์เดิม:</b> {fmt_int(rec['frozen_forecast'])} ราย</span>
                <span><b>เกณฑ์เตือนเดิม:</b> {fmt_int(rec['frozen_threshold'])} ราย</span>
                <span><b>เวลาส่งสัญญาณ:</b> {clean_str(rec['alert_sent_time']) or '-'}</span>
            </div>
            """)

            init_date_str, init_time_str = split_check_datetime(rec["check_start_time"])
            if not init_date_str and not init_time_str:
                now_th = pd.Timestamp.now(tz="Asia/Bangkok")
                init_date_str = now_th.strftime("%d/%m/%Y")
                init_time_str = now_th.strftime("%H:%M")

            rec_uid = f"{rec['run_id']}_{rec['province_code']}_{rec['disease_code']}"

            with st.form("track_form", border=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div style="font-size:14px; font-weight:500; margin-bottom:4px;">1. เวลาเริ่มตรวจ (ใช้คิดความเร็ว)</div>', unsafe_allow_html=True)
                    d_col, t_col = st.columns([1.25, 1.0], gap="small")
                    in_date = d_col.text_input(
                        "วันที่เริ่มตรวจ",
                        value=init_date_str,
                        placeholder="วว/ดด/ปปปป",
                        key=f"in_d_{rec_uid}",
                        label_visibility="collapsed"
                    )
                    in_time = t_col.text_input(
                        "เวลาเริ่มตรวจ",
                        value=init_time_str,
                        placeholder="เช่น 11:30",
                        key=f"in_t_{rec_uid}",
                        label_visibility="collapsed"
                    )
                    st.markdown(
                        '<div style="font-size:12px; color:#047857; margin-top:-6px; margin-bottom:6px; line-height:1.45;">'
                        '* <b>ดึงวันและเวลาปัจจุบันให้อัตโนมัติแล้ว</b> (สามารถพิมพ์แก้ไขได้หากเริ่มตรวจคนละเวลา)'
                        '</div>',
                        unsafe_allow_html=True
                    )

                f_opts = ["", "พบกลุ่มก้อน", "พบการส่งย้อนหลัง", "ยังสรุปไม่ได้", "ปกติ"]
                cur_f = clean_str(rec["field_result"])
                in_field = c2.selectbox("2. ผลตรวจสอบภาคสนาม", f_opts, index=f_opts.index(cur_f) if cur_f in f_opts else 0)

                c3, c4 = st.columns(2)
                in_action = c3.text_input("3. การดำเนินการ", value=clean_str(rec["action_taken"]), placeholder="เช่น แจ้งทีม SRRT พื้นที่ตรวจสอบซ้ำ")
                raw_act = clean_str(rec["settled_actual_cases"])
                in_actual = c4.text_input("4. ยอดจริงนิ่งแล้ว (กรอกเมื่อครบ 1–2 สัปดาห์ หรือกดปุ่มดึงยอดอัตโนมัติด้านบน)", value=str(to_int(raw_act)) if raw_act != "" else "", placeholder="จำนวนเต็ม (เว้นว่างได้หากยอดยังไม่นิ่ง)")

                if st.form_submit_button("บันทึกผลและอัปเดตตัวชี้วัด", type="primary"):
                    try:
                        combined_check_time = parse_check_datetime(in_date, in_time)
                    except ValueError as ve:
                        st.error(str(ve))
                        st.stop()

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
                    update_registry(rec["run_id"], rec["province_code"], rec["disease_code"], rec["target_week"], combined_check_time, in_field, in_action.strip(), actual_val)
                    st.success(f"บันทึกเรียบร้อยแล้ว (เวลาเริ่มตรวจ: {combined_check_time})")
                    st.rerun()

        render_html(f"""
        <div class="sec-hdr">
            <div class="sec-bar"></div>
            <div class="sec-txt">ทะเบียนประวัติการติดตามผล ({len(reg_df):,} รายการ)</div>
        </div>
        """)

        c_flt, c_dl = st.columns([2.8, 1.1])
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
                label="ส่งออกทะเบียน (CSV)",
                data=csv_bytes,
                file_name=f"tracking_registry_{meta['cutoff_date']}.csv",
                mime="text/csv",
                use_container_width=True
            )

        if show_reg.empty:
            st.info("ไม่พบรายการตามตัวกรองที่เลือก")
        else:
            reg_grid_rows = []
            for _, r in show_reg.iterrows():
                act_s = clean_str(r["settled_actual_cases"])
                reg_grid_rows.append({
                    "สัปดาห์เป้าหมาย": str(r["target_week"]),
                    "จังหวัด": f"จ.{r['province_name']}",
                    "กลุ่มโรค": str(r["disease_name"]),
                    "ยอดคาดการณ์": to_int(r["frozen_forecast"]),
                    "เกณฑ์เตือน": to_int(r["frozen_threshold"]),
                    "สถานะ": "🔴 ชี้เป้า" if str(r["is_flagged"]).lower() == "true" else "🟢 ปกติ",
                    "ผลภาคสนาม": clean_str(r["field_result"]) or "-",
                    "การดำเนินการ": clean_str(r["action_taken"]) or "-",
                    "ยอดจริงนิ่งแล้ว": f"{fmt_int(act_s)} ราย" if act_s != "" else "รอยอดนิ่ง"
                })
            df_reg_grid = pd.DataFrame(reg_grid_rows)
            if not is_all_prov:
                df_reg_grid = df_reg_grid.drop(columns=["จังหวัด"])

            st.dataframe(
                df_reg_grid,
                column_config={
                    "ยอดคาดการณ์": st.column_config.NumberColumn("ยอดคาดการณ์", format="%d ราย"),
                    "เกณฑ์เตือน": st.column_config.NumberColumn("เกณฑ์เตือน", format="%d ราย"),
                },
                use_container_width=True,
                hide_index=True,
                height=min(480, 48 + len(df_reg_grid) * 38)
            )
