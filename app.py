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
# สไตล์ Modern SaaS / Public Health Web Application
# =========================================================
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anuphan:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

:root {
    --brand: #047857;
    --brand-dark: #065f46;
    --brand-light: #ecfdf5;
    --brand-border: #a7f3d0;
    --bg-app: #f6f8fa;
    --bg-card: #ffffff;
    --border-subtle: #e5e7eb;
    --border-medium: #d1d5db;
    --text-primary: #111827;
    --text-secondary: #4b5563;
    --text-muted: #6b7280;
    --shadow-sm: 0 1px 2px rgba(16, 24, 40, 0.04);
    --shadow-card: 0 1px 3px rgba(16, 24, 40, 0.04), 0 4px 12px rgba(16, 24, 40, 0.02);
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Anuphan', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 14px !important;
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background-color: var(--bg-app);
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
    height: 2rem !important;
}
#MainMenu, footer { visibility: hidden !important; }

.block-container {
    padding-top: 0.9rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

/* ================= SIDEBAR (MODERN APP RAIL) ================= */
[data-testid="stSidebar"] {
    background-color: var(--bg-card);
    border-right: 1px solid var(--border-subtle);
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

.sb-logo-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 16px;
    background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 6px 16px rgba(4, 120, 87, 0.16);
}
.sb-logo-badge {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.28);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    flex-shrink: 0;
}
.sb-logo-title {
    font-size: 15px;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.25;
}
.sb-logo-sub {
    font-size: 11.5px;
    color: #d1fae5;
    margin-top: 2px;
}

.sb-filter-label {
    font-size: 11.5px;
    font-weight: 700;
    color: var(--brand-dark);
    background: var(--brand-light);
    border: 1px solid var(--brand-border);
    border-radius: 8px;
    padding: 6px 11px;
    margin: 18px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 6px;
}

[data-testid="stSidebar"] .stCheckbox {
    padding: 4px 8px;
    border-radius: 8px;
    margin-bottom: -2px;
    transition: all 0.15s ease;
}
[data-testid="stSidebar"] .stCheckbox:hover {
    background-color: #f3f4f6;
}

.sb-guide-widget {
    background: #f9fafb;
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 15px 16px;
    margin-top: 20px;
}
.sb-guide-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 11px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
}
.sb-guide-item {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    margin-bottom: 10px;
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.55;
}
.sb-guide-item:last-child { margin-bottom: 0; }
.sb-guide-num {
    background: var(--brand);
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    min-width: 20px;
    height: 20px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 1px;
}

/* ================= TOP NAVBAR & PAGE HEADER ================= */
.web-navbar {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    box-shadow: var(--shadow-sm);
}
.web-nav-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.live-indicator {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 0 4px #d1fae5;
    flex-shrink: 0;
}
.web-org-title {
    font-size: 14px;
    font-weight: 700;
    color: var(--text-primary);
}
.web-org-sub {
    font-size: 12.5px;
    color: var(--text-muted);
    border-left: 1px solid var(--border-medium);
    padding-left: 12px;
}

.page-hero {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 22px 26px;
    margin: 12px 0 16px 0;
    box-shadow: var(--shadow-card);
    position: relative;
    overflow: hidden;
}
.page-hero::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 5px;
    background: linear-gradient(180deg, #047857 0%, #10b981 100%);
}
.hero-kicker {
    font-size: 11.5px;
    font-weight: 700;
    color: var(--brand);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.hero-heading {
    font-size: 21px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 14px 0;
    line-height: 1.35;
}
.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}
.pill-brand {
    background: var(--brand);
    color: #ffffff;
    padding: 5px 13px;
    border-radius: 8px;
    font-size: 12.5px;
    font-weight: 600;
}
.pill-gray {
    background: #f9fafb;
    color: var(--text-secondary);
    border: 1px solid var(--border-subtle);
    padding: 5px 12px;
    border-radius: 8px;
    font-size: 12.5px;
}
.pill-amber {
    background: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
    padding: 5px 12px;
    border-radius: 8px;
    font-size: 12.5px;
    font-weight: 600;
}

/* ================= SEGMENTED PILL NAVIGATION TABS ================= */
.stTabs [data-baseweb="tab-list"] {
    background-color: #e5e7eb;
    padding: 5px;
    border-radius: 12px;
    border: none;
    gap: 6px;
    margin-bottom: 20px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border-radius: 9px !important;
    padding: 9px 18px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    transition: all 0.18s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background-color: rgba(255, 255, 255, 0.5) !important;
}
.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    color: var(--brand) !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 3px rgba(16, 24, 40, 0.08) !important;
}
.stTabs [data-baseweb="tab-highlight"], div[data-baseweb="tab-highlight"] {
    display: none !important;
}

/* ================= MODERN CARD CONTAINERS (FIX CLIPPING) ================= */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px !important;
    padding: 22px 26px !important;
    box-shadow: var(--shadow-card) !important;
    margin-bottom: 16px !important;
    overflow: visible !important;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 12px;
    padding-bottom: 14px;
    margin-bottom: 10px;
    border-bottom: 1px solid #f3f4f6;
}
.card-title-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}
.card-h3 {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    line-height: 1.4;
}
.card-desc {
    font-size: 13px;
    color: var(--text-muted);
    margin: 4px 0 0 0;
    line-height: 1.5;
}

.trend-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 8px;
    font-size: 12.5px;
    font-weight: 700;
    white-space: nowrap;
}
.trend-up { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
.trend-down { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
.trend-flat { background: #f3f4f6; color: #374151; border: 1px solid #e5e7eb; }

.guide-callout {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 12px;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.75;
}
.guide-callout-hdr {
    font-weight: 700;
    color: var(--brand-dark);
    margin-bottom: 4px;
    font-size: 13px;
}
.guide-item {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-top: 3px;
}
.dot-blue { width: 8px; height: 8px; border-radius: 50%; background: #2563eb; display: inline-block; flex-shrink: 0; }
.dot-amber { width: 8px; height: 8px; border-radius: 50%; background: #d97706; display: inline-block; flex-shrink: 0; }
.dot-red { width: 8px; height: 8px; border-radius: 50%; background: #be123c; display: inline-block; flex-shrink: 0; }

/* ================= KPI METRIC CARDS ================= */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 16px;
    margin: 4px 0 20px 0;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 18px 22px;
    box-shadow: var(--shadow-card);
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3.5px;
    background: var(--brand);
}
.kpi-card-danger::after {
    background: #e11d48;
}
.kpi-title {
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-muted);
}
.kpi-value {
    font-size: 24px;
    font-weight: 800;
    color: var(--text-primary);
    margin: 6px 0 4px 0;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.01em;
}
.kpi-caption {
    font-size: 12px;
    color: var(--text-muted);
}

/* ================= PRIORITY ALERT CARD ================= */
.priority-card {
    background: var(--bg-card);
    border: 1px solid #fecaca;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: var(--shadow-card);
    margin-bottom: 16px;
}
.priority-hdr {
    background: #fef2f2;
    border-bottom: 1px solid #fecaca;
    padding: 15px 22px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.priority-hdr-text {
    font-size: 15px;
    font-weight: 700;
    color: #991b1b;
}
.priority-row {
    padding: 16px 22px;
    border-bottom: 1px solid #f3f4f6;
}
.priority-row:last-child { border-bottom: none; }
.priority-row:first-of-type { background: #fffcfc; }
.priority-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}
.priority-name {
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
}
.rank-badge-1 { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 8px; }
.rank-badge-2 { background: #ffedd5; color: #c2410c; border: 1px solid #fdba74; font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 8px; }
.rank-badge-3 { background: #fef9c3; color: #a16207; border: 1px solid #fde047; font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 8px; }
.priority-metrics {
    margin: 0;
    padding-left: 18px;
    color: var(--text-secondary);
    font-size: 13.5px;
    line-height: 1.7;
}

.normal-status-card {
    background: var(--bg-card);
    border: 1px solid #a7f3d0;
    border-left: 5px solid var(--brand);
    border-radius: 16px;
    padding: 20px 24px;
    color: #065f46;
    font-size: 14px;
    line-height: 1.7;
    box-shadow: var(--shadow-card);
    margin-bottom: 16px;
}

/* ================= MODERN DATA TABLE ================= */
.section-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 22px 0 12px 0;
}
.section-bar {
    width: 4px;
    height: 20px;
    background: var(--brand);
    border-radius: 4px;
}
.section-text {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
}

.table-surface {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    overflow-x: auto;
    margin-bottom: 20px;
    box-shadow: var(--shadow-card);
}
.modern-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13.5px;
}
.modern-table th {
    background: #f9fafb;
    color: var(--text-secondary);
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.02em;
    padding: 14px 18px;
    text-align: left;
    border-bottom: 1px solid var(--border-subtle);
    white-space: nowrap;
}
.modern-table td {
    padding: 13px 18px;
    border-bottom: 1px solid #f3f4f6;
    color: var(--text-primary);
    vertical-align: middle;
}
.modern-table tbody tr:last-child td {
    border-bottom: none;
}
.modern-table tbody tr:hover {
    background-color: #f9fafb;
}
.num { text-align: right !important; font-variant-numeric: tabular-nums; white-space: nowrap; }
.ctr { text-align: center !important; }
.muted { font-size: 12px; color: var(--text-muted); }

.prov-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 7px;
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
}
.status-alert {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
}
.status-normal {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
}

.record-summary-bar {
    background: #f9fafb;
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 10px 0 16px 0;
    font-size: 13px;
    color: var(--text-secondary);
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    align-items: center;
}

/* ================= FORM CONTROLS & BUTTONS ================= */
[data-baseweb="select"] > div, [data-baseweb="input"] > div {
    border-radius: 10px !important;
    border-color: var(--border-medium) !important;
}

.stButton > button[kind="primary"], .stFormSubmitButton > button {
    background-color: var(--brand) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.5rem !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(4, 120, 87, 0.2) !important;
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {
    background-color: var(--brand-dark) !important;
}

.stButton > button[kind="secondary"] {
    background-color: var(--bg-card) !important;
    color: var(--brand) !important;
    border: 1px solid var(--brand-border) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1rem !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton > button[kind="secondary"]:hover {
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
# 3. โหลดข้อมูลหลัก + แถบหน่วยงานด้านบนและปุ่มระเบียบวิธี
# =========================================================
latest_json, weekly_df = load_files()

top_c1, top_c2 = st.columns([4.1, 1.15], gap="small")
with top_c1:
    render_html("""
    <div class="web-navbar">
        <div class="web-nav-left">
            <span class="live-indicator"></span>
            <span class="web-org-title">สำนักงานป้องกันควบคุมโรคที่ 2 จังหวัดพิษณุโลก</span>
            <span class="web-org-sub">กรมควบคุมโรค • กลุ่มระบาดวิทยาและตอบโต้ภาวะฉุกเฉินทางสาธารณสุข</span>
        </div>
    </div>
    """)
with top_c2:
    if st.button("ระเบียบวิธี / เกณฑ์ P75", use_container_width=True):
        show_methodology_modal()

# =========================================================
# 4. แถบควบคุมด้านซ้าย (Left App Rail)
# =========================================================
min_date = pd.to_datetime("2023-11-23").date()
max_data_dt = weekly_df["week_start"].max().date()
max_date = max(pd.Timestamp.today().date(), max_data_dt)
latest_cutoff = pd.to_datetime(latest_json["meta"]["cutoff_date"]).date()
default_date = min(max(latest_cutoff, min_date), max_date)

with st.sidebar:
    render_html("""
    <div class="sb-logo-card">
        <div class="sb-logo-badge">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="M9 12l2 2 4-4"/>
            </svg>
        </div>
        <div>
            <div class="sb-logo-title">สคร.2 จ.พิษณุโลก</div>
            <div class="sb-logo-sub">ระบบเฝ้าระวังและชี้เป้าโรค (DDS)</div>
        </div>
    </div>
    """)

    render_html('<div class="sb-filter-label">1. ใช้ข้อมูลถึงวันที่ (ทำนายสัปดาห์หน้า)</div>')
    picked_date = st.date_input(
        "วันตัดข้อมูล",
        value=default_date, min_value=min_date, max_value=max_date,
        label_visibility="collapsed"
    )
    picked_str = picked_date.strftime("%Y-%m-%d")

    payload = latest_json if picked_str == latest_json["meta"]["cutoff_date"] else predict_for_date(weekly_df, picked_str)
    sync_to_registry(payload)

    render_html('<div class="sb-filter-label">2. เลือกจังหวัด (สคร.2)</div>')
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

    render_html('<div class="sb-filter-label">3. เลือกกลุ่มโรคเฝ้าระวัง</div>')
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
    <div class="sb-guide-widget">
        <div class="sb-guide-title">ขั้นตอนการใช้งานระบบ</div>
        <div class="sb-guide-item">
            <div class="sb-guide-num">1</div>
            <div><b>กำหนดตัวกรอง:</b> เลือกวันตัดข้อมูล จังหวัด และโรคที่ต้องการติดตาม</div>
        </div>
        <div class="sb-guide-item">
            <div class="sb-guide-num">2</div>
            <div><b>ตรวจสัญญาณเตือน:</b> ดูอันดับเสี่ยงในแท็บ 1 และกราฟแนวโน้มในแท็บ 2</div>
        </div>
        <div class="sb-guide-item">
            <div class="sb-guide-num">3</div>
            <div><b>บันทึกผลภาคสนาม:</b> กรอกผลตรวจสอบและปิดยอดจริงในแท็บ 3</div>
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
# 5. ส่วนหัวสรุปสถานะ (Page Hero) + Segmented Pill Tabs
# =========================================================
status_pill_cls = "pill-amber" if "เบื้องต้น" in str(meta["data_status"]) else "pill-gray"
render_html(f"""
<div class="page-hero">
    <div class="hero-kicker">WEEKLY EPIDEMIOLOGICAL SURVEILLANCE & EARLY WARNING</div>
    <div class="hero-heading">ระบบชี้เป้าโรคที่ควรเฝ้าระวังสัปดาห์หน้าจากข้อมูล DDS — {area_title}</div>
    <div class="hero-pills">
        <span class="pill-brand">สัปดาห์เป้าหมาย: {meta['target_week']}</span>
        <span class="pill-gray">ขอบเขตโรค: <b>{disease_scope_text}</b></span>
        <span class="pill-gray">ตัดข้อมูลถึง: <b>{meta['cutoff_date']}</b></span>
        <span class="pill-gray">รหัสรอบ: <b>{meta['run_id']}</b></span>
        <span class="{status_pill_cls}">สถานะข้อมูล: {meta['data_status']}</span>
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

    has_alert = len(flagged_items) > 0
    alert_color = "#be123c" if has_alert else "#047857"
    kpi_alert_cls = "kpi-card kpi-card-danger" if has_alert else "kpi-card"

    if is_all_prov:
        flagged_provs_cnt = len({d["province_code"] for d in flagged_items})
        kpi1_val = f"พบเตือน {flagged_provs_cnt} จังหวัด" if flagged_provs_cnt > 0 else "ปกติทุกจังหวัด"
        kpi1_note = f"เกินเกณฑ์รวม {len(flagged_items)} จาก {len(view_items)} รายการ"
    else:
        kpi1_val = f"เกินเกณฑ์ {len(flagged_items)} โรค" if has_alert else "ปกติทุกโรค"
        kpi1_note = f"จากทั้งหมด {len(view_items)} โรคที่เฝ้าระวัง"

    render_html(f"""
    <div class="kpi-grid">
        <div class="{kpi_alert_cls}">
            <div class="kpi-title">พื้นที่และโรคที่ต้องเฝ้าระวัง</div>
            <div class="kpi-value" style="color:{alert_color};">{kpi1_val}</div>
            <div class="kpi-caption">{kpi1_note}</div>
        </div>
        <div class="{kpi_alert_cls}">
            <div class="kpi-title">ต้องระวังเป็นอันดับที่ 1</div>
            <div class="kpi-value" style="font-size:18.5px; color:{alert_color};">{top_risk_name}</div>
            <div class="kpi-caption">{top_risk_note}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">ยอดคาดการณ์รวมสัปดาห์หน้า</div>
            <div class="kpi-value">{total_fc:,} ราย</div>
            <div class="kpi-caption">{diff_total_txt}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">ยอดผู้ป่วยจริงรวม (สัปดาห์ล่าสุด)</div>
            <div class="kpi-value">{total_latest:,} ราย</div>
            <div class="kpi-caption">ข้อมูล ณ วันตัดรอบ {meta['cutoff_date']}</div>
        </div>
    </div>
    """)

    col_chart, col_action = st.columns([1.28, 1.0], gap="medium")

    with col_chart:
        with st.container(border=True):
            chart_slice = view_items[:6]
            chart_sub_extra = f" (แสดง {len(chart_slice)} อันดับเสี่ยงสูงสุด)" if len(view_items) > 6 else ""
            render_html(f"""
            <div class="card-header">
                <div>
                    <div class="card-h3">เปรียบเทียบยอดคาดการณ์สัปดาห์หน้ากับเกณฑ์เตือน</div>
                    <div class="card-desc">แท่งสีแดง = เกินเกณฑ์เตือนปกติ (P75) • แท่งสีฟ้า = อยู่ในเกณฑ์ปกติ{chart_sub_extra}</div>
                </div>
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
                textfont=dict(size=12, color="#111827"), hovertemplate="%{y}<br>ยอดคาดสัปดาห์หน้า: %{x:,} ราย<extra></extra>"
            ))

            max_x = max(fc_vals + th_vals + [1])
            fig_ov.update_layout(
                barmode="group", bargap=0.28, bargroupgap=0.12, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, font=dict(size=12)),
                font=dict(family="Plus Jakarta Sans, Anuphan, sans-serif", size=12, color="#374151"),
                margin=dict(l=4, r=24, t=36, b=16), height=max(220, 72 + len(chart_items) * 56)
            )
            fig_ov.update_xaxes(showgrid=True, gridcolor="#f3f4f6", zeroline=False, range=[0, max_x * 1.48], nticks=5, tickformat=",d", title_text="จำนวนผู้ป่วย (ราย)")
            fig_ov.update_yaxes(showgrid=False, automargin=True)
            st.plotly_chart(fig_ov, use_container_width=True)

    with col_action:
        if not top_alerts:
            render_html(f"""
            <div class="normal-status-card">
                <b>สัปดาห์เป้าหมาย {meta['target_week']} ยังไม่มีสัญญาณเตือนภัย</b><br>
                ยอดคาดการณ์ของ <b>{disease_scope_text}</b> ใน <b>{area_title}</b> ยังอยู่ต่ำกว่าเกณฑ์เตือนปกติทั้งหมด สามารถติดตามตามรอบปกติได้
            </div>
            """)
        else:
            badge_classes = ["rank-badge-1", "rank-badge-2", "rank-badge-3"]
            items_html = []
            for i, item in enumerate(top_alerts):
                fc_i, th_i, lt_i = to_int(item["forecast"]), to_int(item["threshold"]), to_int(item["latest_cases"])
                lo_i, hi_i, diff = to_int(item["pi_low"]), to_int(item["pi_high"]), fc_i - th_i
                pt = get_prov_theme(item["province_code"])
                prov_pill = f'<span class="prov-tag" style="background:{pt["bg"]}; color:{pt["color"]}; border:1px solid {pt["border"]}; margin-right:6px;">จ.{item["province_name"]}</span>' if is_all_prov else ""
                b_cls = badge_classes[min(i, 2)]

                items_html.append(f"""
                <div class="priority-row">
                    <div class="priority-top">
                        <div>{prov_pill}<span class="priority-name">{item['disease_name']}</span></div>
                        <span class="{b_cls}">อันดับ #{i+1}</span>
                    </div>
                    <ul class="priority-metrics">
                        <li>ยอดคาดสัปดาห์หน้า: <b style="color:#be123c;">{fc_i:,} ราย</b> <span class="muted">(ช่วง 95%: {lo_i:,}–{hi_i:,} ราย)</span></li>
                        <li>เกณฑ์เตือนปกติ: <b>{th_i:,} ราย</b> <span style="color:#be123c; font-weight:600;">(เกินเกณฑ์ +{diff:,} ราย)</span></li>
                        <li>ยอดจริงสัปดาห์ล่าสุด: <b>{lt_i:,} ราย</b></li>
                    </ul>
                </div>
                """)

            render_html(f"""
            <div class="priority-card">
                <div class="priority-hdr">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#b91c1c" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/>
                        <line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                    <span class="priority-hdr-text">ลำดับพื้นที่และโรคที่ควรเฝ้าระวังก่อน (Top 3)</span>
                </div>
                {"".join(items_html)}
            </div>
            """)

    render_html("""
    <div class="section-heading">
        <div class="section-bar"></div>
        <div class="section-text">ตารางสรุปตัวเลขทั้งหมด (เรียงลำดับความเสี่ยงจากมากไปน้อย)</div>
    </div>
    """)

    rows_html = []
    for rank_idx, d in enumerate(view_items, start=1):
        fc_i, th_i, lt_i = to_int(d["forecast"]), to_int(d["threshold"]), to_int(d["latest_cases"])
        badge = '<span class="status-alert">เฝ้าระวัง</span>' if d["is_flagged"] else '<span class="status-normal">ปกติ</span>'

        pt = get_prov_theme(d["province_code"])
        prov_td = f'<td><span class="prov-tag" style="background:{pt["bg"]}; color:{pt["color"]}; border:1px solid {pt["border"]};">จ.{d["province_name"]}</span></td>' if is_all_prov else ""

        rows_html.append(f"""
        <tr>
            <td class="ctr"><b>{rank_idx}</b></td>
            {prov_td}
            <td><b>{d['disease_name']}</b> <span class="muted">({d['disease_code']})</span></td>
            <td class="num">{lt_i:,}</td>
            <td class="num"><b>{fc_i:,}</b></td>
            <td class="num">{th_i:,}</td>
            <td class="ctr">{badge}</td>
        </tr>
        """)
    prov_th = "<th>จังหวัด</th>" if is_all_prov else ""
    render_html(f"""
    <div class="table-surface"><table class="modern-table">
        <thead>
            <tr>
                <th class="ctr">อันดับ</th>{prov_th}<th>กลุ่มโรคเฝ้าระวัง</th>
                <th class="num">ยอดล่าสุด</th><th class="num">คาดสัปดาห์หน้า</th>
                <th class="num">เกณฑ์เตือน (P75)</th><th class="ctr">สถานะ</th>
            </tr>
        </thead>
        <tbody>{"".join(rows_html)}</tbody>
    </table></div>
    """)

# =========================================================
# แท็บ 2: กราฟแนวโน้มรายโรค (Modern Web App Card Layout)
# =========================================================
with tab2:
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
    st_color = "#be123c" if d["is_flagged"] else "#047857"
    st_label = "เกินเกณฑ์เตือน (ควรเฝ้าระวัง)" if d["is_flagged"] else "ต่ำกว่าเกณฑ์ (ปกติ)"
    kpi_t2_cls = "kpi-card kpi-card-danger" if d["is_flagged"] else "kpi-card"

    render_html(f"""
    <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-title">ยอดจริงสัปดาห์ล่าสุด</div><div class="kpi-value">{lt_i:,} ราย</div></div>
        <div class="{kpi_t2_cls}"><div class="kpi-title">พยากรณ์สัปดาห์เป้าหมาย ({d['target_week']})</div><div class="kpi-value" style="color:{st_color};">{fc_i:,} ราย</div><div class="kpi-caption">ช่วง 95%: {lo_i:,} – {hi_i:,} ราย</div></div>
        <div class="kpi-card"><div class="kpi-title">เกณฑ์เตือนรายโรค (P75)</div><div class="kpi-value">{th_i:,} ราย</div></div>
        <div class="{kpi_t2_cls}"><div class="kpi-title">ผลการเทียบเกณฑ์</div><div class="kpi-value" style="font-size:16.5px; color:{st_color}; margin-top:9px;">{st_label}</div></div>
    </div>
    """)

    hist = pd.DataFrame(d["history"])
    hist["week_dt"] = pd.to_datetime(hist["week"])
    hist["cases_num"] = pd.to_numeric(hist["cases"], errors="coerce")
    target_dt, first_dt = pd.to_datetime(d["target_week"]), hist["week_dt"].iloc[0]

    # การ์ดกราฟที่ 1: ย้อนหลัง 1 ปี (52 สัปดาห์)
    with st.container(border=True):
        render_html(f"""
        <div class="card-header">
            <div>
                <div class="card-title-row">
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
            font=dict(family="Plus Jakarta Sans, Anuphan, sans-serif", size=12.5, color="#6b7280"),
            margin=dict(l=40, r=20, t=45, b=35), height=370
        )
        fig.update_xaxes(showgrid=False, tickformat="%b %Y", nticks=7, zeroline=False, automargin=True, range=[first_dt - pd.Timedelta(days=7), target_dt + pd.Timedelta(days=18)])
        fig.update_yaxes(showgrid=True, gridcolor="#f3f4f6", zeroline=False, nticks=6, tickformat=",d", automargin=True, range=[-max(0.3, y_max * 0.08), y_max * 1.22], title_text="จำนวนผู้ป่วย (ราย)")
        st.plotly_chart(fig, use_container_width=True, key="chart_52w")

        render_html("""
        <div class="guide-callout">
            <div class="guide-callout-hdr">วิธีอ่านกราฟ (ภาพรวม 1 ปี)</div>
            <div class="guide-item"><span class="dot-blue"></span><span><b style="color:#2563eb;">เส้นทึบสีน้ำเงิน:</b> ยอดผู้ป่วยจริงรายสัปดาห์ย้อนหลัง 52 สัปดาห์ ใช้ดูรอบฤดูกาลของโรค</span></div>
            <div class="guide-item"><span class="dot-amber"></span><span><b style="color:#d97706;">เส้นประสีเหลือง:</b> เกณฑ์เตือนปกติ (P75) หากยอดผู้ป่วยสูงเกินเส้นนี้ถือว่าสูงกว่าเกณฑ์ปกติของพื้นที่</span></div>
            <div class="guide-item"><span class="dot-red"></span><span><b style="color:#be123c;">รูปดาวสีแดง (ขวาสุด):</b> ยอดคาดการณ์สัปดาห์หน้า หากดาวลอยอยู่เหนือเส้นประสีเหลือง แสดงว่าสัปดาห์หน้ามีแนวโน้มเกินเกณฑ์เตือน <i>(หมายเหตุ: แถบพื้นหลังสีเหลืองอ่อนช่วง 1–2 สัปดาห์ล่าสุด คือข้อมูลเบื้องต้นที่หน่วยบริการยังทยอยบันทึกเข้าระบบ)</i></span></div>
        </div>
        """)

    # การ์ดกราฟที่ 2: ซูมแนวโน้ม 12 สัปดาห์ล่าสุด + พยากรณ์สัปดาห์หน้า
    with st.container(border=True):
        hist_12 = hist.tail(12).copy()
        first_dt_12 = hist_12["week_dt"].iloc[0]

        diff_wk = fc_i - lt_i
        if diff_wk > 0:
            trend_html = f"""
            <span class="trend-chip trend-up">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
                <span>สัปดาห์หน้าคาดว่าเพิ่มขึ้น +{diff_wk:,} ราย</span>
            </span>
            """
        elif diff_wk < 0:
            trend_html = f"""
            <span class="trend-chip trend-down">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>
                <span>สัปดาห์หน้าคาดว่าลดลง {diff_wk:,} ราย</span>
            </span>
            """
        else:
            trend_html = """
            <span class="trend-chip trend-flat">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
                <span>สัปดาห์หน้าคาดว่าทรงตัว (0 ราย)</span>
            </span>
            """

        render_html(f"""
        <div class="card-header">
            <div>
                <div class="card-title-row">
                    <span class="prov-tag" style="background:{pt_sel['bg']}; color:{pt_sel['color']}; border:1px solid {pt_sel['border']};">จ.{d['province_name']}</span>
                    <span class="card-h3">ซูมแนวโน้มระยะสั้น (12 สัปดาห์ล่าสุด + พยากรณ์สัปดาห์หน้า) — {d['disease_name']}</span>
                </div>
                <div class="card-desc">แสดงตัวเลขรายสัปดาห์ช่วง 3 เดือนล่าสุด เพื่อดูทิศทางขึ้น–ลงก่อนเข้าสู่สัปดาห์เป้าหมาย ({d['target_week']})</div>
            </div>
            {trend_html}
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
            textfont=dict(size=12.5, color="#be123c", family="Plus Jakarta Sans, Anuphan, sans-serif"),
            name=f"พยากรณ์สัปดาห์หน้า ({fc_i:,} ราย)",
            marker=dict(color="#f43f5e", size=14, symbol="star", line=dict(color="#ffffff", width=1)),
            hovertemplate=f"สัปดาห์เป้าหมาย {d['target_week']}<br>พยากรณ์: {fc_i:,} ราย<br>ช่วง 95%: {lo_i:,}–{hi_i:,} ราย<extra></extra>"
        ))

        y_max_12 = max(float(valid_12["cases_num"].max() if not valid_12.empty else 0), float(th_i), float(fc_i), 1.0)
        fig_12w.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, font=dict(size=12.5)),
            font=dict(family="Plus Jakarta Sans, Anuphan, sans-serif", size=12.5, color="#6b7280"),
            margin=dict(l=40, r=40, t=45, b=35), height=355
        )
        fig_12w.update_xaxes(showgrid=False, tickformat="%d %b %y", nticks=7, zeroline=False, automargin=True, range=[first_dt_12 - pd.Timedelta(days=4), target_dt + pd.Timedelta(days=14)])
        fig_12w.update_yaxes(showgrid=True, gridcolor="#f3f4f6", zeroline=False, nticks=6, tickformat=",d", automargin=True, range=[-max(0.3, y_max_12 * 0.08), y_max_12 * 1.25], title_text="จำนวนผู้ป่วย (ราย)")
        st.plotly_chart(fig_12w, use_container_width=True, key="chart_12w")

        render_html("""
        <div class="guide-callout">
            <div class="guide-callout-hdr">วิธีอ่านกราฟ (ซูมระยะสั้น 3 เดือนล่าสุด)</div>
            <div class="guide-item"><span class="dot-blue"></span><span><b style="color:#2563eb;">ตัวเลขบนจุดสีน้ำเงิน:</b> จำนวนผู้ป่วยจริงในแต่ละสัปดาห์ช่วง 12 สัปดาห์ล่าสุด</span></div>
            <div class="guide-item"><span class="dot-red"></span><span><b style="color:#be123c;">เส้นประสีแดงและรูปดาว:</b> แสดงทิศทางและความชันจากสัปดาห์ล่าสุดไปยังสัปดาห์เป้าหมาย เพื่อประเมินว่าสถานการณ์กำลัง <b style="color:#be123c;">ไต่ระดับเพิ่มขึ้น</b> <b style="color:#be123c;">ทรงตัว</b> หรือ <b style="color:#be123c;">เริ่มลดลง</b> เมื่อเทียบกับเส้นเกณฑ์เตือนสีเหลือง</span></div>
        </div>
        """)

# =========================================================
# แท็บ 3: บันทึกและติดตามผล + ตารางทะเบียนข้อมูล
# =========================================================
with tab3:
    reg_df = load_registry(selected_prov_code, selected_disease_code)

    if reg_df.empty:
        st.info("ยังไม่มีรายการในทะเบียนตามจังหวัดและโรคที่เลือก")
    else:
        reg_df["month"] = reg_df["target_week"].astype(str).str.slice(0, 7)

        c_title, c_month = st.columns([3.2, 1])
        with c_title:
            render_html(f"""
            <div class="section-heading" style="margin-top:0;">
                <div class="section-bar"></div>
                <div class="section-text">สรุปตัวชี้วัดประเมินความแม่นยำรายเดือน — {area_title}</div>
            </div>
            """)
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
        <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-title">1. ความแม่นของสัญญาณ (Precision)</div><div class="kpi-value">{m1_val}</div><div class="kpi-caption">เตือนแล้วเข้าเกณฑ์จริง ÷ เตือนทั้งหมด</div></div>
            <div class="kpi-card"><div class="kpi-title">2. ความครอบคลุม (Recall)</div><div class="kpi-value">{m2_val}</div><div class="kpi-caption">จับเหตุการณ์ที่พุ่งเกินเกณฑ์จริงได้</div></div>
            <div class="kpi-card"><div class="kpi-title">3. ภาระการตรวจสอบ</div><div class="kpi-value">{m3_val}</div><div class="kpi-caption">{m3_note}</div></div>
            <div class="kpi-card"><div class="kpi-title">4. ความเร็วในการเริ่มตรวจ</div><div class="kpi-value">{m4_val}</div><div class="kpi-caption">เฉลี่ยจากเวลาส่งสัญญาณถึงเริ่มตรวจ</div></div>
        </div>
        """)

        with st.container(border=True):
            render_html("""
            <div class="card-header">
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

            rec_status = '<span class="status-alert">ชี้เป้าเฝ้าระวัง</span>' if str(rec["is_flagged"]).lower() == "true" else '<span class="status-normal">ปกติ</span>'
            render_html(f"""
            <div class="record-summary-bar">
                <span><b>สถานะรอบนี้:</b> {rec_status}</span>
                <span><b>ยอดคาดการณ์เดิม:</b> {fmt_int(rec['frozen_forecast'])} ราย</span>
                <span><b>เกณฑ์เตือนเดิม:</b> {fmt_int(rec['frozen_threshold'])} ราย</span>
                <span><b>เวลาส่งสัญญาณ:</b> {clean_str(rec['alert_sent_time']) or '-'}</span>
            </div>
            """)

            with st.form("track_form", border=False):
                c1, c2 = st.columns(2)
                in_time = c1.text_input("1. เวลาเริ่มตรวจ (ใช้คิดความเร็ว)", value=clean_str(rec["check_start_time"]), placeholder="เช่น 2026-11-24 11:30")
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

        render_html(f"""
        <div class="section-heading">
            <div class="section-bar"></div>
            <div class="section-text">ทะเบียนประวัติการติดตามผล ({len(reg_df):,} รายการ)</div>
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
            reg_rows = []
            for _, r in show_reg.iterrows():
                f_badge = '<span class="status-alert">ชี้เป้า</span>' if str(r["is_flagged"]).lower() == "true" else '<span class="status-normal">ปกติ</span>'
                act_s = clean_str(r["settled_actual_cases"])
                act_txt = f"<b>{fmt_int(act_s)} ราย</b>" if act_s != "" else '<span class="muted">รอยอดนิ่ง</span>'
                pt = get_prov_theme(r["province_code"])
                prov_td = f'<td><span class="prov-tag" style="background:{pt["bg"]}; color:{pt["color"]}; border:1px solid {pt["border"]};">จ.{r["province_name"]}</span></td>' if is_all_prov else ""
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
            <div class="table-surface"><table class="modern-table">
                <thead>
                    <tr>
                        <th>สัปดาห์เป้าหมาย</th>{prov_th}<th>กลุ่มโรค</th>
                        <th class="num">คาด / เกณฑ์</th><th class="ctr">สถานะ</th>
                        <th>ผลภาคสนาม</th><th>การดำเนินการ</th><th class="num">ยอดจริงนิ่งแล้ว</th>
                    </tr>
                </thead>
                <tbody>{"".join(reg_rows)}</tbody>
            </table></div>
            """)
