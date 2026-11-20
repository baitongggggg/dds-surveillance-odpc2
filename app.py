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
    page_title="แดชบอร์ดชี้เป้าโรคเฝ้าระวัง DDS",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


def render_html(html_str):
    """ล้างช่องว่างหน้าบรรทัด กันไม่ให้ Streamlit แสดงเป็นโค้ด HTML ดิบ"""
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


def calc_threshold(train_series):
    """คำนวณเกณฑ์เตือนเป็นจำนวนเต็ม และป้องกันเกณฑ์เป็น 0"""
    th = int(round(float(np.percentile(train_series, 75))))
    if th == 0:
        if train_series.max() > 0:
            th = max(1, int(round(float(np.percentile(train_series, 90)))))
        else:
            th = 1
    return th


# แนวทางเตรียมรับมือเฉพาะโรคสำหรับทีมระบาดวิทยา
DISEASE_ACTION_GUIDE = {
    "02": "ตรวจเช็กการกระจุกตัวรายอำเภอ/โรงเรียน/โรงงาน ตรวจสอบแหล่งน้ำดื่ม-อาหาร และแจ้งทีม SRRT พื้นที่เฝ้าระวังสุขาภิบาล",
    "15": "เฝ้าระวังการระบาดเป็นกลุ่มก้อนในโรงเรียน/สถานที่แออัด ตรวจผล Lab ยืนยันสายพันธุ์ และเร่งสื่อสารป้องกันในกลุ่มเสี่ยง",
    "31": "ติดตามกลุ่มเสี่ยงสูง (เด็กเล็กและผู้สูงอายุ) เช็กสัดส่วนผู้ป่วยอาการรุนแรง/ใส่เครื่องช่วยหายใจ และดูความเชื่อมโยงกับไข้หวัดใหญ่"
}

DISEASE_NAMES = {
    "02": "โรคอุจจาระร่วงเฉียบพลัน",
    "15": "โรคไข้หวัดใหญ่",
    "31": "โรคปอดอักเสบ"
}

WEEKLY_FILE = "weekly_summary.csv"
JSON_FILE = "dashboard_data.json"
BACKTEST_FILE = "backtest_report.csv"
CSV_REGISTRY = "tracking_registry.csv"

# =========================================================
# สไตล์มินิมอล คลีน อ่านสบายตา
# =========================================================
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans Thai', sans-serif !important;
    font-size: 13.5px !important;
    color: #1e293b;
}
.stApp { background-color: #f8fafc; }
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1140px !important;
}

h1 { font-size: 1.35rem !important; font-weight: 700 !important; color: #0f172a !important; }
h2 { font-size: 1.05rem !important; font-weight: 600 !important; color: #0f172a !important; }
h3 { font-size: 0.95rem !important; font-weight: 600 !important; color: #334155 !important; }

.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 14px;
}
.stTabs [data-baseweb="tab"] {
    padding: 8px 4px;
    font-size: 13.5px !important;
    font-weight: 500;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    color: #0284c7 !important;
    font-weight: 600 !important;
}

.top-header {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #0284c7;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 14px;
}
.top-title { font-size: 19px; font-weight: 700; color: #0f172a; margin: 0 0 6px 0; }
.meta-line { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.meta-tag {
    background: #f1f5f9; color: #475569; padding: 3px 10px;
    border-radius: 999px; font-size: 12px;
}
.meta-tag-warn {
    background: #fffbeb; color: #b45309; border: 1px solid #fde68a;
    padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 500;
}

.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
    margin: 4px 0 16px 0;
}
.kpi-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 13px 16px;
}
.kpi-lbl { font-size: 12px; color: #64748b; }
.kpi-val { font-size: 20px; font-weight: 700; color: #0f172a; margin-top: 3px; }
.kpi-note { font-size: 11.5px; color: #64748b; margin-top: 3px; }

.panel-box {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 14px 18px; margin-bottom: 14px;
}
.panel-title { font-size: 14.5px; font-weight: 600; color: #0f172a; margin: 0 0 4px 0; }
.panel-sub { font-size: 12px; color: #64748b; margin: 0 0 10px 0; }

.action-card {
    background: #ffffff; border: 1px solid #fecdd3; border-left: 4px solid #e11d48;
    border-radius: 10px; padding: 13px 15px; margin-bottom: 10px;
}
.action-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.action-rank {
    background: #fff1f2; color: #be123c; font-size: 11.5px;
    font-weight: 600; padding: 2px 9px; border-radius: 999px;
}
.action-name { font-size: 15px; font-weight: 700; color: #0f172a; }
.action-stats { font-size: 12.5px; color: #334155; margin-bottom: 7px; }
.action-guide {
    background: #f8fafc; border-radius: 8px; padding: 8px 10px;
    font-size: 12px; color: #475569; line-height: 1.55;
}

.safe-card {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 4px solid #16a34a;
    border-radius: 10px; padding: 15px 18px; color: #166534; font-size: 13.5px; line-height: 1.6;
}

.tbl-wrap {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
    overflow-x: auto; margin-top: 6px; margin-bottom: 14px;
}
.m-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.m-table th {
    background: #f8fafc; color: #475569; font-weight: 600; font-size: 12px;
    padding: 10px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; white-space: nowrap;
}
.m-table td {
    padding: 10px 14px; border-bottom: 1px solid #f1f5f9; color: #1e293b; vertical-align: middle;
}
.m-table tr:last-child td { border-bottom: none; }
.m-table tbody tr:hover { background-color: #f8fafc; }
.num { text-align: right !important; font-variant-numeric: tabular-nums; white-space: nowrap; }
.ctr { text-align: center !important; }
.muted { font-size: 11.5px; color: #94a3b8; display: block; }

.badge-red { background: #fff1f2; color: #be123c; padding: 2px 9px; border-radius: 999px; font-size: 11.5px; font-weight: 600; white-space: nowrap; }
.badge-ok { background: #f0fdf4; color: #15803d; padding: 2px 9px; border-radius: 999px; font-size: 11.5px; font-weight: 600; white-space: nowrap; }
.badge-blue { background: #eff6ff; color: #1d4ed8; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; white-space: nowrap; }
.sec-label { font-size: 14px; font-weight: 600; color: #334155; margin: 12px 0 6px 0; }
.hint-text { font-size: 12px; color: #64748b; margin-top: 4px; }
</style>
""")


# =========================================================
# 1. ระบบจัดเก็บทะเบียนติดตามผล (PostgreSQL ออนไลน์ / CSV สำรอง)
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
    """บันทึกรอบพยากรณ์ครบทุกโรค โดยคงค่าพยากรณ์และเกณฑ์ฉบับส่งไว้ไม่ให้ถูกแก้ทับ"""
    rows = []
    cutoff = payload["meta"]["cutoff_date"]
    for p in payload["provinces"]:
        for d in p["all_diseases"]:
            rows.append({
                "run_id": d["run_id"],
                "province_code": str(d["province_code"]),
                "province_name": str(d["province_name"]),
                "disease_code": str(d["disease_code"]),
                "disease_name": str(d["disease_name"]),
                "target_week": str(d["target_week"]),
                "frozen_forecast": to_int(d["forecast"]),
                "frozen_threshold": to_int(d["threshold"]),
                "is_flagged": bool(d["is_flagged"]),
                "alert_sent_time": f"{cutoff} 09:00",
                "check_start_time": "",
                "field_result": "",
                "action_taken": "",
                "settled_actual_cases": None
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


def load_registry(prov_code):
    engine = get_db_engine()
    if engine:
        ensure_table(engine)
        df_r = pd.read_sql("SELECT * FROM tracking_registry ORDER BY target_week DESC, disease_code ASC", engine)
    elif os.path.exists(CSV_REGISTRY):
        df_r = pd.read_csv(CSV_REGISTRY, dtype={"province_code": str, "disease_code": str})
    else:
        return pd.DataFrame()
    if df_r.empty:
        return df_r
    df_r["province_code"] = df_r["province_code"].astype(str).str.strip()
    df_r["disease_code"] = df_r["disease_code"].astype(str).str.strip().str.zfill(2)
    return df_r[df_r["province_code"] == str(prov_code)].copy()


def update_registry(run_id, prov_code, disease_code, target_week, check_time, field_res, action_txt, actual_val):
    engine = get_db_engine()
    if engine:
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE tracking_registry
                SET check_start_time=:ct, field_result=:fr, action_taken=:at, settled_actual_cases=:ac
                WHERE run_id=:rid AND province_code=:pc AND disease_code=:dc AND target_week=:tw;
            """), {
                "ct": check_time, "fr": field_res, "at": action_txt, "ac": actual_val,
                "rid": run_id, "pc": str(prov_code), "dc": str(disease_code), "tw": str(target_week)
            })
    else:
        df_all = pd.read_csv(CSV_REGISTRY, dtype={"province_code": str, "disease_code": str})
        m = (
            (df_all["run_id"] == run_id) &
            (df_all["province_code"] == str(prov_code)) &
            (df_all["disease_code"].str.zfill(2) == str(disease_code).zfill(2)) &
            (df_all["target_week"] == str(target_week))
        )
        df_all.loc[m, "check_start_time"] = check_time
        df_all.loc[m, "field_result"] = field_res
        df_all.loc[m, "action_taken"] = action_txt
        df_all.loc[m, "settled_actual_cases"] = actual_val
        df_all.to_csv(CSV_REGISTRY, index=False, encoding="utf-8-sig")


# =========================================================
# 2. โหลดข้อมูลยอดรวมและคำนวณพยากรณ์ตามวันตัดรอบ
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
            s = grp.set_index("week_start")["cases"].ffill().fillna(0)
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
                "run_id": run_id,
                "province_code": p_code,
                "province_name": p_name,
                "disease_code": d_code,
                "disease_name": DISEASE_NAMES.get(d_code, d_code),
                "target_week": target_week,
                "latest_cases": int(round(float(s.iloc[-1]))),
                "forecast": fc,
                "pi_low": pi_low,
                "pi_high": pi_high,
                "threshold": th,
                "is_flagged": flagged,
                "excess_ratio": round(fc / th, 2) if th > 0 else 0.0,
                "reason": (
                    f"ยอดคาด {fc:,} ราย สูงกว่าเกณฑ์เตือน {th:,} ราย"
                    if flagged else
                    f"ยอดคาด {fc:,} ราย ต่ำกว่าเกณฑ์เตือน {th:,} ราย"
                ),
                "history": [
                    {
                        "week": r["week_start"].strftime("%Y-%m-%d"),
                        "cases": None if pd.isna(r["cases"]) else int(round(float(r["cases"]))),
                        "status": r["data_status"]
                    }
                    for _, r in grp.tail(52).iterrows()
                ]
            })

        top_alerts = sorted([d for d in diseases if d["is_flagged"]], key=lambda x: x["excess_ratio"], reverse=True)[:3]
        provinces.append({
            "province_code": p_code,
            "province_name": p_name,
            "alert_count": len(top_alerts),
            "alerts": top_alerts,
            "all_diseases": diseases
        })

    return {
        "meta": {
            "run_id": run_id,
            "cutoff_date": cutoff_str,
            "target_week": target_week,
            "data_status": df_cut["data_status"].iloc[-1]
        },
        "provinces": provinces
    }


def ratio_text(num, den):
    return f"{num}/{den} ({num/den:.0%})" if den > 0 else f"{num}/0 (ไม่มีตัวหาร)"


def clean_str(val):
    return "" if pd.isna(val) or str(val).strip().lower() == "nan" else str(val).strip()


# =========================================================
# 3. แถบเมนูซ้ายมือ (เปลี่ยนจังหวัด + เปลี่ยนวันตัดข้อมูล)
# =========================================================
latest_json, weekly_df = load_files()
min_date = (weekly_df["week_start"].min() + pd.Timedelta(weeks=4)).date()
max_date = pd.to_datetime(latest_json["meta"]["cutoff_date"]).date()

with st.sidebar:
    st.markdown("### ตัวกรองแดชบอร์ด")
    picked_date = st.date_input(
        "วันตัดข้อมูล (เลือกย้อนหลังเพื่อจำลองรอบได้)",
        value=max_date, min_value=min_date, max_value=max_date
    )
    picked_str = picked_date.strftime("%Y-%m-%d")

    payload = latest_json if picked_str == latest_json["meta"]["cutoff_date"] else predict_for_date(weekly_df, picked_str)
    sync_to_registry(payload)

    prov_map = {
        f"จ.{p['province_name']} ({p['province_code']})  •  เฝ้าระวัง {p['alert_count']} โรค": p
        for p in payload["provinces"]
    }
    prov_label = st.selectbox("เลือกจังหวัด", list(prov_map.keys()))
    prov = prov_map[prov_label]

    st.divider()
    with st.expander("คู่มืออ่านแดชบอร์ดฉบับย่อ"):
        st.markdown("""
        - **เกณฑ์เตือน:** ระดับผู้ป่วยปกติของโรคนั้นในจังหวัด (จากสถิติ 2 ปีแรก)
        - **ช่วงพยากรณ์ 95%:** ช่วงต่ำสุด–สูงสุดที่ยอดจริงมีโอกาสเกิด
        - **แถบสีเหลืองในกราฟ:** ยอด 1–2 สัปดาห์ล่าสุดที่อาจยังมีรายงานค้างส่ง
        """)
    db_mode = "PostgreSQL ออนไลน์" if get_db_engine() else "ไฟล์ CSV สาธิต"
    st.caption(f"จัดเก็บทะเบียน: {db_mode}")

meta = payload["meta"]

# =========================================================
# 4. ส่วนหัวแดชบอร์ด
# =========================================================
status_cls = "meta-tag-warn" if "เบื้องต้น" in str(meta["data_status"]) else "meta-tag"
render_html(f"""
<div class="top-header">
    <div class="top-title">แดชบอร์ดชี้เป้าโรคที่ควรเฝ้าระวังสัปดาห์หน้า — จังหวัด{prov['province_name']}</div>
    <div class="meta-line">
        <span class="meta-tag">รหัสรอบ: <b>{meta['run_id']}</b></span>
        <span class="meta-tag">พยากรณ์สัปดาห์เริ่มวันที่: <b>{meta['target_week']}</b></span>
        <span class="meta-tag">ตัดข้อมูลถึงวันที่: <b>{meta['cutoff_date']}</b></span>
        <span class="{status_cls}">สถานะข้อมูล: {meta['data_status']}</span>
    </div>
</div>
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "แดชบอร์ดชี้เป้าสัปดาห์หน้า",
    "กราฟแนวโน้มรายโรค",
    "บันทึกและติดตามผล",
    "เปรียบเทียบโมเดล"
])

# =========================================================
# แท็บ 1: แดชบอร์ดภาพรวม + กราฟเทียบเกณฑ์ + แนวทางรับมือ
# =========================================================
with tab1:
    alerts = prov["alerts"]
    all_d = prov["all_diseases"]
    total_fc = sum(to_int(d["forecast"]) for d in all_d)
    total_latest = sum(to_int(d["latest_cases"]) for d in all_d)
    diff_total = total_fc - total_latest
    diff_total_txt = f"+{diff_total:,} ราย จากสัปดาห์ล่าสุด" if diff_total >= 0 else f"{diff_total:,} ราย จากสัปดาห์ล่าสุด"

    top_risk_name = alerts[0]["disease_name"] if alerts else "ยังไม่มีโรคเกินเกณฑ์"
    top_risk_note = (
        f"คาด {to_int(alerts[0]['forecast']):,} ราย (เกณฑ์ {to_int(alerts[0]['threshold']):,} ราย)"
        if alerts else "สถานการณ์ทั้ง 3 โรคปกติ"
    )
    alert_color = "#be123c" if len(alerts) > 0 else "#15803d"

    # แถบ KPI สรุปแดชบอร์ด 4 กล่อง
    render_html(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-lbl">โรคที่ต้องเฝ้าระวังสัปดาห์หน้า</div>
            <div class="kpi-val" style="color:{alert_color};">{len(alerts)} จาก {len(all_d)} โรค</div>
            <div class="kpi-note">คัดเฉพาะโรคที่คาดว่าจะเกินเกณฑ์เตือน</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">โรคเสี่ยงสูงสุดอันดับ 1</div>
            <div class="kpi-val" style="font-size:17px; color:{alert_color};">{top_risk_name}</div>
            <div class="kpi-note">{top_risk_note}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">ยอดคาดการณ์รวม 3 โรค (สัปดาห์หน้า)</div>
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

    # ส่วนแดชบอร์ด 2 คอลัมน์: ซ้าย = กราฟภาพรวม 3 โรค | ขวา = โรคที่ควรรับมือและแนวทางปฏิบัติ
    col_chart, col_action = st.columns([1.25, 1.0], gap="medium")

    with col_chart:
        render_html("""
        <div class="panel-box" style="margin-bottom:0; border-bottom-left-radius:0; border-bottom-right-radius:0; border-bottom:none;">
            <div class="panel-title">ภาพรวมเปรียบเทียบ "ยอดคาดสัปดาห์หน้า" กับ "เกณฑ์เตือน"</div>
            <div class="panel-sub">แท่งสีแดง = คาดว่าจะเกินเกณฑ์เตือน (ควรตรวจสอบก่อน) • แท่งสีฟ้า = อยู่ในเกณฑ์ปกติ</div>
        </div>
        """)

        # จัดลำดับกราฟแท่งแนวนอน (ไม่เบียด ไม่ตัดคำภาษาไทย)
        chart_items = list(reversed(all_d))
        y_names = [f"{d['disease_name']} ({d['disease_code']})" for d in chart_items]
        fc_vals = [to_int(d["forecast"]) for d in chart_items]
        th_vals = [to_int(d["threshold"]) for d in chart_items]
        fc_colors = ["#e11d48" if d["is_flagged"] else "#0284c7" for d in chart_items]
        fc_texts = [
            f" {to_int(d['forecast']):,} ราย (เกินเกณฑ์)" if d["is_flagged"] else f" {to_int(d['forecast']):,} ราย"
            for d in chart_items
        ]
        th_texts = [f" {to_int(d['threshold']):,} ราย" for d in chart_items]

        fig_ov = go.Figure()
        fig_ov.add_trace(go.Bar(
            y=y_names,
            x=th_vals,
            name="เกณฑ์เตือนปกติ",
            orientation="h",
            marker=dict(color="#cbd5e1", line=dict(width=0)),
            text=th_texts,
            textposition="outside",
            textfont=dict(size=12, color="#64748b"),
            hovertemplate="%{y}<br>เกณฑ์เตือน: %{x:,} ราย<extra></extra>"
        ))
        fig_ov.add_trace(go.Bar(
            y=y_names,
            x=fc_vals,
            name="ยอดคาดสัปดาห์หน้า",
            orientation="h",
            marker=dict(color=fc_colors, line=dict(width=0)),
            text=fc_texts,
            textposition="outside",
            textfont=dict(size=12.5, color="#0f172a"),
            hovertemplate="%{y}<br>ยอดคาดสัปดาห์หน้า: %{x:,} ราย<extra></extra>"
        ))

        max_x = max(fc_vals + th_vals + [1])
        fig_ov.update_layout(
            barmode="group",
            bargap=0.28,
            bargroupgap=0.12,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0,
                font=dict(size=12.5, color="#334155")
            ),
            font=dict(family="IBM Plex Sans Thai, sans-serif", size=12.5, color="#334155"),
            margin=dict(l=10, r=30, t=42, b=28),
            height=330
        )
        fig_ov.update_xaxes(
            showgrid=True, gridcolor="#f1f5f9", zeroline=False,
            range=[0, max_x * 1.42], nticks=5, tickformat=",d",
            title_text="จำนวนผู้ป่วย (ราย)", title_font=dict(size=12, color="#64748b")
        )
        fig_ov.update_yaxes(showgrid=False, automargin=True, tickfont=dict(size=12.5, color="#0f172a"))
        st.plotly_chart(fig_ov, use_container_width=True)

    with col_action:
        render_html('<div class="sec-label" style="margin-top:0;">โรคที่ทีมระบาดวิทยาควรเฝ้าระวังและเตรียมรับมือสัปดาห์หน้า</div>')
        if not alerts:
            render_html(f"""
            <div class="safe-card">
                <b>สัปดาห์เป้าหมาย {meta['target_week']} ยังไม่มีสัญญาณเตือน (0 โรค)</b><br>
                ยอดคาดการณ์ของโรคอุจจาระร่วงเฉียบพลัน ไข้หวัดใหญ่ และปอดอักเสบ ในจังหวัด{prov['province_name']} ยังอยู่ต่ำกว่าเกณฑ์เตือนทั้งหมด สามารถเฝ้าระวังตามรอบปกติได้ครับ
            </div>
            """)
        else:
            for i, item in enumerate(alerts):
                fc_i = to_int(item["forecast"])
                th_i = to_int(item["threshold"])
                lt_i = to_int(item["latest_cases"])
                lo_i = to_int(item["pi_low"])
                hi_i = to_int(item["pi_high"])
                diff = fc_i - th_i
                guide = DISEASE_ACTION_GUIDE.get(item["disease_code"], "ตรวจสอบข้อมูลรายอำเภอและยืนยันแนวโน้มกับทีมพื้นที่")

                render_html(f"""
                <div class="action-card">
                    <div class="action-head">
                        <span class="action-name">{item['disease_name']}</span>
                        <span class="action-rank">อันดับ #{i+1} • เกินเกณฑ์ +{diff:,} ราย</span>
                    </div>
                    <div class="action-stats">
                        คาดสัปดาห์หน้า <b style="color:#be123c;">{fc_i:,} ราย</b> (เกณฑ์ {th_i:,} ราย) • ล่าสุด {lt_i:,} ราย • ช่วง 95%: {lo_i:,}–{hi_i:,} ราย
                    </div>
                    <div class="action-guide">
                        <b>แนวทางรับมือเบื้องต้น:</b> {guide}
                    </div>
                </div>
                """)

    # ตารางสรุปครบทุกโรคในขอบเขต (รวมโรคปกติ)
    render_html('<div class="sec-label">ตารางสรุปตัวเลขครบทุกโรคในขอบเขต (รวมโรคที่สถานะปกติ)</div>')
    rows_html = []
    for d in all_d:
        fc_i = to_int(d["forecast"])
        th_i = to_int(d["threshold"])
        lt_i = to_int(d["latest_cases"])
        lo_i = to_int(d["pi_low"])
        hi_i = to_int(d["pi_high"])
        diff = fc_i - th_i

        if d["is_flagged"]:
            badge = '<span class="badge-red">ควรเฝ้าระวัง</span>'
            diff_txt = f'<span style="color:#be123c; font-weight:600;">+{diff:,} ราย</span>'
        else:
            badge = '<span class="badge-ok">ปกติ</span>'
            diff_txt = f'<span style="color:#15803d;">{diff:,} ราย</span>'

        rows_html.append(f"""
        <tr>
            <td><b>{d['disease_code']}</b></td>
            <td><b>{d['disease_name']}</b></td>
            <td class="num">{lt_i:,}</td>
            <td class="num"><b>{fc_i:,}</b><span class="muted">ช่วง {lo_i:,}–{hi_i:,}</span></td>
            <td class="num">{th_i:,}</td>
            <td class="num">{diff_txt}</td>
            <td class="ctr">{badge}</td>
        </tr>
        """)
    render_html(f"""
    <div class="tbl-wrap">
    <table class="m-table">
        <thead>
            <tr>
                <th>รหัส</th><th>ชื่อโรค</th><th class="num">ยอดล่าสุด</th>
                <th class="num">ยอดคาดสัปดาห์หน้า</th><th class="num">เกณฑ์เตือน</th>
                <th class="num">ต่างจากเกณฑ์</th><th class="ctr">สถานะ</th>
            </tr>
        </thead>
        <tbody>{"".join(rows_html)}</tbody>
    </table>
    </div>
    """)

# =========================================================
# แท็บ 2: กราฟแนวโน้มรายโรค (จัดวางสะอาด ตัวหนังสือไม่เบียดกัน)
# =========================================================
with tab2:
    d_map = {f"{d['disease_code']} - {d['disease_name']}": d for d in prov["all_diseases"]}
    sel_d = st.selectbox("เลือกโรคที่ต้องการดูแนวโน้มย้อนหลัง 1 ปี", list(d_map.keys()))
    d = d_map[sel_d]

    fc_i = to_int(d["forecast"])
    th_i = to_int(d["threshold"])
    lt_i = to_int(d["latest_cases"])
    lo_i = to_int(d["pi_low"])
    hi_i = to_int(d["pi_high"])

    st_color = "#be123c" if d["is_flagged"] else "#15803d"
    st_label = "เกินเกณฑ์เตือน (ควรเฝ้าระวัง)" if d["is_flagged"] else "ต่ำกว่าเกณฑ์ (ปกติ)"

    render_html(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-lbl">ยอดจริงสัปดาห์ล่าสุด</div>
            <div class="kpi-val">{lt_i:,} ราย</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">พยากรณ์สัปดาห์เป้าหมาย ({d['target_week']})</div>
            <div class="kpi-val" style="color:{st_color};">{fc_i:,} ราย</div>
            <div class="kpi-note">ช่วงที่เป็นไปได้ 95%: {lo_i:,} – {hi_i:,} ราย</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">เกณฑ์เตือนรายโรค</div>
            <div class="kpi-val">{th_i:,} ราย</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">ผลการเทียบเกณฑ์</div>
            <div class="kpi-val" style="font-size:15.5px; color:{st_color}; margin-top:6px;">{st_label}</div>
        </div>
    </div>
    """)

    render_html(f"""
    <div class="panel-box" style="margin-bottom:0; border-bottom-left-radius:0; border-bottom-right-radius:0; border-bottom:none;">
        <div class="panel-title">แนวโน้มผู้ป่วยรายสัปดาห์ — {d['disease_name']} (จังหวัด{prov['province_name']})</div>
        <div class="panel-sub">แสดงข้อมูลย้อนหลัง 1 ปี เทียบกับเส้นเกณฑ์เตือนและจุดพยากรณ์สัปดาห์หน้า (ชี้เมาส์บนกราฟเพื่อดูตัวเลขแต่ละสัปดาห์)</div>
    </div>
    """)

    hist = pd.DataFrame(d["history"])
    hist["week_dt"] = pd.to_datetime(hist["week"])
    hist["cases_num"] = pd.to_numeric(hist["cases"], errors="coerce")
    target_dt = pd.to_datetime(d["target_week"])
    first_dt = hist["week_dt"].iloc[0]

    fig = go.Figure()

    # 1. แถบสีเหลืองอ่อนช่วงข้อมูลเบื้องต้น (ไม่แปะตัวหนังสือทับในกราฟ ใส่ใน Legend ด้านบนแทน)
    prelim = hist[hist["status"].str.contains("เบื้องต้น", na=False)]
    if not prelim.empty:
        fig.add_vrect(
            x0=prelim["week_dt"].iloc[0] - pd.Timedelta(days=3),
            x1=prelim["week_dt"].iloc[-1] + pd.Timedelta(days=3),
            fillcolor="rgba(245, 158, 11, 0.12)",
            line_width=0
        )

    # 2. เส้นเกณฑ์เตือน (ทำเป็น Trace ใน Legend เพื่อให้ข้อความไม่ทับเส้นหรือดาว)
    fig.add_trace(go.Scatter(
        x=[first_dt, target_dt],
        y=[th_i, th_i],
        mode="lines",
        name=f"เกณฑ์เตือน ({th_i:,} ราย)",
        line=dict(color="#f59e0b", width=2.2, dash="dot"),
        hovertemplate=f"เกณฑ์เตือน: {th_i:,} ราย<extra></extra>"
    ))

    # 3. เส้นยอดจริงสีน้ำเงินพร้อมจุดกลม
    fig.add_trace(go.Scatter(
        x=hist["week_dt"],
        y=hist["cases_num"],
        mode="lines+markers",
        name="ยอดผู้ป่วยจริง",
        line=dict(color="#0284c7", width=2.4),
        marker=dict(size=6.5, color="#0284c7", line=dict(color="#ffffff", width=1)),
        hovertemplate="สัปดาห์ %{x|%d %b %Y}<br>ยอดจริง: %{y:,.0f} ราย<extra></extra>"
    ))

    # 4. เส้นประสีแดงเชื่อมจากยอดล่าสุดไปยังจุดพยากรณ์สัปดาห์หน้า
    valid = hist.dropna(subset=["cases_num"])
    if not valid.empty:
        fig.add_trace(go.Scatter(
            x=[valid["week_dt"].iloc[-1], target_dt],
            y=[valid["cases_num"].iloc[-1], fc_i],
            mode="lines",
            line=dict(color="#e11d48", width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip"
        ))

    # 5. ดาวสีแดงพยากรณ์สัปดาห์หน้า
    fig.add_trace(go.Scatter(
        x=[target_dt],
        y=[fc_i],
        mode="markers",
        name=f"พยากรณ์สัปดาห์หน้า ({fc_i:,} ราย)",
        marker=dict(color="#e11d48", size=14, symbol="star", line=dict(color="#ffffff", width=1)),
        hovertemplate=f"สัปดาห์เป้าหมาย {d['target_week']}<br>พยากรณ์: {fc_i:,} ราย<br>ช่วง 95%: {lo_i:,}–{hi_i:,} ราย<extra></extra>"
    ))

    # 6. สัญลักษณ์แถบข้อมูลเบื้องต้นใน Legend (ป้องกันข้อความซ้อนในกราฟ)
    if not prelim.empty:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            name="แถบเหลือง = ข้อมูลเบื้องต้น (รอยอดนิ่ง)",
            marker=dict(symbol="square", size=10, color="rgba(245, 158, 11, 0.35)", line=dict(color="#f59e0b", width=1)),
            hoverinfo="skip"
        ))

    y_max = max(float(valid["cases_num"].max() if not valid.empty else 0), float(th_i), float(fc_i), 1.0)
    y_pad_top = y_max * 1.22
    y_pad_bottom = -max(0.3, y_max * 0.08)

    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
        # วาง Legend แนวนอนด้านบนสุด โล่งและไม่ชนกับเส้นกราฟ
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.04,
            xanchor="left", x=0,
            font=dict(size=12.5, color="#334155"),
            itemclick="toggleothers"
        ),
        font=dict(family="IBM Plex Sans Thai, sans-serif", size=12.5, color="#64748b"),
        margin=dict(l=45, r=25, t=65, b=45),
        height=410
    )
    fig.update_xaxes(
        showgrid=False,
        tickformat="%b %Y",
        nticks=7,
        tickangle=0,
        zeroline=False,
        automargin=True,
        range=[first_dt - pd.Timedelta(days=7), target_dt + pd.Timedelta(days=18)],
        tickfont=dict(size=12, color="#64748b")
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#eef2f6",
        zeroline=False,
        nticks=6,
        tickformat=",d",
        automargin=True,
        range=[y_pad_bottom, y_pad_top],
        title_text="จำนวนผู้ป่วย (ราย)",
        title_font=dict(size=12, color="#64748b"),
        tickfont=dict(size=12, color="#64748b")
    )

    st.plotly_chart(fig, use_container_width=True)

    # แนวทางรับมือเฉพาะโรคที่เลือกดูอยู่
    guide_sel = DISEASE_ACTION_GUIDE.get(d["disease_code"], "")
    if guide_sel:
        render_html(f'<div class="hint-text"><b>ข้อแนะนำสำหรับทีมระบาดวิทยา ({d["disease_name"]}):</b> {guide_sel}</div>')

# =========================================================
# แท็บ 3: บันทึกและติดตามผล + ประเมินรายเดือน 4 ตัวชี้วัด
# =========================================================
with tab3:
    reg_df = load_registry(prov["province_code"])
    if reg_df.empty:
        st.info("ยังไม่มีรายการในทะเบียนของจังหวัดนี้")
    else:
        reg_df["month"] = reg_df["target_week"].astype(str).str.slice(0, 7)
        c_title, c_month = st.columns([3, 1])
        with c_title:
            render_html('<div class="sec-label">ผลประเมินระบบติดตามผลรายเดือน (4 ตัวชี้วัด)</div>')
        sel_m = c_month.selectbox(
            "เลือกเดือน", ["ทั้งหมด"] + sorted(reg_df["month"].unique().tolist(), reverse=True),
            label_visibility="collapsed"
        )

        ev = reg_df if sel_m == "ทั้งหมด" else reg_df[reg_df["month"] == sel_m]
        ev["settled_num"] = pd.to_numeric(ev["settled_actual_cases"], errors="coerce")
        closed = ev[ev["settled_num"].notna()].copy()

        if closed.empty:
            render_html('<div class="hint-text" style="margin-bottom:12px;">ยังไม่มีรอบที่กรอก "ยอดจริงเมื่อข้อมูลนิ่ง" ในช่วงเวลาที่เลือก</div>')
        else:
            flagged = closed["is_flagged"].astype(str).str.lower() == "true"
            actual_met = closed["settled_num"] >= pd.to_numeric(closed["frozen_threshold"], errors="coerce")
            tp = int((flagged & actual_met).sum())
            fp = int((flagged & ~actual_met).sum())
            fn = int((~flagged & actual_met).sum())
            avg_hr = (
                (pd.to_datetime(closed["check_start_time"], errors="coerce") - pd.to_datetime(closed["alert_sent_time"], errors="coerce"))
                .dt.total_seconds() / 3600
            ).dropna().mean()
            spd = f"{avg_hr:.1f} ชม." if pd.notna(avg_hr) else "ยังไม่ระบุ"

            render_html(f"""
            <div class="kpi-row">
                <div class="kpi-card"><div class="kpi-lbl">1. ความแม่นของสัญญาณ</div><div class="kpi-val">{ratio_text(tp, tp+fp)}</div><div class="kpi-note">เตือนแล้วเข้าเกณฑ์จริง</div></div>
                <div class="kpi-card"><div class="kpi-lbl">2. ความครอบคลุม</div><div class="kpi-val">{ratio_text(tp, tp+fn)}</div><div class="kpi-note">จับเหตุการณ์จริงได้</div></div>
                <div class="kpi-card"><div class="kpi-lbl">3. ภาระตรวจสอบ</div><div class="kpi-val">รวม {tp+fp} (เกิน {fp})</div><div class="kpi-note">สัญญาณทั้งหมด / แจ้งเกิน</div></div>
                <div class="kpi-card"><div class="kpi-lbl">4. ความเร็วเริ่มตรวจ</div><div class="kpi-val">{spd}</div><div class="kpi-note">จากส่งสัญญาณถึงเริ่มตรวจ</div></div>
            </div>
            """)

        with st.expander("บันทึกผลการตรวจสอบ / ปิดยอดจริง", expanded=True):
            reg_df["label"] = reg_df["run_id"] + "  |  " + reg_df["disease_code"] + " - " + reg_df["disease_name"] + "  |  สัปดาห์ " + reg_df["target_week"]
            sel_rec = st.selectbox("เลือกรอบและโรคที่ต้องการบันทึก", reg_df["label"].tolist())
            rec = reg_df[reg_df["label"] == sel_rec].iloc[0]

            with st.form("track_form", border=False):
                c1, c2, c3, c4 = st.columns([1.2, 1.1, 1.5, 0.9])
                in_time = c1.text_input("เวลาเริ่มตรวจ", value=clean_str(rec["check_start_time"]), placeholder="2025-12-31 10:30")
                f_opts = ["", "พบกลุ่มก้อน", "พบการส่งย้อนหลัง", "ยังสรุปไม่ได้", "ปกติ"]
                cur_f = clean_str(rec["field_result"])
                in_field = c2.selectbox("ผลภาคสนาม", f_opts, index=f_opts.index(cur_f) if cur_f in f_opts else 0)
                in_action = c3.text_input("การดำเนินการ", value=clean_str(rec["action_taken"]), placeholder="เช่น แจ้ง SRRT ตรวจสอบ")
                raw_act = clean_str(rec["settled_actual_cases"])
                cur_act = str(to_int(raw_act)) if raw_act != "" else ""
                in_actual = c4.text_input("ยอดจริงนิ่งแล้ว", value=cur_act, placeholder="จำนวนเต็ม")

                if st.form_submit_button("บันทึกผล", type="primary"):
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
                    update_registry(
                        rec["run_id"], rec["province_code"], rec["disease_code"], rec["target_week"],
                        in_time.strip(), in_field, in_action.strip(), actual_val
                    )
                    st.success("บันทึกเรียบร้อยแล้ว")
                    st.rerun()

        render_html('<div class="sec-label">ทะเบียนติดตามผล (คงค่าพยากรณ์และเกณฑ์เดิมไว้เทียบผลจริง)</div>')
        reg_rows = []
        for _, r in reg_df.iterrows():
            f_badge = '<span class="badge-red">ชี้เป้า</span>' if str(r["is_flagged"]).lower() == "true" else '<span class="badge-ok">ปกติ</span>'
            act_s = clean_str(r["settled_actual_cases"])
            act_txt = f"<b>{fmt_int(act_s)} ราย</b>" if act_s != "" else '<span style="color:#94a3b8;">รอยอดนิ่ง</span>'
            reg_rows.append(f"""
            <tr>
                <td><b>{r['run_id']}</b><span class="muted">เป้าหมาย {r['target_week']}</span></td>
                <td><b>{r['disease_name']}</b></td>
                <td class="num"><b>{fmt_int(r['frozen_forecast'])}</b><span class="muted">เกณฑ์ {fmt_int(r['frozen_threshold'])}</span></td>
                <td class="ctr">{f_badge}</td>
                <td>{clean_str(r['check_start_time']) or '-'}</td>
                <td>{clean_str(r['field_result']) or '-'}</td>
                <td>{clean_str(r['action_taken']) or '-'}</td>
                <td class="num">{act_txt}</td>
            </tr>
            """)
        render_html(f"""
        <div class="tbl-wrap">
        <table class="m-table">
            <thead>
                <tr>
                    <th>รอบรายงาน</th><th>โรค</th><th class="num">คาด / เกณฑ์</th><th class="ctr">สถานะ</th>
                    <th>เวลาเริ่มตรวจ</th><th>ผลภาคสนาม</th><th>การดำเนินการ</th><th class="num">ยอดจริงนิ่งแล้ว</th>
                </tr>
            </thead>
            <tbody>{"".join(reg_rows)}</tbody>
        </table>
        </div>
        """)

# =========================================================
# แท็บ 4: เปรียบเทียบโมเดลย้อนหลัง
# =========================================================
with tab4:
    render_html('<div class="hint-text" style="margin-bottom:10px;">ทดสอบย้อนหลังในปีที่ 3 แบบเลื่อนทีละสัปดาห์ (ใช้ข้อมูลถึงวันตัดรอบเท่านั้น) • <b>MAE</b> ยิ่งน้อยยิ่งทายแม่น • <b>เตือนถูก</b> ยิ่งมากยิ่งดี • <b>เตือนเกิน/พลาด</b> ยิ่งน้อยยิ่งดี</div>')
    if not os.path.exists(BACKTEST_FILE):
        st.info("ยังไม่มีไฟล์ผลเปรียบเทียบโมเดล (backtest_report.csv)")
    else:
        bt = pd.read_csv(BACKTEST_FILE, dtype={"รหัสจังหวัด": str, "รหัสโรค": str})
        bt["รหัสจังหวัด"] = bt["รหัสจังหวัด"].str.strip().str.replace(r"\.0$", "", regex=True)
        bt = bt[bt["รหัสจังหวัด"] == str(prov["province_code"])].copy()
        if bt.empty:
            st.info("ไม่มีผลทดสอบย้อนหลังสำหรับจังหวัดนี้")
        else:
            best_mae = bt.groupby("รหัสโรค")["MAE"].transform("min")
            bt_rows = []
            for idx, r in bt.iterrows():
                tag = ' <span class="badge-blue">แม่นสุด</span>' if r["MAE"] == best_mae.loc[idx] else ""
                bt_rows.append(f"""
                <tr>
                    <td><b>{r['ชื่อโรค']}</b><span class="muted">เกณฑ์ {fmt_int(r['เกณฑ์ชี้เป้า'])} ราย ({fmt_int(r['สัปดาห์ทดสอบ'])} สัปดาห์)</span></td>
                    <td>{r['โมเดล']}{tag}</td>
                    <td class="num"><b>{r['MAE']}</b></td>
                    <td class="num" style="color:#15803d; font-weight:600;">{fmt_int(r['จับได้ (TP)'])}</td>
                    <td class="num" style="color:#b45309;">{fmt_int(r['แจ้งเกิน (FP)'])}</td>
                    <td class="num" style="color:#be123c;">{fmt_int(r['พลาด (FN)'])}</td>
                    <td class="num">{r['ความแม่นของสัญญาณ']}</td>
                    <td class="num">{r['ความครอบคลุม']}</td>
                </tr>
                """)
            render_html(f"""
            <div class="tbl-wrap">
            <table class="m-table">
                <thead>
                    <tr>
                        <th>โรค</th><th>วิธีคำนวณ</th><th class="num">ทายพลาดเฉลี่ย (MAE)</th>
                        <th class="num">เตือนถูก</th><th class="num">เตือนเกิน</th><th class="num">พลาด</th>
                        <th class="num">ความแม่นสัญญาณ</th><th class="num">ความครอบคลุม</th>
                    </tr>
                </thead>
                <tbody>{"".join(bt_rows)}</tbody>
            </table>
            </div>
            <div class="hint-text">หมายเหตุ: หากไฟล์ที่ใช้ทดสอบเป็นข้อมูลปรับย้อนหลัง ยอดผู้ป่วยจริงอาจสูงกว่ายอดที่เห็น ณ วันพยากรณ์จริงเนื่องจากความล่าช้าในการส่งรายงาน</div>
            """)
