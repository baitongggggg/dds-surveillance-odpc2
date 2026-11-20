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
    page_title="ระบบชี้เป้าโรคเฝ้าระวัง DDS",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# สไตล์มินิมอล คลีน สบายตา (ปรับขนาดฟอนต์ทั้งหมดให้พอดีสายตา)
# =========================================================
st.markdown("""
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

/* จัดระยะขอบหน้าเว็บให้พอดี ไม่กว้างหรือแคบเกินไป */
.block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1120px !important;
}

/* ย่อขนาดหัวข้อมาตรฐานของ Streamlit ไม่ให้ใหญ่เทอะทะ */
h1 { font-size: 1.35rem !important; font-weight: 700 !important; color: #0f172a !important; margin-bottom: 0.2rem !important; }
h2 { font-size: 1.05rem !important; font-weight: 600 !important; color: #0f172a !important; }
h3 { font-size: 0.95rem !important; font-weight: 600 !important; color: #334155 !important; }

/* แถบแท็บด้านบนแบบมินิมอล */
.stTabs [data-baseweb="tab-list"] {
    gap: 18px;
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

/* ส่วนหัวเว็บแบบเรียบง่าย ไม่กินพื้นที่จอ */
.top-header {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
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

/* แถบสรุปภาษาคนแบบบาง ไม่แย่งสายตา */
.summary-bar {
    background: #f0f9ff; border-left: 3px solid #0284c7; color: #0c4a6e;
    padding: 10px 14px; border-radius: 8px; font-size: 13.5px; margin-bottom: 14px;
}
.summary-bar-ok {
    background: #f0fdf4; border-left: 3px solid #16a34a; color: #166534;
    padding: 10px 14px; border-radius: 8px; font-size: 13.5px; margin-bottom: 14px;
}

/* การ์ดชี้เป้า (คลีน โปร่ง ตัวเลขชัด) */
.alert-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 12px;
    margin-bottom: 18px;
}
.alert-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #e11d48;
    border-radius: 12px;
    padding: 15px 17px;
}
.alert-rank {
    display: inline-block; background: #fff1f2; color: #be123c;
    font-size: 11.5px; font-weight: 600; padding: 2px 9px; border-radius: 999px;
}
.alert-name { font-size: 16px; font-weight: 700; color: #0f172a; margin: 8px 0 10px 0; }
.alert-nums {
    display: flex; justify-content: space-between; align-items: baseline;
    background: #f8fafc; border-radius: 8px; padding: 8px 12px; margin-bottom: 8px;
}
.alert-big { font-size: 20px; font-weight: 700; color: #be123c; }
.alert-sub { font-size: 12.5px; color: #64748b; }
.alert-detail { font-size: 12.5px; color: #475569; line-height: 1.65; }

/* การ์ดตัวเลข KPI ขนาดกะทัดรัด */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 10px;
    margin: 6px 0 14px 0;
}
.kpi-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 12px 15px;
}
.kpi-lbl { font-size: 12px; color: #64748b; }
.kpi-val { font-size: 19px; font-weight: 700; color: #0f172a; margin-top: 2px; }
.kpi-note { font-size: 11.5px; color: #94a3b8; margin-top: 2px; }

/* ตารางมินิมอล เส้นบาง จัดตัวเลขชิดขวา */
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
""", unsafe_allow_html=True)

WEEKLY_FILE = "weekly_summary.csv"
JSON_FILE = "dashboard_data.json"
BACKTEST_FILE = "backtest_report.csv"
CSV_REGISTRY = "tracking_registry.csv"
DISEASE_NAMES = {
    "02": "โรคอุจจาระร่วงเฉียบพลัน",
    "15": "โรคไข้หวัดใหญ่",
    "31": "โรคปอดอักเสบ"
}


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
                "frozen_forecast": float(d["forecast"]),
                "frozen_threshold": float(d["threshold"]),
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
            th = round(float(np.percentile(s.iloc[:min(104, len(s))], 75)), 1)
            try:
                m = ExponentialSmoothing(s, trend=None, seasonal=None).fit()
                fc = max(0.0, float(m.forecast(1).iloc[0]))
                std = float(np.std(m.resid))
            except Exception:
                fc, std = float(s.iloc[-1]), float(np.std(s))

            pi_low = round(max(0.0, fc - 1.96 * std), 1)
            pi_high = round(fc + 1.96 * std, 1)
            flagged = fc >= th
            diseases.append({
                "run_id": run_id,
                "province_code": p_code,
                "province_name": p_name,
                "disease_code": d_code,
                "disease_name": DISEASE_NAMES.get(d_code, d_code),
                "target_week": target_week,
                "latest_cases": int(s.iloc[-1]),
                "forecast": round(fc, 1),
                "pi_low": pi_low,
                "pi_high": pi_high,
                "threshold": th,
                "is_flagged": flagged,
                "excess_ratio": round(fc / th, 2) if th > 0 else 0.0,
                "reason": (
                    f"ยอดคาด {round(fc,1)} ราย เกินเกณฑ์ {th} ราย"
                    if flagged else
                    f"ยอดคาด {round(fc,1)} ราย ต่ำกว่าเกณฑ์ {th} ราย"
                ),
                "history": [
                    {
                        "week": r["week_start"].strftime("%Y-%m-%d"),
                        "cases": None if pd.isna(r["cases"]) else int(r["cases"]),
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
# 3. แถบเมนูซ้ายมือ (กะทัดรัด ใช้งานง่าย)
# =========================================================
latest_json, weekly_df = load_files()
min_date = (weekly_df["week_start"].min() + pd.Timedelta(weeks=4)).date()
max_date = pd.to_datetime(latest_json["meta"]["cutoff_date"]).date()

with st.sidebar:
    st.markdown("### ตัวกรองข้อมูล")
    picked_date = st.date_input(
        "วันตัดข้อมูล (เลือกย้อนหลังได้)",
        value=max_date, min_value=min_date, max_value=max_date
    )
    picked_str = picked_date.strftime("%Y-%m-%d")

    payload = latest_json if picked_str == latest_json["meta"]["cutoff_date"] else predict_for_date(weekly_df, picked_str)
    sync_to_registry(payload)

    prov_map = {
        f"จ.{p['province_name']} ({p['province_code']})  •  ชี้เป้า {p['alert_count']} โรค": p
        for p in payload["provinces"]
    }
    prov_label = st.selectbox("เลือกจังหวัด", list(prov_map.keys()))
    prov = prov_map[prov_label]

    st.divider()
    with st.expander("คู่มืออ่านผลฉบับย่อ"):
        st.markdown("""
        - **เกณฑ์เตือน:** ระดับผู้ป่วยอ้างอิงจากสถิติ 2 ปีแรกของจังหวัด
        - **ช่วงพยากรณ์ 95%:** ช่วงต่ำสุด–สูงสุดที่ยอดจริงมีโอกาสเกิด
        - **ข้อมูลเบื้องต้น:** ยอด 1–2 สัปดาห์ล่าสุดที่อาจยังมีรายงานค้างส่ง
        """)
    db_mode = "PostgreSQL ออนไลน์" if get_db_engine() else "ไฟล์ CSV สาธิต"
    st.caption(f"จัดเก็บทะเบียน: {db_mode}")

meta = payload["meta"]

# =========================================================
# 4. ส่วนหัวแบบมินิมอล (แถวเดียว ไม่เปลืองพื้นที่จอ)
# =========================================================
status_cls = "meta-tag-warn" if "เบื้องต้น" in str(meta["data_status"]) else "meta-tag"
st.markdown(f"""
<div class="top-header">
    <div class="top-title">ชี้เป้าโรคที่ควรเฝ้าระวังสัปดาห์หน้า — จังหวัด{prov['province_name']}</div>
    <div class="meta-line">
        <span class="meta-tag">รอบ: <b>{meta['run_id']}</b></span>
        <span class="meta-tag">สัปดาห์เป้าหมาย: <b>{meta['target_week']}</b></span>
        <span class="meta-tag">ตัดข้อมูลถึง: <b>{meta['cutoff_date']}</b></span>
        <span class="{status_cls}">สถานะ: {meta['data_status']}</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "ชี้เป้าสัปดาห์หน้า",
    "กราฟรายละเอียดโรค",
    "ติดตามผล",
    "เปรียบเทียบโมเดล"
])

# =========================================================
# แท็บ 1: ชี้เป้าสัปดาห์หน้า
# =========================================================
with tab1:
    alerts = prov["alerts"]

    if not alerts:
        st.markdown(f"""
        <div class="summary-bar-ok">
            <b>จ.{prov['province_name']} สัปดาห์ {meta['target_week']}:</b> ไม่มีโรคที่คาดว่าจะเกินเกณฑ์เตือน (0 สัญญาณ) สถานการณ์ทั้ง 3 โรคอยู่ในระดับปกติ
        </div>
        """, unsafe_allow_html=True)
    else:
        names = " และ ".join([a["disease_name"] for a in alerts])
        st.markdown(f"""
        <div class="summary-bar">
            <b>สรุป จ.{prov['province_name']} (สัปดาห์ {meta['target_week']}):</b>
            พบ <b>{len(alerts)} โรค</b> ที่ควรตรวจสอบก่อน ได้แก่ <b>{names}</b> เนื่องจากยอดคาดการณ์สูงกว่าเกณฑ์ปกติของจังหวัด
        </div>
        """, unsafe_allow_html=True)

        cards = []
        for i, item in enumerate(alerts):
            diff = round(item["forecast"] - item["threshold"], 1)
            cards.append(f"""
            <div class="alert-card">
                <span class="alert-rank">อันดับ #{i+1} • รหัส {item['disease_code']}</span>
                <div class="alert-name">{item['disease_name']}</div>
                <div class="alert-nums">
                    <div>
                        <div class="alert-sub">ยอดคาดสัปดาห์หน้า</div>
                        <div class="alert-big">{item['forecast']} ราย</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="alert-sub">เกณฑ์เตือน</div>
                        <div style="font-size:16px; font-weight:600; color:#334155;">{item['threshold']} ราย</div>
                    </div>
                </div>
                <div class="alert-detail">
                    เกินเกณฑ์: <b style="color:#be123c;">+{diff} ราย</b> ({item['excess_ratio']} เท่า)<br>
                    ยอดล่าสุด: <b>{item['latest_cases']} ราย</b> • ช่วง 95%: {item['pi_low']}–{item['pi_high']} ราย
                </div>
            </div>
            """)
        st.markdown(f'<div class="alert-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">สรุปครบทุกโรคในขอบเขต</div>', unsafe_allow_html=True)
    rows_html = []
    for d in prov["all_diseases"]:
        diff = round(d["forecast"] - d["threshold"], 1)
        if d["is_flagged"]:
            badge = '<span class="badge-red">ควรตรวจสอบก่อน</span>'
            diff_txt = f'<span style="color:#be123c; font-weight:600;">+{diff} ราย</span>'
        else:
            badge = '<span class="badge-ok">ปกติ</span>'
            diff_txt = f'<span style="color:#15803d;">{diff} ราย</span>'

        rows_html.append(f"""
        <tr>
            <td><b>{d['disease_code']}</b></td>
            <td><b>{d['disease_name']}</b></td>
            <td class="num">{d['latest_cases']}</td>
            <td class="num"><b>{d['forecast']}</b><span class="muted">ช่วง {d['pi_low']}–{d['pi_high']}</span></td>
            <td class="num">{d['threshold']}</td>
            <td class="num">{diff_txt}</td>
            <td class="ctr">{badge}</td>
        </tr>
        """)
    st.markdown(f"""
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
    """, unsafe_allow_html=True)

# =========================================================
# แท็บ 2: กราฟรายละเอียดโรค (คลีนตามภาพตัวอย่าง)
# =========================================================
with tab2:
    d_map = {f"{d['disease_code']} - {d['disease_name']}": d for d in prov["all_diseases"]}
    sel_d = st.selectbox("เลือกโรคที่ต้องการดูแนวโน้ม", list(d_map.keys()), label_visibility="collapsed")
    d = d_map[sel_d]

    st_color = "#be123c" if d["is_flagged"] else "#15803d"
    st_label = "เกินเกณฑ์ (ชี้เป้า)" if d["is_flagged"] else "ต่ำกว่าเกณฑ์ (ปกติ)"
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-lbl">ยอดจริงสัปดาห์ล่าสุด</div>
            <div class="kpi-val">{d['latest_cases']} ราย</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">พยากรณ์สัปดาห์หน้า ({d['target_week']})</div>
            <div class="kpi-val" style="color:{st_color};">{d['forecast']} ราย</div>
            <div class="kpi-note">ช่วง 95%: {d['pi_low']} – {d['pi_high']} ราย</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">เกณฑ์เตือนรายโรค</div>
            <div class="kpi-val">{d['threshold']} ราย</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">สถานะเทียบเกณฑ์</div>
            <div class="kpi-val" style="font-size:16px; color:{st_color}; margin-top:6px;">{st_label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    hist = pd.DataFrame(d["history"])
    hist["cases_num"] = pd.to_numeric(hist["cases"], errors="coerce")

    fig = go.Figure()
    prelim = hist[hist["status"].str.contains("เบื้องต้น", na=False)]
    if not prelim.empty:
        fig.add_vrect(
            x0=prelim["week"].iloc[0], x1=prelim["week"].iloc[-1],
            fillcolor="rgba(245, 158, 11, 0.08)", line_width=0,
            annotation_text="ข้อมูลเบื้องต้น", annotation_position="top left",
            annotation_font=dict(color="#b45309", size=11)
        )

    fig.add_trace(go.Scatter(
        x=hist["week"], y=hist["cases_num"], mode="lines+markers", name="ยอดจริง",
        line=dict(color="#0066cc", width=2.2), marker=dict(size=6, color="#0066cc"),
        hovertemplate="สัปดาห์ %{x|%d %b %Y}<br>ยอดจริง: %{y} ราย<extra></extra>"
    ))

    valid = hist.dropna(subset=["cases_num"])
    if not valid.empty:
        fig.add_trace(go.Scatter(
            x=[valid["week"].iloc[-1], d["target_week"]],
            y=[valid["cases_num"].iloc[-1], d["forecast"]],
            mode="lines", line=dict(color="#ef4444", width=1.8, dash="dash"),
            showlegend=False, hoverinfo="skip"
        ))

    fig.add_trace(go.Scatter(
        x=[d["target_week"]], y=[d["forecast"]], mode="lines+markers", name="พยากรณ์สัปดาห์หน้า",
        line=dict(color="#ef4444", width=1.8, dash="dash"),
        marker=dict(color="#ef4444", size=12, symbol="star"),
        hovertemplate=f"สัปดาห์ {d['target_week']}<br>พยากรณ์: {d['forecast']} ราย<br>ช่วง 95%: {d['pi_low']}–{d['pi_high']} ราย<extra></extra>"
    ))

    fig.add_hline(
        y=d["threshold"], line_dash="dot", line_color="#f59e0b", line_width=2,
        annotation_text=f"เกณฑ์เตือน ({d['threshold']} ราย) ", annotation_position="top right",
        annotation_font=dict(color="#64748b", size=12)
    )

    y_top = max(float(valid["cases_num"].max() if not valid.empty else 0), float(d["threshold"]), float(d["forecast"]), 1.0) * 1.18
    fig.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", hovermode="x unified",
        legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.02, font=dict(size=12.5)),
        font=dict(family="IBM Plex Sans Thai, sans-serif", size=12, color="#64748b"),
        margin=dict(l=35, r=155, t=25, b=35), height=370
    )
    fig.update_xaxes(showgrid=False, tickformat="%b %Y", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False, range=[-y_top * 0.06, y_top])

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="hint-text">เส้นสีน้ำเงิน = ยอดจริงย้อนหลัง 1 ปี • แถบเหลืองอ่อน = ช่วงข้อมูลเบื้องต้น • ดาวสีแดง = ยอดพยากรณ์สัปดาห์หน้า</div>', unsafe_allow_html=True)

# =========================================================
# แท็บ 3: ติดตามผลและประเมินรายเดือน
# =========================================================
with tab3:
    reg_df = load_registry(prov["province_code"])
    if reg_df.empty:
        st.info("ยังไม่มีรายการในทะเบียนของจังหวัดนี้")
    else:
        # 1) สรุปตัวชี้วัดรายเดือน 4 ตัวแบบกะทัดรัด
        reg_df["month"] = reg_df["target_week"].astype(str).str.slice(0, 7)
        c_title, c_month = st.columns([3, 1])
        c_title.markdown('<div class="sec-label">ผลประเมินระบบติดตามผลรายเดือน</div>', unsafe_allow_html=True)
        sel_m = c_month.selectbox(
            "เลือกเดือน", ["ทั้งหมด"] + sorted(reg_df["month"].unique().tolist(), reverse=True),
            label_visibility="collapsed"
        )

        ev = reg_df if sel_m == "ทั้งหมด" else reg_df[reg_df["month"] == sel_m]
        ev["settled_num"] = pd.to_numeric(ev["settled_actual_cases"], errors="coerce")
        closed = ev[ev["settled_num"].notna()].copy()

        if closed.empty:
            st.markdown('<div class="hint-text" style="margin-bottom:12px;">ยังไม่มีรอบที่กรอก "ยอดจริงเมื่อข้อมูลนิ่ง" ในช่วงเวลาที่เลือก</div>', unsafe_allow_html=True)
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

            st.markdown(f"""
            <div class="kpi-row">
                <div class="kpi-card"><div class="kpi-lbl">1. ความแม่นของสัญญาณ</div><div class="kpi-val">{ratio_text(tp, tp+fp)}</div><div class="kpi-note">เตือนแล้วเข้าเกณฑ์จริง</div></div>
                <div class="kpi-card"><div class="kpi-lbl">2. ความครอบคลุม</div><div class="kpi-val">{ratio_text(tp, tp+fn)}</div><div class="kpi-note">จับเหตุการณ์จริงได้</div></div>
                <div class="kpi-card"><div class="kpi-lbl">3. ภาระตรวจสอบ</div><div class="kpi-val">รวม {tp+fp} (เกิน {fp})</div><div class="kpi-note">สัญญาณทั้งหมด / แจ้งเกิน</div></div>
                <div class="kpi-card"><div class="kpi-lbl">4. ความเร็วเริ่มตรวจ</div><div class="kpi-val">{spd}</div><div class="kpi-note">จากส่งสัญญาณถึงเริ่มตรวจ</div></div>
            </div>
            """, unsafe_allow_html=True)

        # 2) ฟอร์มบันทึกผลการตรวจสอบ (กะทัดรัด ล็อกค่าพยากรณ์และเกณฑ์เดิมอัตโนมัติ)
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
                try:
                    cur_act = str(int(float(raw_act))) if raw_act != "" else ""
                except ValueError:
                    cur_act = ""
                in_actual = c4.text_input("ยอดจริงนิ่งแล้ว", value=cur_act, placeholder="ตัวเลข")

                if st.form_submit_button("บันทึกผล", type="primary"):
                    actual_val = None
                    if in_actual.strip() != "":
                        try:
                            actual_val = float(in_actual.strip())
                            if actual_val < 0:
                                st.error("ยอดผู้ป่วยจริงต้องไม่ติดลบ")
                                st.stop()
                        except ValueError:
                            st.error("กรุณากรอกยอดจริงเป็นตัวเลข")
                            st.stop()
                    update_registry(
                        rec["run_id"], rec["province_code"], rec["disease_code"], rec["target_week"],
                        in_time.strip(), in_field, in_action.strip(), actual_val
                    )
                    st.success("บันทึกเรียบร้อยแล้ว")
                    st.rerun()

        # 3) ตารางทะเบียนติดตามผลแบบคลีน
        st.markdown('<div class="sec-label">ทะเบียนติดตามผล (คงค่าพยากรณ์และเกณฑ์เดิมไว้เทียบผลจริง)</div>', unsafe_allow_html=True)
        reg_rows = []
        for _, r in reg_df.iterrows():
            f_badge = '<span class="badge-red">ชี้เป้า</span>' if str(r["is_flagged"]).lower() == "true" else '<span class="badge-ok">ปกติ</span>'
            act_s = clean_str(r["settled_actual_cases"])
            act_txt = f"<b>{int(float(act_s))} ราย</b>" if act_s != "" else '<span style="color:#94a3b8;">รอยอดนิ่ง</span>'
            reg_rows.append(f"""
            <tr>
                <td><b>{r['run_id']}</b><span class="muted">เป้าหมาย {r['target_week']}</span></td>
                <td><b>{r['disease_name']}</b></td>
                <td class="num"><b>{r['frozen_forecast']}</b><span class="muted">เกณฑ์ {r['frozen_threshold']}</span></td>
                <td class="ctr">{f_badge}</td>
                <td>{clean_str(r['check_start_time']) or '-'}</td>
                <td>{clean_str(r['field_result']) or '-'}</td>
                <td>{clean_str(r['action_taken']) or '-'}</td>
                <td class="num">{act_txt}</td>
            </tr>
            """)
        st.markdown(f"""
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
        """, unsafe_allow_html=True)

# =========================================================
# แท็บ 4: เปรียบเทียบโมเดลย้อนหลัง
# =========================================================
with tab4:
    st.markdown(
        '<div class="hint-text" style="margin-bottom:10px;">ทดสอบย้อนหลังในปีที่ 3 แบบเลื่อนทีละสัปดาห์ (ใช้ข้อมูลถึงวันตัดรอบเท่านั้น) • <b>MAE</b> ยิ่งน้อยยิ่งทายแม่น • <b>เตือนถูก</b> ยิ่งมากยิ่งดี • <b>เตือนเกิน/พลาด</b> ยิ่งน้อยยิ่งดี</div>',
        unsafe_allow_html=True
    )
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
                    <td><b>{r['ชื่อโรค']}</b><span class="muted">เกณฑ์ {r['เกณฑ์ชี้เป้า']} ราย ({r['สัปดาห์ทดสอบ']} สัปดาห์)</span></td>
                    <td>{r['โมเดล']}{tag}</td>
                    <td class="num"><b>{r['MAE']}</b></td>
                    <td class="num" style="color:#15803d; font-weight:600;">{r['จับได้ (TP)']}</td>
                    <td class="num" style="color:#b45309;">{r['แจ้งเกิน (FP)']}</td>
                    <td class="num" style="color:#be123c;">{r['พลาด (FN)']}</td>
                    <td class="num">{r['ความแม่นของสัญญาณ']}</td>
                    <td class="num">{r['ความครอบคลุม']}</td>
                </tr>
                """)
            st.markdown(f"""
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
            """, unsafe_allow_html=True)
