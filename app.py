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
st.set_page_config(page_title="ระบบชี้เป้าโรคเฝ้าระวัง DDS", page_icon="🩺", layout="wide")

# ปรับฟอนต์ภาษาไทยและพื้นหลังโทนสว่างสบายตา
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', sans-serif !important; }
.stApp { background-color: #f7f9fb; }
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
# ส่วนที่ 1: ระบบเก็บทะเบียนติดตามผล (PostgreSQL ออนไลน์ / CSV สำรอง)
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
    """บันทึกรอบพยากรณ์ครบทุกโรค โดยห้ามเขียนทับค่าพยากรณ์และเกณฑ์ฉบับแรก"""
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
# ส่วนที่ 2: โหลดไฟล์ยอดรวมและคำนวณพยากรณ์เมื่อเปลี่ยนวันที่
# =========================================================
@st.cache_data
def load_files():
    if not os.path.exists(JSON_FILE) or not os.path.exists(WEEKLY_FILE):
        st.error("ไม่พบไฟล์ dashboard_data.json หรือ weekly_summary.csv กรุณาอัปโหลดให้ครบ 5 ไฟล์")
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
                    f"คาดว่าผู้ป่วยจะอยู่ที่ {round(fc,1)} ราย สูงกว่าเกณฑ์เตือน {th} ราย"
                    if flagged else
                    f"คาดว่าผู้ป่วยจะอยู่ที่ {round(fc,1)} ราย ยังต่ำกว่าเกณฑ์เตือน {th} ราย"
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
    return f"{num}/{den} ({num/den:.1%})" if den > 0 else f"{num}/0 (ไม่มีตัวหาร)"


# =========================================================
# ส่วนที่ 3: แถบควบคุมด้านซ้าย (เลือกจังหวัด + เปลี่ยนวันตัดข้อมูล)
# =========================================================
latest_json, weekly_df = load_files()
min_date = (weekly_df["week_start"].min() + pd.Timedelta(weeks=4)).date()
max_date = pd.to_datetime(latest_json["meta"]["cutoff_date"]).date()

with st.sidebar:
    st.header("ตัวเลือกการแสดงผล")
    picked_date = st.date_input(
        "วันตัดข้อมูล (เลือกย้อนหลังเพื่อจำลองรอบได้)",
        value=max_date, min_value=min_date, max_value=max_date
    )
    picked_str = picked_date.strftime("%Y-%m-%d")

    payload = latest_json if picked_str == latest_json["meta"]["cutoff_date"] else predict_for_date(weekly_df, picked_str)
    sync_to_registry(payload)

    prov_map = {
        f"จ.{p['province_name']} ({p['province_code']}) — ชี้เป้า {p['alert_count']} โรค": p
        for p in payload["provinces"]
    }
    prov_label = st.selectbox("เลือกจังหวัด", list(prov_map.keys()))
    prov = prov_map[prov_label]

    st.divider()
    with st.expander("คู่มืออ่านผลแบบภาษาคน"):
        st.write("- **เกณฑ์เตือน:** ระดับผู้ป่วยปกติของโรคนั้น หากคาดว่าจะเกินเส้นนี้ระบบจะเตือน")
        st.write("- **ช่วงพยากรณ์ 95%:** ช่วงต่ำสุด–สูงสุดที่ยอดจริงมีโอกาสเกิดขึ้น")
        st.write("- **ข้อมูลเบื้องต้น:** ยอด 1–2 สัปดาห์ล่าสุดที่อาจยังมีรายงานค้างส่ง")
    st.caption("ฐานข้อมูลทะเบียน: " + ("เชื่อมต่อ PostgreSQL ออนไลน์แล้ว" if get_db_engine() else "โหมดไฟล์ CSV สาธิต"))

meta = payload["meta"]

# ส่วนหัวหน้าเว็บ
st.title(f"ระบบชี้เป้าโรคที่ควรเฝ้าระวังสัปดาห์หน้า — จ.{prov['province_name']}")
h1, h2, h3, h4 = st.columns(4)
h1.metric("รหัสรอบรายงาน", meta["run_id"])
h2.metric("สัปดาห์เป้าหมาย", meta["target_week"])
h3.metric("วันที่ตัดข้อมูล", meta["cutoff_date"])
h4.metric("สถานะข้อมูล", meta["data_status"])

tab1, tab2, tab3, tab4 = st.tabs([
    "1. ชี้เป้าสัปดาห์หน้า",
    "2. กราฟรายละเอียดโรค",
    "3. บันทึกและติดตามผล",
    "4. เปรียบเทียบโมเดล"
])

# =========================================================
# หน้า 1: ชี้เป้าสัปดาห์หน้า
# =========================================================
with tab1:
    alerts = prov["alerts"]
    if not alerts:
        st.success(f"สรุป จ.{prov['province_name']}: สัปดาห์เป้าหมาย {meta['target_week']} ยังไม่มีโรคที่คาดว่าจะเกินเกณฑ์เตือน สถานการณ์ทั้ง 3 โรคอยู่ในระดับปกติ")
    else:
        names = " และ ".join([a["disease_name"] for a in alerts])
        st.warning(f"สรุป จ.{prov['province_name']}: สัปดาห์เป้าหมาย {meta['target_week']} พบ {len(alerts)} โรคที่ควรตรวจสอบก่อน ได้แก่ {names}")
        cols = st.columns(len(alerts))
        for i, item in enumerate(alerts):
            diff = round(item["forecast"] - item["threshold"], 1)
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"**ควรตรวจก่อนอันดับ #{i+1} • รหัส {item['disease_code']}**")
                    st.subheader(item["disease_name"])
                    st.metric("ยอดคาดสัปดาห์หน้า", f"{item['forecast']} ราย", f"+{diff} ราย จากเกณฑ์ {item['threshold']}", delta_color="inverse")
                    st.write(f"ยอดสัปดาห์ล่าสุด: **{item['latest_cases']} ราย**")
                    st.write(f"ช่วงพยากรณ์ 95%: **{item['pi_low']} – {item['pi_high']} ราย**")
                    st.caption(f"เหตุผล: {item['reason']}")

    st.subheader("ตารางสรุปครบทุกโรคในขอบเขต (รวมโรคที่สถานะปกติ)")
    ov_df = pd.DataFrame([{
        "รหัสโรค": d["disease_code"],
        "ชื่อโรค": d["disease_name"],
        "ยอดล่าสุด (ราย)": d["latest_cases"],
        "ยอดคาดสัปดาห์หน้า (ราย)": d["forecast"],
        "ช่วงพยากรณ์ 95%": f"{d['pi_low']} – {d['pi_high']}",
        "เกณฑ์เตือน (ราย)": d["threshold"],
        "เทียบกับเกณฑ์": f"+{round(d['forecast']-d['threshold'],1)} (เกินเกณฑ์)" if d["forecast"] >= d["threshold"] else f"{round(d['forecast']-d['threshold'],1)} (ปกติ)",
        "คำแนะนำ": "🔴 ควรตรวจสอบก่อน" if d["is_flagged"] else "🟢 ปกติ"
    } for d in prov["all_diseases"]])
    st.dataframe(ov_df, use_container_width=True, hide_index=True)

# =========================================================
# หน้า 2: กราฟรายละเอียดโรค (สไตล์คลีนตามภาพตัวอย่าง)
# =========================================================
with tab2:
    d_map = {f"{d['disease_code']} - {d['disease_name']}": d for d in prov["all_diseases"]}
    d = d_map[st.selectbox("เลือกโรคที่ต้องการดูแนวโน้มย้อนหลัง 1 ปี", list(d_map.keys()))]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ยอดจริงสัปดาห์ล่าสุด", f"{d['latest_cases']} ราย")
    k2.metric("พยากรณ์สัปดาห์หน้า", f"{d['forecast']} ราย")
    k3.metric("เกณฑ์เตือนรายโรค", f"{d['threshold']} ราย")
    k4.metric("สถานะ", "🔴 เกินเกณฑ์" if d["is_flagged"] else "🟢 ปกติ")

    hist = pd.DataFrame(d["history"])
    hist["cases_num"] = pd.to_numeric(hist["cases"], errors="coerce")

    fig = go.Figure()
    prelim = hist[hist["status"].str.contains("เบื้องต้น", na=False)]
    if not prelim.empty:
        fig.add_vrect(
            x0=prelim["week"].iloc[0], x1=prelim["week"].iloc[-1],
            fillcolor="rgba(245,158,11,0.09)", line_width=0,
            annotation_text="ข้อมูลเบื้องต้น", annotation_position="top left"
        )

    fig.add_trace(go.Scatter(
        x=hist["week"], y=hist["cases_num"], mode="lines+markers", name="ยอดจริง",
        line=dict(color="#0066cc", width=2.2), marker=dict(size=6.5, color="#0066cc")
    ))

    valid = hist.dropna(subset=["cases_num"])
    if not valid.empty:
        fig.add_trace(go.Scatter(
            x=[valid["week"].iloc[-1], d["target_week"]],
            y=[valid["cases_num"].iloc[-1], d["forecast"]],
            mode="lines", line=dict(color="#ef4444", width=2, dash="dash"),
            showlegend=False, hoverinfo="skip"
        ))

    fig.add_trace(go.Scatter(
        x=[d["target_week"]], y=[d["forecast"]], mode="lines+markers", name="พยากรณ์สัปดาห์หน้า",
        line=dict(color="#ef4444", width=2, dash="dash"),
        marker=dict(color="#ef4444", size=13, symbol="star"),
        hovertemplate=f"สัปดาห์ {d['target_week']}<br>พยากรณ์: {d['forecast']} ราย<br>ช่วง 95%: {d['pi_low']}–{d['pi_high']} ราย<extra></extra>"
    ))

    fig.add_hline(
        y=d["threshold"], line_dash="dot", line_color="#f59e0b", line_width=2.2,
        annotation_text=f"เกณฑ์เตือน ({d['threshold']} ราย) ", annotation_position="top right"
    )

    y_top = max(float(valid["cases_num"].max() if not valid.empty else 0), float(d["threshold"]), float(d["forecast"]), 1.0) * 1.18
    fig.update_layout(
        title=f"จ.{prov['province_name']} — {d['disease_name']}",
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", hovermode="x unified",
        legend=dict(yanchor="top", y=1.0, xanchor="left", x=1.02),
        margin=dict(l=40, r=160, t=50, b=40), height=400
    )
    fig.update_xaxes(showgrid=False, tickformat="%b %Y", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False, range=[-y_top * 0.06, y_top])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("วิธีอ่านกราฟ: เส้นสีน้ำเงิน = ยอดจริงในอดีต | แถบเหลืองอ่อน = ช่วงข้อมูลเบื้องต้น | ดาวสีแดง = พยากรณ์สัปดาห์หน้า (ถ้าอยู่เหนือเส้นประสีส้มแปลว่าควรเฝ้าระวัง)")

# =========================================================
# หน้า 3: บันทึกและติดตามผล + ตัวชี้วัดรายเดือน 4 ตัว
# =========================================================
with tab3:
    reg_df = load_registry(prov["province_code"])
    if reg_df.empty:
        st.info("ยังไม่มีรายการในทะเบียนของจังหวัดนี้")
    else:
        reg_df["label"] = reg_df["run_id"] + " | " + reg_df["disease_code"] + " - " + reg_df["disease_name"] + " | สัปดาห์ " + reg_df["target_week"]
        rec = reg_df[reg_df["label"] == st.selectbox("เลือกรอบรายงานและโรคที่ต้องการบันทึกผล", reg_df["label"].tolist())].iloc[0]

        with st.form("track_form"):
            c1, c2 = st.columns(2)
            in_time = c1.text_input(
                "เวลาที่เริ่มตรวจสอบ (YYYY-MM-DD HH:MM)",
                value="" if pd.isna(rec["check_start_time"]) else str(rec["check_start_time"]),
                placeholder="เช่น 2025-12-31 10:30"
            )
            f_opts = ["", "พบกลุ่มก้อน", "พบการส่งย้อนหลัง", "ยังสรุปไม่ได้", "ปกติ"]
            cur_f = "" if pd.isna(rec["field_result"]) else str(rec["field_result"])
            in_field = c2.selectbox("ผลตรวจภาคสนาม (แยกจากสถานะยอดเข้าเกณฑ์)", f_opts, index=f_opts.index(cur_f) if cur_f in f_opts else 0)

            c3, c4 = st.columns(2)
            in_action = c3.text_input(
                "การดำเนินการของทีม",
                value="" if pd.isna(rec["action_taken"]) else str(rec["action_taken"]),
                placeholder="เช่น แจ้งทีม SRRT ตรวจสอบพื้นที่"
            )
            raw_act = "" if pd.isna(rec["settled_actual_cases"]) else str(rec["settled_actual_cases"]).strip()
            try:
                cur_act = str(int(float(raw_act))) if raw_act != "" else ""
            except ValueError:
                cur_act = ""
            in_actual = c4.text_input("ยอดผู้ป่วยจริงเมื่อข้อมูลนิ่งแล้ว (ราย)", value=cur_act, placeholder="เว้นว่างไว้หากข้อมูลยังไม่นิ่ง")

            if st.form_submit_button("บันทึกผลการตรวจสอบ", type="primary"):
                actual_val = None
                if in_actual.strip() != "":
                    try:
                        actual_val = float(in_actual.strip())
                        if actual_val < 0:
                            st.error("ยอดผู้ป่วยจริงต้องไม่ติดลบ")
                            st.stop()
                    except ValueError:
                        st.error("กรุณากรอกยอดผู้ป่วยจริงเป็นตัวเลข")
                        st.stop()
                update_registry(
                    rec["run_id"], rec["province_code"], rec["disease_code"], rec["target_week"],
                    in_time.strip(), in_field, in_action.strip(), actual_val
                )
                st.success("บันทึกเรียบร้อยแล้ว (ค่าพยากรณ์และเกณฑ์เดิมถูกล็อกไว้ตามต้นฉบับ)")
                st.rerun()

        st.divider()
        st.subheader("ผลประเมินระบบติดตามผลรายเดือน (4 ตัวชี้วัด)")
        reg_df["month"] = reg_df["target_week"].astype(str).str.slice(0, 7)
        sel_m = st.selectbox("เลือกเดือนที่ต้องการประเมิน", ["ทั้งหมด"] + sorted(reg_df["month"].unique().tolist(), reverse=True))
        ev = reg_df if sel_m == "ทั้งหมด" else reg_df[reg_df["month"] == sel_m]
        ev["settled_num"] = pd.to_numeric(ev["settled_actual_cases"], errors="coerce")
        closed = ev[ev["settled_num"].notna()].copy()

        if closed.empty:
            st.info("ยังไม่มีรายการที่กรอก 'ยอดผู้ป่วยจริงเมื่อข้อมูลนิ่งแล้ว' ในเดือนที่เลือก")
        else:
            flagged = closed["is_flagged"].astype(str).str.lower() == "true"
            actual_met = closed["settled_num"] >= pd.to_numeric(closed["frozen_threshold"], errors="coerce")
            tp = int((flagged & actual_met).sum())
            fp = int((flagged & ~actual_met).sum())
            fn = int((~flagged & actual_met).sum())
            avg_hr = ((pd.to_datetime(closed["check_start_time"], errors="coerce") - pd.to_datetime(closed["alert_sent_time"], errors="coerce")).dt.total_seconds() / 3600).dropna().mean()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("1. ความแม่นของสัญญาณ", ratio_text(tp, tp + fp), help="เตือนแล้วพุ่งเกินเกณฑ์จริงกี่ครั้ง")
            m2.metric("2. ความครอบคลุม", ratio_text(tp, tp + fn), help="ครั้งที่พุ่งเกินเกณฑ์จริง ระบบจับได้กี่ครั้ง")
            m3.metric("3. ภาระตรวจสอบ", f"รวม {tp+fp} (แจ้งเกิน {fp})", help="จำนวนสัญญาณทั้งหมดและที่แจ้งเกิน")
            m4.metric("4. ความเร็วในการตรวจ", f"{avg_hr:.1f} ชม." if pd.notna(avg_hr) else "ยังไม่ระบุเวลา")

        st.subheader("ทะเบียนติดตามผลทั้งหมด")
        st.dataframe(reg_df[[
            "run_id", "disease_name", "target_week", "frozen_forecast", "frozen_threshold",
            "is_flagged", "check_start_time", "field_result", "action_taken", "settled_actual_cases"
        ]].rename(columns={
            "run_id": "รหัสรอบ", "disease_name": "ชื่อโรค", "target_week": "สัปดาห์เป้าหมาย",
            "frozen_forecast": "ยอดคาด (คงเดิม)", "frozen_threshold": "เกณฑ์ (คงเดิม)", "is_flagged": "ชี้เป้า",
            "check_start_time": "เวลาเริ่มตรวจ", "field_result": "ผลภาคสนาม",
            "action_taken": "การดำเนินการ", "settled_actual_cases": "ยอดจริงนิ่งแล้ว"
        }), use_container_width=True, hide_index=True)

# =========================================================
# หน้า 4: เปรียบเทียบโมเดลย้อนหลัง
# =========================================================
with tab4:
    st.info("วิธีอ่านตาราง: ระบบฝึกจากข้อมูล 2 ปีแรกแล้วลองทายปีที่ 3 ทีละสัปดาห์ • **MAE** ยิ่งน้อยยิ่งทายแม่น • **เตือนถูก** ยิ่งเยอะยิ่งดี • **เตือนเกิน/พลาด** ยิ่งน้อยยิ่งดี")
    if os.path.exists(BACKTEST_FILE):
        bt = pd.read_csv(BACKTEST_FILE, dtype={"รหัสจังหวัด": str, "รหัสโรค": str})
        bt["รหัสจังหวัด"] = bt["รหัสจังหวัด"].str.strip().str.replace(r"\.0$", "", regex=True)
        bt = bt[bt["รหัสจังหวัด"] == str(prov["province_code"])].copy()
        if bt.empty:
            st.info("ไม่มีผลทดสอบย้อนหลังสำหรับจังหวัดนี้")
        else:
            bt["หมายเหตุ"] = np.where(bt["MAE"] == bt.groupby("รหัสโรค")["MAE"].transform("min"), "⭐ แม่นสุด", "")
            st.dataframe(bt[[
                "ชื่อโรค", "โมเดล", "MAE", "จับได้ (TP)", "แจ้งเกิน (FP)",
                "พลาด (FN)", "ความแม่นของสัญญาณ", "ความครอบคลุม", "หมายเหตุ"
            ]].rename(columns={
                "โมเดล": "วิธีคำนวณ", "MAE": "ทายพลาดเฉลี่ย (ราย/สัปดาห์)",
                "จับได้ (TP)": "เตือนถูก", "แจ้งเกิน (FP)": "เตือนเกิน", "พลาด (FN)": "พลาดสัญญาณ"
            }), use_container_width=True, hide_index=True)
            st.caption("ข้อจำกัด: หากชุดข้อมูลทดสอบเป็นไฟล์ปรับย้อนหลัง ยอดจริงอาจสูงกว่ายอดที่เห็น ณ วันพยากรณ์จริงเนื่องจากความล่าช้าในการส่งรายงาน")
