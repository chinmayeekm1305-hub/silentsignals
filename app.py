"""SilentSignal - hackathon prototype (Track 01 AI, PS-01).

Run:  streamlit run app.py
"""
import os
from datetime import datetime, timedelta
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import engine
from ai import analyse, get_key

BASE = Path(__file__).parent
DATA = BASE / "data"
PHOTOS = DATA / "photos"
DECISIONS = DATA / "decisions.csv"

st.set_page_config(page_title="SilentSignal", page_icon="📡", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 3rem;}
.big-title {font-size: 2.1rem; font-weight: 800; margin-bottom: 0;}
.tag {color: #6b7280; font-style: italic; margin-top: 0;}
.notice {background:#fff4e0; border-radius:8px; padding:8px 12px; font-size:0.9rem;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; color:white; font-weight:600; font-size:0.8rem;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data
def load_base():
    reports = pd.read_csv(DATA / "reports.csv")
    areas = pd.read_csv(DATA / "areas.csv")
    return reports, areas


base_reports, areas = load_base()
if "live_reports" not in st.session_state:
    st.session_state.live_reports = []

all_reports = pd.concat([base_reports, pd.DataFrame(st.session_state.live_reports)], ignore_index=True) \
    if st.session_state.live_reports else base_reports
reports = engine.prepare_reports(all_reports)
NOW = pd.to_datetime(base_reports["time"]).max() + timedelta(minutes=10)   # scenario clock

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Control room")
    regions = ["All regions"] + sorted(areas["region"].unique())
    region = st.selectbox("Disaster region", regions)
    show_silent = st.toggle("Show Silent Zones", value=True)
    st.caption(f"Scenario time: **{NOW:%d %b %Y, %H:%M}**")
    st.subheader("Available resources")
    res = {
        "Boat": st.number_input("Boats", 0, 50, 3),
        "Rescue team": st.number_input("Rescue teams", 0, 50, 6),
        "Ambulance": st.number_input("Ambulances", 0, 50, 2),
        "JCB": st.number_input("JCBs / road clearing", 0, 50, 2),
        "Relief team": st.number_input("Relief teams (food / shelter)", 0, 50, 2),
        "Drone": st.number_input("Drones", 0, 20, 3),
    }
    st.divider()
    st.caption("Decision support only. The final decision is taken by the authorised officer.")

if region != "All regions":
    reports_v = reports[reports["region"] == region]
    areas_v = areas[areas["region"] == region]
else:
    reports_v, areas_v = reports, areas

incidents = engine.group_incidents(reports_v)
zones = engine.silent_zones(areas_v, reports_v, incidents, NOW)
plan, left = engine.match_resources(incidents, zones if show_silent else zones.iloc[0:0], res)

# ------------------------------------------------------------------ header
st.markdown('<p class="big-title">📡 SilentSignal</p>', unsafe_allow_html=True)
st.markdown('<p class="tag">Every system ranks the people shouting for help. We also find the people who can\'t.</p>',
            unsafe_allow_html=True)
st.markdown('<div class="notice">⚠️ <b>Decision support only.</b> AI suggestions must be approved by the authorised officer. '
            'Scores are not proof; yellow and purple items need verification.</div>', unsafe_allow_html=True)
st.write("")

c = st.columns(5)
c[0].metric("Raw reports", len(reports_v))
c[1].metric("Incidents (after merging)", len(incidents))
c[2].metric("Critical (80+)", int((incidents["score"] >= 80).sum()))
c[3].metric("Needs checking 🟡", int(incidents["unsure"].sum()))
c[4].metric("Silent Zones 🟣", int(zones["silent"].sum()) if show_silent else 0)

# ------------------------------------------------------------------ map
def build_map():
    pts = list(zip(incidents["lat"], incidents["lon"])) + list(zip(areas_v["lat"], areas_v["lon"]))
    m = folium.Map(location=[pts[0][0], pts[0][1]], zoom_start=12, tiles="OpenStreetMap")
    if show_silent:
        for _, z in zones[zones["silent"]].iterrows():
            folium.Circle([z["lat"], z["lon"]], radius=900, color="#7c5cff", weight=3, dash_array="8",
                          fill=True, fill_color="#7c5cff", fill_opacity=0.25,
                          tooltip=f"🟣 SILENT ZONE: {z['area']} (danger expected {z['danger_expected']})",
                          popup=folium.Popup(f"<b>Silent Zone: {z['area']}</b><br>{z['reason']}<br><i>Please verify - send drone or patrol.</i>", max_width=300)
                          ).add_to(m)
    for rank, r in incidents.iterrows():
        tip = f"#{rank+1} · {r['area']} · {r['disaster']} · level {r['level']} · score {r['score']}"
        if r["unsure"]:
            tip += " · 🟡 please check"
        html = (f"<b>#{rank+1} {r['area']}</b> ({r['disaster']})<br>Level {r['level']} {engine.LEVEL_NAMES[r['level']]} · "
                f"score {r['score']}<br>{r['measurement']}<br>Reports merged: {r['n_reports']}<br>"
                f"Help: {', '.join(r['help_needed'])}")
        folium.CircleMarker([r["lat"], r["lon"]], radius=6 + r["level"] * 2, color="white", weight=2,
                            fill=True, fill_color=r["color"], fill_opacity=0.95, tooltip=tip,
                            popup=folium.Popup(html, max_width=280)).add_to(m)
    m.fit_bounds([[min(p[0] for p in pts) - 0.01, min(p[1] for p in pts) - 0.01],
                  [max(p[0] for p in pts) + 0.01, max(p[1] for p in pts) + 0.01]])
    return m


left_col, right_col = st.columns([3, 2])
with left_col:
    st.subheader("🗺️ Situation map")
    st_folium(build_map(), height=470, use_container_width=True, returned_objects=[])
    st.caption("🔴 Critical · 🟠 High · 🟡 Medium / needs checking · 🟢 Low · 🟣 Silent Zone (dashed purple area)")
with right_col:
    st.subheader("🚑 Suggested resource plan")
    st.dataframe(plan.head(12), hide_index=True, use_container_width=True,
                 column_config={"id": None, "priority": st.column_config.NumberColumn("Score")})
    st.caption("Left over: " + ", ".join(f"{k} {v}" for k, v in left.items()))


# ------------------------------------------------------------------ tabs
def save_decision(case_id, label, suggestion, decision, reason):
    row = pd.DataFrame([{"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "case": case_id, "place": label,
                         "ai_suggestion": suggestion, "officer_decision": decision, "reason": reason}])
    row.to_csv(DECISIONS, mode="a", header=not DECISIONS.exists(), index=False)


tab1, tab2, tab3, tab4 = st.tabs(["📋 Priority list", "🟣 Silent Zones", "➕ Add live report (AI)", "📝 Decision log"])

with tab1:
    st.caption("Highest priority first. Open a case to see the evidence and take a decision.")
    for rank, r in incidents.iterrows():
        flag = " 🟡 PLEASE CHECK" if r["unsure"] else ""
        head = (f"#{rank+1} · Score {r['score']} ({r['band']}) · {r['area']} · {r['disaster'].title()} · "
                f"Level {r['level']} {engine.LEVEL_NAMES[r['level']]}{flag}")
        with st.expander(head, expanded=(rank == 0)):
            a, b = st.columns([3, 2])
            with a:
                st.markdown(f"**What the AI measured:** {r['measurement']}")
                if r["level_bumped"]:
                    st.markdown("**Level raised by 1:** people confirmed at risk with vulnerable people present.")
                people = r["people"] + (f" (about {int(r['people_count'])})" if pd.notna(r["people_count"]) else "")
                st.markdown(f"**People at risk:** {people}" + (f" · **Vulnerable:** {', '.join(r['vulnerable'])}" if r["vulnerable"] else ""))
                st.markdown(f"**Road blocked:** {r['road_blocked']} · **Critical in about:** {r['hours_to_critical']:.0f} h"
                            + (f" · **Hazards:** {', '.join(r['hazards'])}" if r["hazards"] else ""))
                st.markdown(f"**Suggested help:** {', '.join(r['help_needed'])}")
                st.markdown(f"**Evidence:** {r['n_reports']} report(s) from {r['sources']} "
                            f"({r['report_ids']}) · combined confidence **{int(r['confidence']*100)}%**")
                for t in r["texts"]:
                    st.markdown(f"> {t}")
                if r["unsure"]:
                    st.warning(f"Not sure - please verify: {r['unsure_reason']}")
                for p in r["photos"]:
                    if (PHOTOS / p).exists():
                        st.image(str(PHOTOS / p), width=260)
            with b:
                st.markdown("**Why this score**")
                st.dataframe(pd.DataFrame({
                    "Factor": ["Severity (35)", "People / exposure (30)", "Time pressure (15)", "Access difficulty (10)", "Confirmation (10)"],
                    "Points": [r["p_severity"], r["p_exposure"], r["p_time"], r["p_access"], r["p_confirm"]],
                }), hide_index=True, use_container_width=True)
                st.markdown("**Officer decision**")
                d = st.radio("Decision", ["Approve", "Change", "Send someone to check"], key=f"d_{r['incident_id']}",
                             horizontal=True, label_visibility="collapsed")
                reason = st.text_input("Reason (required for Change)", key=f"r_{r['incident_id']}")
                if st.button("Save decision", key=f"s_{r['incident_id']}"):
                    if d == "Change" and not reason.strip():
                        st.error("Please give a reason for changing the AI suggestion.")
                    else:
                        save_decision(r["incident_id"], r["area"], ", ".join(r["help_needed"]), d, reason)
                        st.success("Saved to decision log.")

with tab2:
    st.caption("Areas where danger is expected to be high but no messages are coming. Silence can also mean safe - always verify.")
    if not show_silent:
        st.info("Silent Zones are switched off in the sidebar.")
    for _, z in zones.iterrows():
        if z["silent"]:
            with st.container(border=True):
                st.markdown(f"### 🟣 {z['area']} - danger expected {z['danger_expected']}/100")
                st.markdown(f"{z['reason']}")
                st.markdown(f"**Population:** about {z['population']:,} · **Suggested:** send drone or patrol team to check")
                d = st.radio("Decision", ["Send drone / patrol", "Ignore for now", "Already verified safe"],
                             key=f"zd_{z['area']}", horizontal=True)
                if st.button("Save decision", key=f"zs_{z['area']}"):
                    save_decision(f"SZ-{z['area']}", z["area"], "Send drone / patrol", d, "")
                    st.success("Saved to decision log.")
    st.markdown("**All areas checked**")
    st.dataframe(zones[["area", "disaster", "danger_expected", "recent_msgs", "earlier_msgs", "silent"]],
                 hide_index=True, use_container_width=True)

with tab3:
    st.caption("Upload a real photo + message. Gemini reads both and the Damage Ruler scores it live.")
    key = get_key(st.secrets if (BASE / ".streamlit" / "secrets.toml").exists() else None)
    if not key:
        st.info("Add your free Gemini API key to `.streamlit/secrets.toml` as `GEMINI_API_KEY = \"...\"` to use live AI. "
                "The demo data above works without it.")
    with st.form("live"):
        photo = st.file_uploader("Photo", type=["jpg", "jpeg", "png"])
        msg = st.text_area("Message (any language)", placeholder="e.g. Water till chest, grandmother stuck upstairs")
        area_names = sorted(areas_v["area"].unique())
        area = st.selectbox("Area", area_names)
        a_row = areas[areas["area"] == area].iloc[0]
        lat = st.number_input("Latitude", value=float(a_row["lat"]), format="%.5f")
        lon = st.number_input("Longitude", value=float(a_row["lon"]), format="%.5f")
        go = st.form_submit_button("Analyse with AI", disabled=not key)
    if go:
        with st.spinner("Damage Ruler is reading the photo and message..."):
            try:
                model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
                out = analyse(photo.getvalue() if photo else None, photo.type if photo else None, msg, key, model)
                pid = f"LIVE{len(st.session_state.live_reports)+1:02d}"
                if photo:
                    PHOTOS.mkdir(parents=True, exist_ok=True)
                    (PHOTOS / f"{pid}.jpg").write_bytes(photo.getvalue())
                st.session_state.live_reports.append({
                    "report_id": pid, "time": NOW.strftime("%Y-%m-%d %H:%M"), "source": "Live upload",
                    "region": a_row["region"], "area": area, "lat": lat, "lon": lon, "language": "",
                    "text": out.get("english_text") or msg, "photo_file": f"{pid}.jpg" if photo else "",
                    "disaster_type": out.get("disaster_type", a_row["disaster"]), "measurement": out.get("measurement", ""),
                    "damage_level": out["damage_level"], "people_at_risk": out.get("people_at_risk", "possible"),
                    "people_count": out.get("people_count"), "vulnerable": ";".join(out["vulnerable"]),
                    "road_blocked": str(out.get("road_blocked", "unknown")).lower(), "hazards": ";".join(out["hazards"]),
                    "help_needed": ";".join(out["help_needed"]), "hours_to_critical": out.get("hours_to_critical", 6),
                    "photo_text_match": str(out.get("photo_text_match", True)).lower(), "confidence": out["confidence"],
                    "reason_if_unsure": out.get("reason_if_unsure", ""),
                })
                st.success("Added. The map, priority list and Silent Zones have been updated.")
                st.json(out)
                st.rerun()
            except Exception as e:
                st.error(f"AI call failed: {e}. Check your API key and internet, or continue with demo data.")

with tab4:
    if DECISIONS.exists():
        st.dataframe(pd.read_csv(DECISIONS).iloc[::-1], hide_index=True, use_container_width=True)
    else:
        st.caption("No decisions saved yet.")
