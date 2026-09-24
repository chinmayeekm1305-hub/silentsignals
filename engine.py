"""SilentSignal core logic.

Everything here is plain, explainable rules on top of the AI's answers:
- Damage Ruler final level (+1 for people / vulnerable, max 5)
- grouping duplicate reports into incidents
- priority score 0-100
- Silent Zones (high danger expected + no messages)
- resource matching
"""
import math
from datetime import timedelta

import pandas as pd

LEVEL_NAMES = {1: "Minor", 2: "Low", 3: "Serious", 4: "Severe", 5: "Critical"}

# Common danger scale shown to the officer (also pasted into the AI prompt)
DANGER_SCALE = {
    "flood": ["Ankle-deep (<0.3 m)", "Knee-deep (0.3-0.6 m)", "Waist-deep (0.6-1 m)",
              "Chest-deep (1-1.5 m)", "Above head / roof (>1.5 m)"],
    "cyclone": ["Small branches down", "Trees and poles down", "Roof damaged",
                "Roof blown off", "House destroyed"],
    "earthquake": ["Hairline cracks", "Big cracks in walls", "Part of building fallen",
                   "Building partly collapsed", "Building fully collapsed"],
    "landslide": ["Little mud on road", "Road partly blocked", "Road fully blocked",
                  "House partly buried", "House fully buried"],
}

CLUSTER_METERS = 250          # reports closer than this are treated as the same incident
NEIGHBOUR_KM = 4              # nearby damage radius for Silent Zones
RECENT_HOURS = 2              # "last 2 hours" window
SILENT_THRESHOLD = 60         # danger expected needed to call an area a Silent Zone
UNSURE_CONFIDENCE = 0.6


def km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def split_list(v):
    if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
        return []
    return [x.strip() for x in str(v).replace(",", ";").split(";") if x.strip()]


def band(score):
    if score >= 80:
        return "Critical", "#b3261e"
    if score >= 60:
        return "High", "#e9803a"
    if score >= 40:
        return "Medium", "#e8c33a"
    return "Low", "#2e9e6b"


# ---------------------------------------------------------------- reports
def prepare_reports(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["damage_level"] = pd.to_numeric(df["damage_level"], errors="coerce").fillna(3).astype(int)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.5)
    df["hours_to_critical"] = pd.to_numeric(df["hours_to_critical"], errors="coerce").fillna(12)
    df["people_count"] = pd.to_numeric(df["people_count"], errors="coerce")
    for col in ["vulnerable", "help_needed", "hazards"]:
        df[col] = df[col].apply(split_list)
    df["road_blocked"] = df["road_blocked"].astype(str).str.lower()
    df["photo_text_match"] = df["photo_text_match"].astype(str).str.lower() != "false"
    df["reason_if_unsure"] = df["reason_if_unsure"].fillna("")
    # Damage Ruler: +1 level if people confirmed at risk with vulnerable people present
    bump = (df["people_at_risk"] == "confirmed") & df["vulnerable"].apply(len).gt(0)
    df["final_level"] = (df["damage_level"] + bump.astype(int)).clip(1, 5)
    df["level_bumped"] = bump
    return df


def group_incidents(df: pd.DataFrame) -> pd.DataFrame:
    """Join reports about the same spot (within CLUSTER_METERS) into one incident."""
    groups, gid = [], {}
    for idx, r in df.sort_values("time").iterrows():
        placed = False
        for g in groups:
            if g["disaster"] == r["disaster_type"] and km(g["lat"], g["lon"], r["lat"], r["lon"]) * 1000 <= CLUSTER_METERS:
                g["rows"].append(idx)
                placed = True
                break
        if not placed:
            groups.append({"disaster": r["disaster_type"], "lat": r["lat"], "lon": r["lon"], "rows": [idx]})

    incidents = []
    for n, g in enumerate(groups, start=1):
        rows = df.loc[g["rows"]]
        best = rows.sort_values(["final_level", "confidence"], ascending=False).iloc[0]
        # confidence rises with independent sources: 1 - (1-c1)(1-c2)...
        miss = 1.0
        for c in rows["confidence"]:
            miss *= (1 - c)
        conf = round(1 - miss, 2)
        people_order = {"none": 0, "possible": 1, "confirmed": 2}
        people = max(rows["people_at_risk"], key=lambda p: people_order.get(p, 0))
        vulnerable = sorted({v for lst in rows["vulnerable"] for v in lst})
        help_needed = list(dict.fromkeys(h for lst in rows["help_needed"] for h in lst))
        hazards = sorted({h for lst in rows["hazards"] for h in lst})
        road = "true" if (rows["road_blocked"] == "true").any() else ("unknown" if (rows["road_blocked"] == "unknown").all() else "false")
        mismatch = not rows["photo_text_match"].all()
        unsure_reasons = [x for x in rows["reason_if_unsure"] if x]
        unsure = conf < UNSURE_CONFIDENCE or (mismatch and len(rows) == 1)
        incidents.append({
            "incident_id": f"INC-{n:02d}",
            "region": best["region"], "area": best["area"], "disaster": best["disaster_type"],
            "lat": rows["lat"].mean(), "lon": rows["lon"].mean(),
            "level": int(rows["final_level"].max()), "level_bumped": bool(best["level_bumped"]),
            "measurement": best["measurement"], "people": people,
            "people_count": rows["people_count"].max(), "vulnerable": vulnerable,
            "road_blocked": road, "hazards": hazards, "help_needed": help_needed,
            "hours_to_critical": float(rows["hours_to_critical"].min()),
            "confidence": conf, "n_reports": len(rows), "sources": ", ".join(sorted(set(rows["source"]))),
            "report_ids": ", ".join(rows["report_id"]), "texts": list(rows["text"]),
            "photos": [p for p in rows["photo_file"] if isinstance(p, str) and p],
            "unsure": unsure, "unsure_reason": "; ".join(unsure_reasons) or ("Low confidence" if unsure else ""),
            "last_time": rows["time"].max(),
        })
    inc = pd.DataFrame(incidents)
    return score_incidents(inc)


def score_incidents(inc: pd.DataFrame) -> pd.DataFrame:
    """Priority score 0-100 = 35% severity + 30% exposure + 15% time + 10% access + 10% confirmation."""
    def parts(r):
        s = r["level"] / 5
        e = {"none": 0.0, "possible": 0.5, "confirmed": 1.0}.get(r["people"], 0.3)
        if r["vulnerable"]:
            e = min(1.0, e + 0.3)
        h = r["hours_to_critical"]
        t = 1.0 if h <= 2 else 0.6 if h <= 6 else 0.3 if h <= 12 else 0.1
        a = 1.0 if r["road_blocked"] == "true" else 0.5 if r["road_blocked"] == "unknown" else 0.0
        c = min(r["n_reports"], 3) / 3
        return pd.Series({"p_severity": round(35 * s, 1), "p_exposure": round(30 * e, 1),
                          "p_time": round(15 * t, 1), "p_access": round(10 * a, 1), "p_confirm": round(10 * c, 1)})
    p = inc.apply(parts, axis=1)
    inc = pd.concat([inc, p], axis=1)
    inc["score"] = p.sum(axis=1).round(0).astype(int)
    inc["band"] = inc["score"].apply(lambda x: band(x)[0])
    inc["color"] = inc.apply(lambda r: "#e8c33a" if r["unsure"] else band(r["score"])[1], axis=1)
    inc = inc.sort_values("score", ascending=False).reset_index(drop=True)
    inc["incident_id"] = [f"INC-{i+1:02d}" for i in range(len(inc))]
    return inc


# ---------------------------------------------------------------- Silent Zones
def silent_zones(areas: pd.DataFrame, reports: pd.DataFrame, incidents: pd.DataFrame, now) -> pd.DataFrame:
    rows = []
    recent_from = now - timedelta(hours=RECENT_HOURS)
    for _, a in areas.iterrows():
        mine = reports[reports["area"] == a["area"]]
        recent = int((mine["time"] >= recent_from).sum())
        before = int((mine["time"] < recent_from).sum())
        # nearby damage from the Damage Ruler (other areas within NEIGHBOUR_KM)
        near = incidents[(incidents["area"] != a["area"]) & (incidents["disaster"] == a["disaster"])]
        near = near[near.apply(lambda r: km(a["lat"], a["lon"], r["lat"], r["lon"]) <= NEIGHBOUR_KM, axis=1)]
        near_level = int(near["level"].max()) if len(near) else 0
        near_n = int((near["level"] >= 4).sum()) if len(near) else 0
        people = min(1.0, 0.5 * a["population"] / 50000 + 0.5 * a["elderly_pct"] / 15)
        expected = 100 * (0.35 * a["terrain_risk"] + 0.30 * a["hazard_intensity"] + 0.25 * (near_level / 5) + 0.10 * people)
        expected = round(expected)
        went_quiet = before >= 3 and recent == 0
        silent = expected >= SILENT_THRESHOLD and (recent == 0 or went_quiet)
        reason = (f"{a['terrain_note']}; {a['hazard_note']}; "
                  f"{near_n} severe incident(s) within {NEIGHBOUR_KM} km (worst level {near_level}); "
                  f"{recent} message(s) in the last {RECENT_HOURS} h")
        if went_quiet:
            reason += f" (was sending {before} earlier, then stopped - network may be down)"
        rows.append({"area": a["area"], "region": a["region"], "disaster": a["disaster"], "lat": a["lat"], "lon": a["lon"],
                     "danger_expected": expected, "recent_msgs": recent, "earlier_msgs": before,
                     "went_quiet": went_quiet, "silent": silent, "reason": reason,
                     "population": int(a["population"])})
    return pd.DataFrame(rows).sort_values("danger_expected", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- resources
RESOURCE_FOR_NEED = {
    "boat": "Boat", "rescue team": "Rescue team", "search and rescue team": "Rescue team",
    "ambulance": "Ambulance", "jcb": "JCB", "road clearing team": "JCB",
    "shelter": "Relief team", "food": "Relief team", "water": "Relief team", "evacuation": "Rescue team",
    "power cut": "Electricity board", "engineer inspection": "Engineer", "drone": "Drone",
}


def match_resources(inc: pd.DataFrame, zones: pd.DataFrame, available: dict) -> pd.DataFrame:
    """Greedy: highest priority first gets the first free resource of the type it needs."""
    left = dict(available)
    queue = []
    for _, r in inc.iterrows():
        queue.append((r["score"], r["incident_id"], f"{r['area']} ({r['disaster']}, level {r['level']})", r["help_needed"]))
    for _, z in zones[zones["silent"]].iterrows():
        queue.append((z["danger_expected"], f"SZ-{z['area']}", f"Silent Zone: {z['area']}", ["drone"]))
    queue.sort(key=lambda x: -x[0])
    plan = []
    for score, cid, label, needs in queue:
        given, waiting = [], []
        for need in needs:
            res = RESOURCE_FOR_NEED.get(need.lower())
            if not res:
                continue
            if res == "Drone" and left.get("Drone", 0) == 0 and left.get("Rescue team", 0) > 0:
                res = "Rescue team"  # no drone left: send a patrol team to check
            if res in ("Electricity board", "Engineer"):
                given.append(res)  # notify, not a counted vehicle
                continue
            if left.get(res, 0) > 0 and res not in given:
                left[res] -= 1
                given.append(res)
            elif res not in given:
                waiting.append(res)
        plan.append({"priority": round(score), "case": label, "id": cid,
                     "assigned": ", ".join(given) or "-", "waiting_for": ", ".join(sorted(set(waiting))) or "-"})
    return pd.DataFrame(plan), left
