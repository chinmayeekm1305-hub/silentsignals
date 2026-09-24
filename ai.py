"""Damage Ruler - live AI call (Google Gemini).

Sends one photo + message to Gemini and gets back structured JSON.
Needs GEMINI_API_KEY in .streamlit/secrets.toml (or as an environment variable).
"""
import json
import os

from engine import DANGER_SCALE

PROMPT = """You are a disaster damage assessor helping emergency officers in India.
Look at the photo and read the message together. Return ONLY valid JSON, no extra text, with these keys:
disaster_type: one of flood | cyclone | earthquake | landslide | unclear
reference_objects: list of objects you used to measure (car, person, door, scooter, building floor...)
measurement: one line of what you measured, e.g. "water reaches the car window, about 1.1 m"
damage_level: integer 1-5 using the scale below
people_at_risk: one of none | possible | confirmed
people_count: number or null
vulnerable: list from [children, elderly, pregnant, injured, disabled]
road_blocked: one of true | false | unknown
hazards: list, e.g. ["electric pole in water"]
help_needed: list from [boat, rescue team, search and rescue team, ambulance, JCB, road clearing team, shelter, food, water, evacuation, power cut, engineer inspection, monitor]
hours_to_critical: number (how soon this becomes life-threatening)
photo_text_match: true or false
confidence: number 0-1
reason_if_unsure: one line, empty if confident
english_text: the message translated to English

Danger scale (level 1 to 5):
""" + "\n".join(f"{k}: " + " | ".join(f"{i+1}={v}" for i, v in enumerate(v_list)) for k, v_list in DANGER_SCALE.items()) + """

Rules: never invent details you cannot see or read. If there is no object to measure against, say so and lower confidence.
If the photo and message disagree, set photo_text_match to false and explain in reason_if_unsure.

Message: """


def get_key(secrets=None):
    try:
        if secrets is not None and "GEMINI_API_KEY" in secrets:
            return secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def analyse(photo_bytes, mime_type, message, api_key, model="gemini-2.5-flash"):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    parts = [PROMPT + (message or "(no message)")]
    if photo_bytes:
        parts.insert(0, types.Part.from_bytes(data=photo_bytes, mime_type=mime_type or "image/jpeg"))
    resp = client.models.generate_content(
        model=model, contents=parts,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
    )
    data = json.loads(resp.text)
    # clean-up so the rest of the app can trust the fields
    data["damage_level"] = max(1, min(5, int(data.get("damage_level") or 3)))
    data["confidence"] = float(data.get("confidence") or 0.5)
    for k in ["vulnerable", "help_needed", "hazards", "reference_objects"]:
        v = data.get(k) or []
        data[k] = v if isinstance(v, list) else [str(v)]
    return data
