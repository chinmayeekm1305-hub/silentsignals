# SilentSignal - hackathon prototype (Track 01 AI, PS-01)

AI that measures damage from any disaster (Damage Ruler) and finds areas that can't call for help (Silent Zones).

## Run on your laptop
1. Install Python 3.10 or newer from python.org (tick "Add Python to PATH").
2. Open this folder in a terminal / Command Prompt.
3. pip install -r requirements.txt
4. streamlit run app.py   -> the app opens in your browser.

## Live AI (optional)
Rename .streamlit/secrets.toml.example to secrets.toml and paste your free Gemini key from https://aistudio.google.com/apikey

## Files
- app.py - screens, map, priority list, officer decisions
- engine.py - Damage Ruler levels, merging, priority score, Silent Zones, resource matching
- ai.py - Gemini photo + text analysis (live mode)
- data/reports.csv - 31 sample reports (flood, cyclone, earthquake, landslide)
- data/areas.csv - area risk data used by Silent Zones

Decision support only. Final decisions are taken by the authorised officer.
