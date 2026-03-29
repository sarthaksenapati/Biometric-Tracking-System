# dashboard.py
#
# Monitoring dashboard — no live video feed (use OpenCV window for that).
#
# USAGE:
# ──────
# Run BOTH of these in separate terminals:
#
#   Terminal 1 (live video + tracking):
#       python run_tracker_multi.py
#
#   Terminal 2 (dashboard):
#       python -m streamlit run dashboard.py
#
# The tracker writes state to tracker_state.json every second.
# The dashboard reads from that file — no shared memory, no model loading.

import time
import json
import os
from datetime import datetime
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

STATE_FILE         = "tracker_state.json"   # written by run_tracker_multi.py
REFRESH_INTERVAL_S = 2
MAX_EVENTS_SHOWN   = 40
STALE_THRESHOLD_S  = 5    # if state file not updated for this long → show warning

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title            = "Biometric Tracker",
    page_icon             = "🎯",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .person-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .person-card .p-name { font-size: 1.05rem; font-weight: 600; color: #cdd6f4; }
    .person-card .p-loc  { font-size: 0.85rem; color: #89b4fa; margin-top: 3px; }
    .person-card .p-conf { font-size: 0.80rem; color: #a6e3a1; margin-top: 2px; }
    .person-card .p-time { font-size: 0.75rem; color: #6c7086; margin-top: 4px; }

    .ev-row          { padding: 5px 0; border-bottom: 1px solid #313244; font-size: 0.88rem; }
    .ev-handoff      { color: #fab387; font-weight: 600; }
    .ev-sighting     { color: #89dceb; }

    .cam-online      { color: #a6e3a1; font-weight: 700; }
    .cam-offline     { color: #f38ba8; font-weight: 700; }

    .stale-banner    { background:#2e1e1e; border-left:4px solid #f38ba8;
                       padding:10px 14px; border-radius:6px; color:#f38ba8;
                       font-size:0.9rem; margin-bottom:12px; }
    .ok-banner       { background:#1e2e1e; border-left:4px solid #a6e3a1;
                       padding:10px 14px; border-radius:6px; color:#a6e3a1;
                       font-size:0.9rem; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Read state from file
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict | None:
    """
    Read the latest tracker state from tracker_state.json.
    Returns None if file doesn't exist or is unreadable.
    """
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # File may be mid-write — just skip this refresh cycle
        return None


state       = load_state()
file_exists = state is not None
file_age    = (time.time() - state["written_at"]) if state else None
tracker_live = file_exists and file_age is not None and file_age < STALE_THRESHOLD_S

# Unpack state safely
active_people = state.get("active_people", []) if state else []
all_events    = state.get("events",         []) if state else []
cameras       = state.get("cameras",        []) if state else []
retention_min = state.get("retention_min",  5)  if state else 5
online_count  = sum(1 for c in cameras if c.get("online", False))
handoff_count = sum(1 for e in all_events if e.get("type") == "handoff")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎯 Biometric Tracker")
    st.markdown("*Dashboard — monitoring only*")
    st.markdown("---")

    # ── Tracker connection status ──────────────────────────────────────────
    st.markdown("#### 🔌 Tracker Status")
    if not file_exists:
        st.error("tracker_state.json not found.\nStart the tracker first:\n`python run_tracker_multi.py`")
    elif not tracker_live:
        st.warning(f"State file is {file_age:.0f}s old — tracker may have stopped.")
    else:
        st.success(f"Tracker running  ·  updated {file_age:.1f}s ago")

    st.markdown("---")

    # ── Camera status ──────────────────────────────────────────────────────
    st.markdown("#### 📷 Cameras")
    if cameras:
        for cam in cameras:
            dot = '<span class="cam-online">● ONLINE</span>' \
                  if cam.get("online") \
                  else '<span class="cam-offline">● OFFLINE</span>'
            st.markdown(
                f"**{cam['location']}** &nbsp; {dot}",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No camera data yet.")

    st.markdown("---")

    # ── Person search ──────────────────────────────────────────────────────
    st.markdown("#### 🔍 Find Person")
    search_raw = st.text_input(
        "Name", placeholder="e.g. Prityanshu",
        label_visibility="collapsed"
    )

    if search_raw.strip():
        search_lower = search_raw.strip().lower()
        match = next(
            (p for p in active_people if p["name"].lower() == search_lower),
            None,
        )
        if match:
            age = time.time() - match["last_seen"]
            st.success(
                f"**{match['name']}** is at **{match['location']}**  \n"
                f"Confidence: {int(match['confidence'] * 100)}%  |  "
                f"Last seen {age:.0f}s ago"
            )
        else:
            # Check event log for most recent sighting
            past = [e for e in reversed(all_events)
                    if e.get("type") == "sighting"
                    and e.get("person", "").lower() == search_lower]
            if past:
                e   = past[0]
                age = time.time() - e["timestamp"]
                st.info(
                    f"**{search_raw}** was last seen at **{e['location']}**  \n"
                    f"{age:.0f}s ago (no longer in active window)"
                )
            else:
                st.warning(f"'{search_raw}' not found.")

    st.markdown("---")

    # ── Filters ────────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Event Filter")
    event_filter = st.radio(
        "Show",
        ["All", "Handoffs only", "Sightings only"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption(
        f"Retention: {retention_min} min  |  "
        f"Refresh: {REFRESH_INTERVAL_S}s  |  "
        f"Live video: OpenCV window"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# Multi-Camera Biometric Tracker")
st.caption("Live video is shown in the OpenCV window. This dashboard shows identity & event data.")

# ── Tracker health banner ─────────────────────────────────────────────────────
if not file_exists:
    st.markdown(
        '<div class="stale-banner">⚠️  Tracker not running — '
        'start it with <code>python run_tracker_multi.py</code></div>',
        unsafe_allow_html=True,
    )
elif not tracker_live:
    st.markdown(
        f'<div class="stale-banner">⚠️  Tracker state is {file_age:.0f}s old — '
        f'it may have stopped.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="ok-banner">✅  Tracker live — '
        f'last update {file_age:.1f}s ago</div>',
        unsafe_allow_html=True,
    )

# ── Top metrics ───────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Active People",     len(active_people))
m2.metric("Cameras Online",    f"{online_count} / {len(cameras)}")
m3.metric("Handoffs (window)", handoff_count)
m4.metric("Events (window)",   len(all_events))

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Active people
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 👤 Active People")

if not active_people:
    st.info("No people in the current retention window." if tracker_live
            else "Start the tracker to see detections here.")
else:
    cols = st.columns(min(len(active_people), 3))   # up to 3 per row
    for col, person in zip(
        (cols * ((len(active_people) // len(cols)) + 1)),   # cycle cols
        sorted(active_people, key=lambda x: x["last_seen"], reverse=True),
    ):
        age      = time.time() - person["last_seen"]
        time_str = datetime.fromtimestamp(person["last_seen"]).strftime("%H:%M:%S")
        conf_pct = int(person["confidence"] * 100)
        col.markdown(f"""
<div class="person-card">
  <div class="p-name">👤 {person['name']}</div>
  <div class="p-loc">📍 {person['location']}</div>
  <div class="p-conf">Confidence: {conf_pct}%</div>
  <div class="p-time">Last seen {time_str} ({age:.0f}s ago)</div>
</div>""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Event log
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 📋 Event Log")

if event_filter == "Handoffs only":
    filtered = [e for e in all_events if e.get("type") == "handoff"]
elif event_filter == "Sightings only":
    filtered = [e for e in all_events if e.get("type") == "sighting"]
else:
    filtered = all_events

events_to_show = list(reversed(filtered))[:MAX_EVENTS_SHOWN]

if not events_to_show:
    st.info("No events in the current retention window.")
else:
    for event in events_to_show:
        ts    = datetime.fromtimestamp(event["timestamp"]).strftime("%H:%M:%S")
        etype = event.get("type", "")

        if etype == "handoff":
            st.markdown(
                f'<div class="ev-row ev-handoff">'
                f'🔄 [{ts}] &nbsp;<b>HANDOFF</b>&nbsp; {event["person"]} &nbsp;'
                f'{event["from_loc"]} → {event["to_loc"]} &nbsp;'
                f'<span style="color:#6c7086">({event["elapsed_s"]}s gap)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif etype == "sighting":
            conf_str = f'{int(event.get("confidence", 0) * 100)}%'
            st.markdown(
                f'<div class="ev-row ev-sighting">'
                f'👁 [{ts}] &nbsp;<b>SIGHTED</b>&nbsp; {event["person"]} &nbsp;'
                f'@ {event["location"]} &nbsp;'
                f'<span style="color:#6c7086">conf {conf_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh countdown
# ─────────────────────────────────────────────────────────────────────────────

placeholder = st.empty()
for remaining in range(REFRESH_INTERVAL_S, 0, -1):
    placeholder.caption(f"⟳ Refreshing in {remaining}s…")
    time.sleep(1)
placeholder.empty()
st.rerun()