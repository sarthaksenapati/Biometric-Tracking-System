# dashboard.py
#
# Monitoring dashboard — no live video feed (use OpenCV window for that).
#
# USAGE:
# ──────
# Terminal 1:  python run_tracker_multi.py
# Terminal 2:  python -m streamlit run dashboard.py

import os
import time
import json
from datetime import datetime, timedelta
import streamlit as st

# Database support (optional)
USE_DATABASE = os.getenv("USE_DATABASE", "true").lower() == "true"
try:
    if USE_DATABASE:
        from db.models import Event as DBEvent, Person as DBPerson

        print("[DASHBOARD] ✅ Database module loaded")
    else:
        raise ImportError("File-based mode")
except ImportError:
    print("[DASHBOARD] ⚠️  Using file-based storage")
    USE_DATABASE = False

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

STATE_FILE = "tracker_state.json"
HISTORY_FILE = "tracker_history.json"  # persistent across sessions
REFRESH_INTERVAL_S = 2
MAX_EVENTS_SHOWN = 60
STALE_THRESHOLD_S = 5
HISTORY_RETENTION_DAYS = 7  # keep 7 days of history

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Biometric Tracker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
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
    .ev-old          { opacity: 0.6; }

    .cam-online      { color: #a6e3a1; font-weight: 700; }
    .cam-offline     { color: #f38ba8; font-weight: 700; }

    .stale-banner    { background:#2e1e1e; border-left:4px solid #f38ba8;
                       padding:10px 14px; border-radius:6px; color:#f38ba8;
                       font-size:0.9rem; margin-bottom:12px; }
    .ok-banner       { background:#1e2e1e; border-left:4px solid #a6e3a1;
                       padding:10px 14px; border-radius:6px; color:#a6e3a1;
                       font-size:0.9rem; margin-bottom:12px; }
    .search-found    { background:#1e2e1e; border-left:4px solid #89b4fa;
                       padding:10px 14px; border-radius:6px; margin-top:8px; }
    .search-past     { background:#2e2e1e; border-left:4px solid #f9e2af;
                       padding:10px 14px; border-radius:6px; margin-top:8px;
                       color:#f9e2af; font-size:0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────
# Helper: format age
# ─────────────────────────────────────────────────────────────────────


def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.0f}m"
    if seconds < 86400:
        return f"{seconds/3600:.1f}h"
    return f"{seconds/86400:.1f}d"


# ─────────────────────────────────────────────────────────────────────
# Load state and history (Database or File-based)
# ─────────────────────────────────────────────────────────────────────


def load_state() -> dict | None:
    """Load tracker state from DB or file."""
    if USE_DATABASE:
        try:
            # Try Redis cache first
            from cache import get_cache

            cache = get_cache()
            if cache.is_available:
                state = cache.get_cached_tracker_state()
                if state:
                    return state
        except Exception:
            pass

    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_history() -> dict:
    """Load persistent history from DB or file."""
    if USE_DATABASE:
        try:
            history = {"last_seen": {}, "events": []}
            # Load recent events from DB
            events = DBEvent.get_recent(limit=1000)
            history["events"] = [dict(e) for e in events]

            # Build last_seen from persons
            persons = DBPerson.list_all()
            for p in persons:
                history["last_seen"][p["name"]] = {
                    "location": "Unknown",
                    "timestamp": p["created_at"].timestamp() if p.get("created_at") else time.time(),
                    "confidence": 0.0,
                }

            return history
        except Exception as e:
            print(f"[DASHBOARD] DB load failed: {e}")

    # Fallback to file
    if not os.path.exists(HISTORY_FILE):
        return {"last_seen": {}, "events": []}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"last_seen": {}, "events": []}


def merge_and_save_history(current_state: dict):
    """Merge current tracker_state.json events into history."""
    if not current_state:
        return

    if USE_DATABASE:
        try:
            # Events are already in DB (logged by tracker)
            # Just update last_seen
            for person in current_state.get("active_people", []):
                name = person.get("name")
                if not name or name == "Unknown":
                    continue
                # Update in DB via tracker
                pass
            return
        except Exception as e:
            print(f"[DASHBOARD] DB merge failed: {e}")

    # File-based fallback
    history = load_history()
    cutoff = time.time() - HISTORY_RETENTION_DAYS * 86400

    # Build a set of existing event keys to avoid duplicates
    existing_keys = set()
    for e in history["events"]:
        key = (e.get("type"), e.get("person"), round(e.get("timestamp", 0), 1))
        existing_keys.add(key)

    # Merge new events from current state
    new_events = []
    for e in current_state.get("events", []):
        key = (e.get("type"), e.get("person"), round(e.get("timestamp", 0), 1))
        if key not in existing_keys:
            new_events.append(e)
            existing_keys.add(key)

    history["events"].extend(new_events)

    # Prune old events
    history["events"] = [e for e in history["events"] if e.get("timestamp", 0) >= cutoff]

    # Update last_seen for each active person
    for person in current_state.get("active_people", []):
        name = person.get("name")
        if not name or name == "Unknown":
            continue
        existing = history["last_seen"].get(name)
        if existing is None or person["last_seen"] > existing["timestamp"]:
            history["last_seen"][name] = {
                "location": person["location"],
                "timestamp": person["last_seen"],
                "confidence": person["confidence"],
                "cam_id": person.get("cam_id"),
            }

    # Also update last_seen from sighting events
    for e in current_state.get("events", []):
        if e.get("type") != "sighting":
            continue
        name = e.get("person")
        if not name or name == "Unknown":
            continue
        existing = history["last_seen"].get(name)
        if existing is None or e["timestamp"] > existing["timestamp"]:
            history["last_seen"][name] = {
                "location": e["location"],
                "timestamp": e["timestamp"],
                "confidence": e.get("confidence", 0),
                "cam_id": e.get("cam_id"),
            }

    # Prune last_seen entries older than retention
    history["last_seen"] = {
        name: info for name, info in history["last_seen"].items() if info.get("timestamp", 0) >= cutoff
    }

    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except OSError as exc:
        st.warning(f"Could not write history: {exc}")


# ── Load everything ────────────────────────────────────────────────────

state = load_state()
file_exists = state is not None
file_age = (time.time() - state["written_at"]) if state else None
tracker_live = file_exists and file_age is not None and file_age < STALE_THRESHOLD_S

# Merge current state into persistent history
if state:
    merge_and_save_history(state)

history = load_history()
active_people = state.get("active_people", []) if state else []
all_events = state.get("events", []) if state else []
cameras = state.get("cameras", []) if state else []
retention_min = state.get("retention_min", 5) if state else 5
online_count = sum(1 for c in cameras if c.get("online", False))
handoff_count = sum(1 for e in all_events if e.get("type") == "handoff")

# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎯 Biometric Tracker")
    st.markdown("*Dashboard — monitoring only*")
    st.markdown("---")

    # ── Tracker status ────────────────────────────────────────────────────
    st.markdown("#### 🔌 Tracker Status")
    if not file_exists:
        st.error("tracker_state.json not found.\nStart:\n`python run_tracker_multi.py`")
    elif not tracker_live:
        st.warning(f"State file is {file_age:.0f}s old — tracker may have stopped.")
    else:
        st.success(f"Tracker running  ·  updated {file_age:.1f}s ago")

    st.markdown("---")

    # ── Camera status ─────────────────────────────────────────────────────
    st.markdown("#### 📷 Cameras")
    if cameras:
        for cam in cameras:
            dot = (
                '<span class="cam-online">● ONLINE</span>'
                if cam.get("online")
                else '<span class="cam-offline">● OFFLINE</span>'
            )
            st.markdown(
                f"**{cam['location']}** &nbsp; {dot}",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No camera data yet.")

    st.markdown("---")

    # ── Person search ─────────────────────────────────────────────────────
    st.markdown("#### 🔍 Find Person")
    st.caption("Searches current session AND history across restarts")
    search_raw = st.text_input("Name", placeholder="e.g. Vineet", label_visibility="collapsed")

    if search_raw.strip():
        search_lower = search_raw.strip().lower()

        # 1. Check if currently active
        active_match = next(
            (p for p in active_people if p["name"].lower() == search_lower),
            None,
        )

        if active_match:
            age = time.time() - active_match["last_seen"]
            st.markdown(
                f'<div class="search-found">'
                f'🟢 <b>{active_match["name"]}</b> is currently at '
                f'<b>{active_match["location"]}</b><br>'
                f'Confidence: {int(active_match["confidence"] * 100)}%  ·  '
                f"Last seen {age:.0f}s ago"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            # 2. Check current session events
            session_past = [
                e
                for e in reversed(all_events)
                if e.get("type") == "sighting" and e.get("person", "").lower() == search_lower
            ]

            # 3. Check persistent history
            history_match = next(
                ((name, info) for name, info in history["last_seen"].items() if name.lower() == search_lower),
                None,
            )

            if session_past:
                e = session_past[0]
                age = time.time() - e["timestamp"]
                st.markdown(
                    f'<div class="search-past">'
                    f"🟡 <b>{search_raw}</b> was last seen at "
                    f'<b>{e["location"]}</b><br>'
                    f"{_fmt_age(age)} ago (this session, no longer in active window)"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif history_match:
                name, info = history_match
                age = time.time() - info["timestamp"]
                when = datetime.fromtimestamp(info["timestamp"]).strftime("%d %b %Y  %H:%M:%S")
                st.markdown(
                    f'<div class="search-past">'
                    f"🕐 <b>{name}</b> was last seen at "
                    f'<b>{info["location"]}</b><br>'
                    f"{when}  ({_fmt_age(age)} ago) — from a previous session"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning(f"'{search_raw}' not found in current session or history.")

    st.markdown("---")

    # ── History scope selector ────────────────────────────────────────────
    st.markdown("#### 📅 Event History Scope")
    history_scope = st.radio(
        "Show events from",
        ["Current session", "Last 1 hour", "Last 24 hours", "Last 7 days"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── Event filter ──────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Event Filter")
    event_filter = st.radio(
        "Show",
        ["All", "Handoffs only", "Sightings only"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption(
        f"Session retention: {retention_min} min  |  "
        f"History: {HISTORY_RETENTION_DAYS} days  |  "
        f"Refresh: {REFRESH_INTERVAL_S}s"
    )

# ─────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────

st.markdown("# Multi-Camera Biometric Tracker")
st.caption("Live video is shown in the OpenCV window. This dashboard shows identity & event data.")

# ── Tracker health banner ──────────────────────────────────────────────────────
if not file_exists:
    st.markdown(
        '<div class="stale-banner">⚠️  Tracker not running — '
        "start with <code>python run_tracker_multi.py</code></div>",
        unsafe_allow_html=True,
    )
elif not tracker_live:
    st.markdown(
        f'<div class="stale-banner">⚠️  Tracker state is {file_age:.0f}s old — '
        f"it may have stopped. Showing last known data.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="ok-banner">✅  Tracker live — ' f"last update {file_age:.1f}s ago</div>",
        unsafe_allow_html=True,
    )

# ── Top metrics ────────────────────────────────────────────────────────────────
total_known = len(history["last_seen"])
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Active Now", len(active_people))
m2.metric("Cameras Online", f"{online_count} / {len(cameras)}")
m3.metric("Handoffs (session)", handoff_count)
m4.metric("Events (session)", len(all_events))
m5.metric("Known Persons (history)", total_known)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────
# Active people
# ─────────────────────────────────────────────────────────────────────

st.markdown("### 👤 Active People")

if not active_people:
    st.info(
        "No people in the current retention window." if tracker_live else "Start the tracker to see detections here."
    )
else:
    cols = st.columns(min(len(active_people), 3))
    for col, person in zip(
        (cols * ((len(active_people) // len(cols)) + 1)),
        sorted(active_people, key=lambda x: x["last_seen"], reverse=True),
    ):
        age = time.time() - person["last_seen"]
        time_str = datetime.fromtimestamp(person["last_seen"]).strftime("%H:%M:%S")
        conf_pct = int(person["confidence"] * 100)
        col.markdown(
            f"""
<div class="person-card">
  <div class="p-name">👤 {person['name']}</div>
  <div class="p-loc">📍 {person['location']}</div>
  <div class="p-conf">Confidence: {conf_pct}%</div>
  <div class="p-time">Last seen {time_str} ({age:.0f}s ago)</div>
</div>""",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────
# Known persons history panel
# ─────────────────────────────────────────────────────────────────────

with st.expander(f"📚 All Known Persons — History ({total_known} people)", expanded=False):
    if not history["last_seen"]:
        st.info("No history yet — run the tracker to build history.")
    else:
        sorted_persons = sorted(
            history["last_seen"].items(),
            key=lambda x: x[1]["timestamp"],
            reverse=True,
        )
        h_cols = st.columns(3)
        for i, (name, info) in enumerate(sorted_persons):
            col = h_cols[i % 3]
            age = time.time() - info["timestamp"]
            when = datetime.fromtimestamp(info["timestamp"]).strftime("%d %b %H:%M")
            col.markdown(
                f"""
<div class="person-card">
  <div class="p-name">👤 {name}</div>
  <div class="p-loc">📍 {info['location']}</div>
  <div class="p-conf">Confidence: {int(info['confidence'] * 100)}%</div>
  <div class="p-time">{when}  ({_fmt_age(age)} ago)</div>
</div>""",
                unsafe_allow_html=True,
            )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────
# Event log — scope aware
# ─────────────────────────────────────────────────────────────────────

st.markdown("### 📋 Event Log")

# Select events based on scope
now = time.time()
if history_scope == "Current session":
    scope_events = all_events
elif history_scope == "Last 1 hour":
    cutoff = now - 3600
    scope_events = [e for e in history["events"] if e["timestamp"] >= cutoff]
elif history_scope == "Last 24 hours":
    cutoff = now - 86400
    scope_events = [e for e in history["events"] if e["timestamp"] >= cutoff]
else:  # Last 7 days
    scope_events = history["events"]

# Apply type filter
if event_filter == "Handoffs only":
    filtered = [e for e in scope_events if e.get("type") == "handoff"]
elif event_filter == "Sightings only":
    filtered = [e for e in scope_events if e.get("type") == "sighting"]
else:
    filtered = scope_events

events_to_show = list(reversed(filtered))[:MAX_EVENTS_SHOWN]
st.caption(f"Showing {len(events_to_show)} of {len(filtered)} events " f"({history_scope.lower()})")

if not events_to_show:
    st.info("No events in the selected scope.")
else:
    session_cutoff = now - (retention_min * 60)
    for event in events_to_show:
        ts = datetime.fromtimestamp(event["timestamp"]).strftime("%d %b  %H:%M:%S")
        etype = event.get("type", "")
        # Dim events outside the active retention window
        is_old = event.get("timestamp", now) < session_cutoff
        old_class = " ev-old" if is_old else ""

        if etype == "handoff":
            st.markdown(
                f'<div class="ev-row ev-handoff{old_class}">'
                f'🔄 [{ts}] &nbsp;<b>HANDOFF</b>&nbsp; {event["person"]} &nbsp;'
                f'{event["from_loc"]} → {event["to_loc"]} &nbsp;'
                f'<span style="color:#6c7086">({event["elapsed_s"]}s gap)</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
        elif etype == "sighting":
            conf_str = f'{int(event.get("confidence", 0) * 100)}%'
            st.markdown(
                f'<div class="ev-row ev-sighting{old_class}">'
                f'👁 [{ts}] &nbsp;<b>SIGHTED</b>&nbsp; {event["person"]} &nbsp;'
                f'@ {event["location"]} &nbsp;'
                f'<span style="color:#6c7086">conf {conf_str}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────
# Auto-refresh
# ─────────────────────────────────────────────────────────────────────

placeholder = st.empty()
for remaining in range(REFRESH_INTERVAL_S, 0, -1):
    placeholder.caption(f"⟳ Refreshing in {remaining}s…")
    time.sleep(1)
placeholder.empty()
st.rerun()
