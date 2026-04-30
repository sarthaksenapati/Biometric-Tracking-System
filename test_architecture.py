#!/usr/bin/env python3
"""
Test script to verify the new architecture components:
1. PostgreSQL Database
2. Redis Cache
3. Message Queue (optional)
"""

import os
import sys

print("=" * 60)
print("Architecture Test Suite - Biometric Tracking System")
print("=" * 60)

# ── Test 1: Database Connection ─────────────────────────────
print("\n[1/4] Testing PostgreSQL Database...")
try:
    from db.connection import get_db, init_db, close_db

    init_db()
    print("✅ Database schema initialized")

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM persons")
            count = cur.fetchone()["count"]
            print(f"✅ Database connection OK (persons: {count})")

    print("✅ Database test PASSED")
except Exception as e:
    print(f"❌ Database test FAILED: {e}")
    print("   (Set USE_DATABASE=false to use file-based storage)")

# ── Test 2: Redis Cache ─────────────────────────────────
print("\n[2/4] Testing Redis Cache...")
try:
    from cache import get_cache

    cache = get_cache()
    if cache.is_available:
        cache.cache_tracker_state({"test": True})
        state = cache.get_cached_tracker_state()
        if state and state.get("test"):
            print("✅ Redis cache test PASSED")
        else:
            print("❌ Redis cache test FAILED: could not read back data")
    else:
        print("⚠️  Redis not available (optional)")
except Exception as e:
    print(f"❌ Redis test FAILED: {e}")

# ── Test 3: Message Queue ──────────────────────────────
print("\n[3/4] Testing Message Queue...")
try:
    from task_queue import get_queue

    q = get_queue()
    print(f"✅ Message queue initialized (type: {q.queue_type})")
    print("✅ Message queue test PASSED")
except Exception as e:
    print(f"❌ Message queue test FAILED: {e}")

# ── Test 4: Matcher with DB ────────────────────────────
print("\n[4/4] Testing Matcher with Database...")
try:
    from core.matcher import Matcher

    m = Matcher()
    count = len(m.database)
    print(f"✅ Matcher loaded {count} persons from database")
    print("✅ Matcher test PASSED")
except Exception as e:
    print(f"❌ Matcher test FAILED: {e}")

# ── Summary ───────────────────────────────────────────
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)

print("\nEnvironment:")
print(f"  USE_DATABASE = {os.getenv('USE_DATABASE', 'true')}")
print(f"  USE_QUEUE    = {os.getenv('USE_QUEUE', 'false')}")
print(f"  DATABASE_URL  = {os.getenv('DATABASE_URL', 'not set')}")
print(f"  REDIS_URL     = {os.getenv('REDIS_URL', 'not set')}")

print("\nTo run with all features:")
print("  export USE_DATABASE=true")
print("  export USE_QUEUE=true")
print("  python run_tracker_multi.py")
print()
