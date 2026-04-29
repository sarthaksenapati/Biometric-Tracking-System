#!/usr/bin/env python3
"""
Patch Streamlit's polling_path_watcher.py to fix queue.SimpleQueue error.
This is a Python 3.12+ compatibility issue.

Run this script before starting Streamlit:
    python patch_streamlit.py
    streamlit run dashboard.py
"""

import sys
import os


def patch_streamlit():
    """Patch the polling_path_watcher.py file in Streamlit."""
    try:
        import streamlit
        streamlit_path = os.path.dirname(streamlit.__file__)
        watcher_path = os.path.join(
            streamlit_path, 'watcher', 'polling_path_watcher.py'
        )

        if not os.path.exists(watcher_path):
            print(f"❌ File not found: {watcher_path}")
            return False

        with open(watcher_path, 'r') as f:
            content = f.read()

        # Check if already patched
        if 'SimpleQueue' not in content:
            print("✅ Streamlit already patched or not affected")
            return True

        # Replace queue.SimpleQueue with queue.Queue
        # SimpleQueue was removed in Python 3.12+
        patched_content = content.replace(
            'queue.SimpleQueue()',
            'queue.Queue()'
        )

        if patched_content == content:
            print("⚠️  No SimpleQueue found to patch")
            return False

        # Also patch concurrent.futures.thread if needed
        try:
            import concurrent.futures.thread as thread_module
            thread_file = thread_module.__file__
            if thread_file:
                with open(thread_file, 'r') as f:
                    thread_content = f.read()
                if 'queue.SimpleQueue' in thread_content:
                    patched_thread = thread_content.replace(
                        'queue.SimpleQueue()',
                        'queue.Queue()'
                    )
                    with open(thread_file, 'w') as f:
                        f.write(patched_thread)
                    print(f"✅ Patched {thread_file}")
        except Exception as e:
            print(f"⚠️  Could not patch concurrent.futures: {e}")

        with open(watcher_path, 'w') as f:
            f.write(patched_content)

        print(f"✅ Patched Streamlit: {watcher_path}")
        return True

    except Exception as e:
        print(f"❌ Failed to patch Streamlit: {e}")
        return False


if __name__ == '__main__':
    success = patch_streamlit()
    sys.exit(0 if success else 1)
