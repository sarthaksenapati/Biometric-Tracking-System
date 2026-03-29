# utils/admin_controls.py
# Runtime admin controls for MultiCameraTracker
# Call these from a separate terminal while the tracker is running
# Usage: python -m utils.admin_controls <command> [args]

import json
import os
import sys
import time

CONTROL_FILE = "tracker_admin.json"


def send_command(command: dict):
    """Write a command to the control file for the tracker to pick up."""
    command["timestamp"] = time.time()
    with open(CONTROL_FILE, "w") as f:
        json.dump(command, f, indent=2)
    print(f"[ADMIN] Command sent: {command}")


def clear_unknown_persons():
    """Delete unknown_persons.json so AutoEnroller starts fresh."""
    if os.path.exists("unknown_persons.json"):
        os.remove("unknown_persons.json")
        print("[ADMIN] Cleared unknown_persons.json")
    send_command({"action": "reset_enroller"})


def rename_person(label: str, new_name: str):
    """Rename a Person_N to a real name."""
    send_command({"action": "rename", "label": label, "new_name": new_name})


def force_register(label: str):
    """Force promote a Person_N to embeddings_db immediately."""
    send_command({"action": "force_promote", "label": label})


def list_persons():
    """List all currently tracked unknown persons."""
    send_command({"action": "list"})


def clear_cache():
    """Clear the shared identity cache."""
    send_command({"action": "clear_cache"})


def reset_enroller():
    """Reset the AutoEnroller completely — clears all Person_N candidates."""
    send_command({"action": "reset_enroller"})


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m utils.admin_controls reset_enroller")
        print("  python -m utils.admin_controls rename Person_2 Ranjan")
        print("  python -m utils.admin_controls force_promote Person_2")
        print("  python -m utils.admin_controls list")
        print("  python -m utils.admin_controls clear_cache")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "reset_enroller":
        reset_enroller()
    elif cmd == "rename" and len(sys.argv) == 4:
        rename_person(sys.argv[2], sys.argv[3])
    elif cmd == "force_promote" and len(sys.argv) == 3:
        force_register(sys.argv[2])
    elif cmd == "list":
        list_persons()
    elif cmd == "clear_cache":
        clear_cache()
    else:
        print(f"Unknown command: {cmd}")