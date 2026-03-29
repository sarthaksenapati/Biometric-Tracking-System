# rename_person.py
import json
import sys
import os

FILE = "unknown_persons.json"

def load():
    if not os.path.exists(FILE):
        print(f"No {FILE} found — run the tracker first to register unknown persons.")
        return {}
    with open(FILE) as f:
        return json.load(f)

def save(data):
    with open(FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {FILE}")

def list_persons(data):
    if not data:
        print("No unknown persons registered yet.")
        return
    print(f"\n{'Label':<12}  {'Display Name'}")
    print("-" * 30)
    for label, name in sorted(data.items()):
        marker = "  (unnamed)" if name == label else ""
        print(f"{label:<12}  {name}{marker}")
    print()

if __name__ == "__main__":
    data = load()

    # List mode
    if len(sys.argv) == 1:
        list_persons(data)
        sys.exit(0)

    # Rename mode
    if len(sys.argv) == 3:
        label    = sys.argv[1]   # e.g. Person_1
        new_name = sys.argv[2]   # e.g. Raj
        if label not in data:
            print(f"'{label}' not found. Available:")
            list_persons(data)
            sys.exit(1)
        old_name = data[label]
        data[label] = new_name
        save(data)
        print(f"✅  '{label}' renamed: '{old_name}' → '{new_name}'")
        sys.exit(0)

    print("Usage:")
    print("  python rename_person.py              ← list all")
    print("  python rename_person.py Person_1 Raj ← rename")