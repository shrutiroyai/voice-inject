"""Vocabulary management — load custom terms and inject into LLM prompt."""

import sys
from pathlib import Path
import yaml

VOCAB_PATH = Path.home() / ".voice-inject" / "vocab.yaml"


def _ensure_vocab():
    if not VOCAB_PATH.exists():
        VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        VOCAB_PATH.write_text(yaml.dump({"corrections": []}, default_flow_style=False))


def load_vocab() -> list[dict]:
    _ensure_vocab()
    data = yaml.safe_load(VOCAB_PATH.read_text()) or {}
    return data.get("corrections", [])


def load_vocab_prompt() -> str:
    corrections = load_vocab()
    if not corrections:
        return ""
    lines = ["VOCABULARY RULES (always use these exact spellings):"]
    for entry in corrections:
        variants = " / ".join(f'"{h}"' for h in entry["hear"])
        lines.append(f'- {variants} → {entry["use"]}')
    return "\n".join(lines)


def add_term(hear_variants: list[str], correct: str):
    _ensure_vocab()
    data = yaml.safe_load(VOCAB_PATH.read_text()) or {"corrections": []}
    data.setdefault("corrections", []).append({"hear": hear_variants, "use": correct})
    VOCAB_PATH.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    print(f"✅ Added: {hear_variants} → {correct}")


def list_terms():
    for entry in load_vocab():
        variants = ", ".join(entry["hear"])
        print(f"  {variants} → {entry['use']}")


def remove_term(correct: str):
    _ensure_vocab()
    data = yaml.safe_load(VOCAB_PATH.read_text()) or {"corrections": []}
    before = len(data.get("corrections", []))
    data["corrections"] = [c for c in data.get("corrections", []) if c["use"] != correct]
    VOCAB_PATH.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    removed = before - len(data["corrections"])
    print(f"🗑️  Removed {removed} entry(ies) for '{correct}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vocab.py [list|add|remove]")
        print("  add <correct> <variant1> <variant2> ...")
        print("  remove <correct>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "list":
        list_terms()
    elif cmd == "add" and len(sys.argv) >= 4:
        add_term(hear_variants=sys.argv[3:], correct=sys.argv[2])
    elif cmd == "remove" and len(sys.argv) >= 3:
        remove_term(sys.argv[2])
    else:
        print("Invalid usage. Run without args for help.")
