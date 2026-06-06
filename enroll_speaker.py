#!/usr/bin/env python3
"""enroll_speaker.py — CLI for managing the voice-inject speaker database."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd


def _load_dotenv():
    """Load .env from the project directory into os.environ (no extra deps needed)."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

_load_dotenv()

SAMPLE_RATE = 16000


def _record(seconds: int, label: str = "Speak now!") -> np.ndarray:
    """Countdown, record, return float32 mono array."""
    for i in range(3, 0, -1):
        print(f"Recording in {i}...", end=" ", flush=True)
        time.sleep(1)
    print(label)
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    print("Done.")
    # int16 -> float32 normalised to [-1, 1]
    return audio[:, 0].astype(np.float32) / 32768.0


def _make_identifier():
    """Build a SpeakerIdentifier, exiting cleanly on missing HF token."""
    from speaker_id import SpeakerIdentifier

    try:
        si = SpeakerIdentifier()
        si._ensure_model()  # surface token errors early
        return si
    except ValueError as exc:
        # token missing — give a clear actionable message
        if "HuggingFace token" in str(exc):
            print(
                "Error: HuggingFace token not found.\n"
                "Fix: export HUGGINGFACE_TOKEN=hf_...\n"
                "  or add  hf_token: hf_...  to ~/.voice-inject/config.yaml",
                file=sys.stderr,
            )
            sys.exit(1)
        raise


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_enroll(args):
    from speaker_db import SpeakerDB

    si = _make_identifier()
    db = SpeakerDB()

    print(f"Enrolling '{args.name}' — record {args.seconds}s of clear speech.")
    audio = _record(args.seconds)

    embedding = si.get_embedding(audio, SAMPLE_RATE)
    if np.all(embedding == 0):
        print("Error: embedding extraction failed (audio too short or silent).", file=sys.stderr)
        sys.exit(1)

    db.add_speaker(args.name, embedding)
    count = db.get_embedding_count(args.name)
    print(f"Enrolled {args.name} ({count} embedding{'s' if count != 1 else ''} total).")


def cmd_list(_args):
    from speaker_db import SpeakerDB

    db = SpeakerDB()
    speakers = db.list_speakers()
    if not speakers:
        print("No speakers enrolled yet.")
        return
    print(f"{'Speaker':<30} {'Embeddings':>10}")
    print("-" * 42)
    for name in speakers:
        count = db.get_embedding_count(name)
        print(f"{name:<30} {count:>10}")


def cmd_remove(args):
    from speaker_db import SpeakerDB

    db = SpeakerDB()
    try:
        db.remove_speaker(args.name)
        print(f"Removed '{args.name}' from the database.")
    except KeyError:
        print(f"Error: '{args.name}' is not in the database.", file=sys.stderr)
        sys.exit(1)


def cmd_test(_args):
    from speaker_db import SpeakerDB

    si = _make_identifier()
    db = SpeakerDB()

    if not db.list_speakers():
        print("No speakers enrolled yet. Run:  enroll_speaker.py enroll <name>", file=sys.stderr)
        sys.exit(1)

    print("Recording 3s — speak now...")
    audio = _record(3, label="Speak!")
    embedding = si.get_embedding(audio, SAMPLE_RATE)

    name, similarity = db.find_closest(embedding)
    if name is None:
        print("Result: Unknown (empty database).")
    else:
        threshold = si.similarity_threshold
        match = similarity >= threshold
        verdict = name if match else "Unknown"
        print(f"Result: {verdict}  (closest={name}, similarity={similarity:.3f}, threshold={threshold:.2f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Manage the voice-inject speaker database.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_enroll = sub.add_parser("enroll", help="Record and enroll a speaker.")
    p_enroll.add_argument("name", help="Speaker name (e.g. \"John\")")
    p_enroll.add_argument("--seconds", type=int, default=5, help="Recording length (default 5s)")
    p_enroll.set_defaults(func=cmd_enroll)

    p_list = sub.add_parser("list", help="List all enrolled speakers.")
    p_list.set_defaults(func=cmd_list)

    p_remove = sub.add_parser("remove", help="Remove a speaker from the database.")
    p_remove.add_argument("name", help="Speaker name to remove.")
    p_remove.set_defaults(func=cmd_remove)

    p_test = sub.add_parser("test", help="Record 3s and identify the speaker.")
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
