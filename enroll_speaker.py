#!/usr/bin/env python3
"""enroll_speaker.py — CLI for managing the voice-inject speaker database."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from speaker_db import SpeakerDB
from speaker_id import SpeakerIdentifier

SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5  # seconds for enrollment

def cmd_add(args):
    db = SpeakerDB()
    sid = SpeakerIdentifier(db, use_auth_token=os.environ.get("HUGGINGFACE_TOKEN", True))
    
    print(f"🎤 Recording {DURATION}s for '{args.name}'...")
    print("   Please speak naturally...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
    sd.wait()
    
    emb = sid.get_embedding(audio.flatten())
    db.add_speaker(args.name, emb)
    print(f"✅ Enrolled '{args.name}' (dim={len(emb)})")

def cmd_list(args):
    db = SpeakerDB()
    speakers = db.list_speakers()
    if not speakers:
        print("Empty database.")
    else:
        print(f"Speakers ({len(speakers)}):")
        for s in speakers:
            count = db.get_embedding_count(s)
            print(f"  - {s} ({count} embeddings)")

def cmd_remove(args):
    db = SpeakerDB()
    try:
        db.remove_speaker(args.name)
        print(f"✅ Removed '{args.name}'.")
    except KeyError:
        print(f"❌ Speaker '{args.name}' not found.")

def cmd_test(args):
    db = SpeakerDB()
    sid = SpeakerIdentifier(db, use_auth_token=os.environ.get("HUGGINGFACE_TOKEN", True))
    
    print("🎤 Recording 3s to identify...")
    audio = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
    sd.wait()
    
    name = sid.identify(audio.flatten())
    print(f"👤 Result: {name}")

def main():
    parser = argparse.ArgumentParser(description="Voice Inject Speaker Enrollment CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Enroll a new speaker.")
    p_add.add_argument("name", help="Name of the person.")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="List all enrolled speakers.")
    p_list.set_defaults(func=cmd_list)

    p_remove = sub.add_parser("remove", help="Remove a speaker.")
    p_remove.add_argument("name", help="Name to remove.")
    p_remove.set_defaults(func=cmd_remove)

    p_test = sub.add_parser("test", help="Identify the speaker from a 3s clip.")
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
