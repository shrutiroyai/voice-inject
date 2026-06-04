#!/usr/bin/env python3
"""Voice Inject — hold Control to dictate, release to paste AI-cleaned text."""

import asyncio
import subprocess
import threading
import queue
import sounddevice as sd
from pynput import keyboard
from aws_backend import AWSBackend
from vocab import load_vocab_prompt

try:
    from config import SAMPLE_RATE, CHANNELS, COMMAND_WAKE_WORD
except ImportError:
    from config_example import SAMPLE_RATE, CHANNELS, COMMAND_WAKE_WORD

CHUNK_SAMPLES = int(SAMPLE_RATE * 0.1)  # 100ms chunks

audio_queue: queue.Queue = queue.Queue()
is_recording = False
transcript_parts: list[str] = []
transcribe_thread: threading.Thread | None = None

# Initialize AWS backend
aws_backend = AWSBackend()


def audio_callback(indata, frames, time_info, status):
    """Callback for audio recording."""
    if is_recording:
        audio_queue.put(indata.copy().tobytes())


async def transcribe_stream():
    """Transcribe audio stream using AWS backend."""
    global transcript_parts
    transcript_parts = await aws_backend.transcribe_stream(
        audio_queue,
        lambda: is_recording
    )


def build_system_prompt(raw_text: str) -> tuple[str, str]:
    """
    Build system prompt based on mode (normal vs command).
    
    Returns:
        Tuple of (system_prompt, processed_text)
    """
    # Check if this is command mode (starts with wake word)
    is_command_mode = raw_text.lower().strip().startswith(COMMAND_WAKE_WORD.lower())
    
    vocab_section = load_vocab_prompt()
    
    if is_command_mode:
        # Command mode: full LLM assistance
        system_prompt = (
            "You are Molly, a helpful dictation assistant. The user is giving you a command "
            "to execute on some text. Parse the command and execute it.\n\n"
            "Examples:\n"
            "- 'Molly, make this more formal: the numbers look good' → 'The metrics are performing well.'\n"
            "- 'Molly, summarize this: we had a meeting...' → provide concise summary\n"
            "- 'Molly, rewrite as bullet points: first X then Y' → format as bullet list\n\n"
            "Output ONLY the result, no explanations."
        )
        processed_text = raw_text
    else:
        # Normal mode: light editing with XML tag wrapping
        system_prompt = (
            "Extract and correct the text inside <dictation> tags. "
            "Fix grammar, punctuation, and spelling. "
            "Remove ONLY hesitation filler words (um, uh, like, you know). "
            "PRESERVE intentional expressions like: laughter (ha ha, haha), reactions (LOL, OMG), "
            "and other meaningful interjections. "
            "Output ONLY the corrected text."
        )
        # Wrap in XML tags to mark it as data, not conversation
        processed_text = f"<dictation>{raw_text}</dictation>"
    
    if vocab_section:
        system_prompt += f"\n\n{vocab_section}"
    
    return system_prompt, processed_text


def clean_with_llm(raw_text: str) -> str:
    """Clean text using LLM via AWS backend."""
    if not raw_text.strip():
        return ""
    
    system_prompt, processed_text = build_system_prompt(raw_text)
    return aws_backend.clean_text(processed_text, system_prompt)


def paste_text(text: str):
    """Copy to clipboard and attempt auto-paste."""
    # Copy to clipboard (this should always work)
    try:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
    except Exception as e:
        print(f"⚠️  Clipboard copy failed: {e}")
        return
    
    # Try to auto-paste, but don't crash if it fails
    result = subprocess.run([
        "osascript", "-e",
        'tell application "System Events" to keystroke "v" using command down'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("⚠️  Auto-paste failed (use Cmd+V manually). Text is in clipboard.")


def run_pipeline():
    """Process transcript: clean with LLM and paste."""
    global transcript_parts
    raw = " ".join(transcript_parts).strip()
    transcript_parts = []
    
    if not raw:
        print("⚠️  No speech detected.")
        return
    
    print(f"🎤 Raw: {raw}")
    cleaned = clean_with_llm(raw)
    print(f"✨ Clean: {cleaned}")
    paste_text(cleaned)
    print("📋 Pasted!\n")


def on_press(key):
    """Handle key press events."""
    global is_recording, transcribe_thread
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r) and not is_recording:
        is_recording = True
        transcript_parts.clear()
        while not audio_queue.empty():
            audio_queue.get_nowait()
        print("🔴 Recording...")
        transcribe_thread = threading.Thread(
            target=lambda: asyncio.run(transcribe_stream()),
            daemon=True
        )
        transcribe_thread.start()


def on_release(key):
    """Handle key release events."""
    global is_recording, transcribe_thread
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r) and is_recording:
        is_recording = False
        print("⏹️  Processing...")

        def wait_and_run():
            if transcribe_thread is not None:
                transcribe_thread.join(timeout=10)
            run_pipeline()

        threading.Thread(target=wait_and_run, daemon=True).start()
    
    if key == keyboard.Key.esc:
        return False


def main():
    """Main entry point."""
    print("🎙️  Voice Inject — hold Control to dictate, release to paste.")
    print("   Press Esc to quit.\n")
    
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK_SAMPLES,
        callback=audio_callback
    ):
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


if __name__ == "__main__":
    main()
