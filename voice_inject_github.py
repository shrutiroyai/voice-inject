#!/usr/bin/env python3
"""Voice Inject — hold Control to dictate, release to paste AI-cleaned text."""

import asyncio
import subprocess
import threading
import queue
import sounddevice as sd
from pynput import keyboard
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from amazon_transcribe.auth import StaticCredentialResolver
import boto3
from vocab import load_vocab_prompt

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SAMPLES = int(SAMPLE_RATE * 0.1)  # 100ms chunks
BEDROCK_MODEL = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
AWS_REGION = "us-west-2"

audio_queue: queue.Queue = queue.Queue()
is_recording = False
transcript_parts: list[str] = []
transcribe_thread: threading.Thread | None = None


def get_credential_resolver():
    """Fetch fresh AWS credentials each time (auto-refreshes if expired)."""
    session = boto3.Session(region_name=AWS_REGION)
    creds = session.get_credentials().get_frozen_credentials()
    return StaticCredentialResolver(
        access_key_id=creds.access_key,
        secret_access_key=creds.secret_key,
        session_token=creds.token or "",
    )


def get_bedrock_client():
    """Create a fresh Bedrock client with current credentials."""
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


class Handler(TranscriptResultStreamHandler):
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if not result.is_partial:
                for alt in result.alternatives:
                    transcript_parts.append(alt.transcript)
                    print(f"  📝 {alt.transcript}")


async def transcribe_stream():
    client = TranscribeStreamingClient(region=AWS_REGION, credential_resolver=get_credential_resolver())
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=SAMPLE_RATE,
        media_encoding="pcm",
    )

    async def send_audio():
        while is_recording or not audio_queue.empty():
            try:
                chunk = audio_queue.get(timeout=0.2)
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
            except queue.Empty:
                if not is_recording:
                    break
        await stream.input_stream.end_stream()

    handler = Handler(stream.output_stream)
    await asyncio.gather(send_audio(), handler.handle_events())


def audio_callback(indata, frames, time_info, status):
    if is_recording:
        audio_queue.put(indata.copy().tobytes())


def clean_with_bedrock(raw_text: str) -> str:
    if not raw_text.strip():
        return ""
    
    # Check if this is a command mode (starts with "Molly")
    is_command_mode = raw_text.lower().strip().startswith("molly")
    
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
    else:
        # Normal mode: light editing with XML tag wrapping
        system_prompt = "Extract and correct the text inside <dictation> tags. Fix grammar, punctuation, and spelling. Remove filler words (um, uh, like). Output ONLY the corrected text."
        # Wrap text in XML tags to mark it as data, not conversation
        raw_text = f"<dictation>{raw_text}</dictation>"
    
    if vocab_section:
        system_prompt += f" {vocab_section}"

    # Get fresh client each time (auto-refreshes credentials if needed)
    client = get_bedrock_client()
    resp = client.converse(
        modelId=BEDROCK_MODEL,
        messages=[{"role": "user", "content": [{"text": raw_text}]}],
        system=[{"text": system_prompt}],
    )
    return resp["output"]["message"]["content"][0]["text"]


def paste_text(text: str):
    # Copy to clipboard (this should always work)
    try:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
    except Exception as e:
        print(f"⚠️  Clipboard copy failed: {e}")
        return
    
    # Try to auto-paste, but don't crash if it fails (text is in clipboard anyway)
    result = subprocess.run([
        "osascript", "-e",
        'tell application "System Events" to keystroke "v" using command down'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("⚠️  Auto-paste failed (use Cmd+V manually). Text is in clipboard.")

def run_pipeline():
    global transcript_parts
    raw = " ".join(transcript_parts).strip()
    transcript_parts = []
    if not raw:
        print("⚠️  No speech detected.")
        return
    print(f"🎤 Raw: {raw}")
    cleaned = clean_with_bedrock(raw)
    print(f"✨ Clean: {cleaned}")
    paste_text(cleaned)
    print("📋 Pasted!\n")


def on_press(key):
    global is_recording, transcribe_thread
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r) and not is_recording:
        is_recording = True
        transcript_parts.clear()
        while not audio_queue.empty():
            audio_queue.get_nowait()
        print("🔴 Recording...")
        transcribe_thread = threading.Thread(target=lambda: asyncio.run(transcribe_stream()), daemon=True)
        transcribe_thread.start()


def on_release(key):
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
    print("🎙️  Voice Inject — hold Control to dictate, release to paste.")
    print("   Press Esc to quit.\n")
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="int16", blocksize=CHUNK_SAMPLES,
                        callback=audio_callback):
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


if __name__ == "__main__":
    main()
