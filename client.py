#!/usr/bin/env python3
"""Voice Inject Client — two modes: Meeting Transcription + Command (auto-paste)."""

import subprocess
import signal
import sys
import sounddevice as sd
from pynput import keyboard
import numpy as np
import time
import asyncio
import websockets
import json
import threading
import queue
from datetime import datetime
from pathlib import Path

SAMPLE_RATE = 16000
CHANNELS = 1

# Global state
meeting_active = False       # Meeting mode: record button in browser
command_recording = False     # Command mode: double-tap Right Option
command_buffer = []           # Audio buffer for command mode
whisper_model = None
last_option_press = 0
DOUBLE_TAP_THRESHOLD = 0.4
continuous_transcriber = None

# Message queue for WebSocket
message_queue = queue.Queue()
ws_connected = False


def load_whisper():
    """Lazy load faster-whisper model."""
    global whisper_model
    if whisper_model is None:
        print("  📦 Loading faster-whisper 'small' model (first time only)...")
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        print("  ✅ Model loaded with CTranslate2 optimization")
    return whisper_model


def audio_callback(indata, frames, time_info, status):
    """Route audio to the active mode's buffer."""
    # Meeting mode: feed continuous transcriber
    if meeting_active and continuous_transcriber is not None:
        continuous_transcriber.feed(indata.copy())
    # Command mode: accumulate for paste
    if command_recording:
        command_buffer.append(indata.copy())


# === COMMAND MODE (double-tap Right Option → transcribe → paste) ===

def paste_text(text: str):
    """Copy to clipboard and auto-paste via Cmd+V."""
    try:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
    except Exception as e:
        print(f"⚠️  Clipboard copy failed: {e}")
        return
    subprocess.run([
        "osascript", "-e",
        'tell application "System Events" to keystroke "v" using command down'
    ], capture_output=True, text=True)


def command_transcribe_and_paste():
    """Transcribe command buffer and paste result."""
    global command_buffer
    if not command_buffer:
        print("⚠️  No audio recorded.")
        return

    audio_data = np.concatenate(command_buffer, axis=0)
    command_buffer = []

    try:
        audio_float = audio_data.astype(np.float32).flatten() / 32768.0
        model = load_whisper()
        segments, info = model.transcribe(audio_float, language="en", beam_size=5)
        text = " ".join([seg.text.strip() for seg in segments]).strip()

        if not text:
            print("⚠️  No speech detected.")
            return

        # Basic cleanup
        if text[0].islower():
            text = text[0].upper() + text[1:]
        if text[-1] not in '.!?':
            text += '.'

        print(f"✨ {text}")
        paste_text(text)
        print("📋 Pasted!\n")
    except Exception as e:
        print(f"❌ Command transcription error: {e}")


def toggle_command():
    """Toggle command mode recording (double-tap Right Option)."""
    global command_recording, command_buffer

    if not command_recording:
        command_recording = True
        command_buffer = []
        print("🎤 Command mode: recording...")
        message_queue.put({"type": "status", "recording": True, "mode": "command"})
    else:
        command_recording = False
        print("⏹️  Command mode: transcribing...")
        message_queue.put({"type": "status", "recording": False, "mode": "command"})
        threading.Thread(target=command_transcribe_and_paste, daemon=True).start()


# === MEETING MODE (Record button → continuous VAD transcription → file) ===

def toggle_meeting():
    """Toggle meeting mode (Record button in browser)."""
    global meeting_active

    if not meeting_active:
        meeting_active = True
        continuous_transcriber.start_session()
        print("🔴 Meeting mode: recording started")
        message_queue.put({"type": "status", "recording": True, "mode": "meeting"})
    else:
        meeting_active = False
        continuous_transcriber.end_session()
        print("⏹️  Meeting mode: session ended")
        message_queue.put({"type": "status", "recording": False, "mode": "meeting"})


# === KEYBOARD HANDLER ===

def on_press(key):
    """Double-tap Right Option → command mode toggle."""
    global last_option_press

    if key == keyboard.Key.alt_r:
        current_time = time.time()
        if (current_time - last_option_press) < DOUBLE_TAP_THRESHOLD:
            toggle_command()
            last_option_press = 0
        else:
            last_option_press = current_time


def on_release(key):
    pass


# === WEBSOCKET ===

async def websocket_client():
    """WebSocket client that runs in background thread."""
    global ws_connected
    uri = "ws://localhost:3000/ws"

    while True:
        try:
            async with websockets.connect(uri) as websocket:
                ws_connected = True
                print("✅ Connected to server\n")

                async def send_messages():
                    while True:
                        try:
                            while not message_queue.empty():
                                msg = message_queue.get_nowait()
                                await websocket.send(json.dumps(msg))
                            await asyncio.sleep(0.1)
                        except Exception as e:
                            print(f"Send error: {e}")
                            break

                async def receive_messages():
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            if data.get("type") == "toggle_recording":
                                toggle_meeting()
                        except Exception as e:
                            print(f"Receive error: {e}")

                await asyncio.gather(send_messages(), receive_messages(), return_exceptions=True)

        except Exception:
            ws_connected = False
            await asyncio.sleep(5)


def start_websocket_thread():
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(websocket_client())
    threading.Thread(target=run, daemon=True).start()


# === CONTINUOUS TRANSCRIBER (VAD-based, for meeting mode) ===

class ContinuousTranscriber:
    """VAD-based transcriber for meeting mode. Creates per-session transcript files."""

    SILENCE_THRESHOLD = 0.8
    MAX_SEGMENT_DURATION = 30
    MIN_SPEECH_ENERGY = 100
    TRANSCRIPTS_DIR = Path("transcripts")

    def __init__(self, model, sample_rate: int, message_queue: queue.Queue):
        self._model = model
        self._sample_rate = sample_rate
        self._message_queue = message_queue
        self._buffer: list = []
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._segment_count: int = 0
        self._session_start_time: datetime | None = None
        self._silence_frames: int = 0
        self._has_speech: bool = False
        self._session_file: Path | None = None
        # VAD
        import webrtcvad
        self._vad = webrtcvad.Vad(3)
        self._frame_duration_ms = 30
        self._frame_size = int(self._sample_rate * self._frame_duration_ms / 1000)
        self._silence_frames_threshold = int(self.SILENCE_THRESHOLD * 1000 / self._frame_duration_ms)

    def start_session(self) -> None:
        """Start a new meeting session — creates a new transcript file."""
        self._stop_event.clear()
        self._session_start_time = datetime.now()
        self._segment_count = 0
        self._silence_frames = 0
        self._has_speech = False

        self.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = self._session_start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self._session_file = self.TRANSCRIPTS_DIR / f"session_{timestamp}.txt"

        with open(self._session_file, "w", encoding="utf-8") as f:
            f.write(f"# Meeting Transcript\n")
            f.write(f"# Started: {self._session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 40 + "\n")

        self._message_queue.put({
            "type": "session_started",
            "session_file": str(self._session_file)
        })

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def end_session(self) -> None:
        """End the current meeting session."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=15)
            self._thread = None

        # Transcribe remaining buffer
        with self._buffer_lock:
            remaining = self._buffer
            self._buffer = []

        if remaining:
            audio_data = np.concatenate(remaining, axis=0)
            if audio_data.shape[0] >= self._sample_rate:
                text = self._transcribe(audio_data)
                if text:
                    self._segment_count += 1
                    self._write_segment(text)
                    self._message_queue.put({
                        "type": "transcript_segment",
                        "text": text,
                        "timestamp": datetime.now().isoformat()
                    })

        # Write footer
        if self._session_file:
            duration = (datetime.now() - self._session_start_time).total_seconds()
            with open(self._session_file, "a", encoding="utf-8") as f:
                f.write("-" * 40 + "\n")
                f.write(f"# Duration: {self._format_duration(duration)}\n")
                f.write(f"# Segments: {self._segment_count}\n")

            self._message_queue.put({
                "type": "session_ended",
                "duration_seconds": duration,
                "segment_count": self._segment_count,
                "session_file": str(self._session_file)
            })

    def feed(self, audio_chunk: np.ndarray) -> None:
        """Feed audio from callback."""
        with self._buffer_lock:
            self._buffer.append(audio_chunk)

    def _run_loop(self) -> None:
        """VAD loop: cut on silence, transcribe, write."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.1)

            with self._buffer_lock:
                if not self._buffer:
                    continue
                current_buffer = list(self._buffer)

            total_audio = np.concatenate(current_buffer, axis=0).flatten()
            if len(total_audio) < self._frame_size:
                continue

            # Check last frame for speech/silence
            last_frame = total_audio[-self._frame_size:]
            frame_bytes = last_frame.astype(np.int16).tobytes()
            try:
                is_speech = self._vad.is_speech(frame_bytes, self._sample_rate)
            except Exception:
                is_speech = True

            if is_speech:
                self._has_speech = True
                self._silence_frames = 0
            else:
                self._silence_frames += 1

            total_duration = len(total_audio) / self._sample_rate
            should_cut = (
                (self._has_speech and self._silence_frames >= self._silence_frames_threshold)
                or total_duration >= self.MAX_SEGMENT_DURATION
            )

            if should_cut and total_duration >= 0.5:
                with self._buffer_lock:
                    captured = self._buffer
                    self._buffer = []
                audio_data = np.concatenate(captured, axis=0)
                self._silence_frames = 0
                self._has_speech = False

                rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
                if rms < self.MIN_SPEECH_ENERGY:
                    continue

                text = self._transcribe(audio_data)
                if text:
                    self._segment_count += 1
                    print(f"📝 [{self._segment_count}] {text}")
                    self._write_segment(text)
                    self._message_queue.put({
                        "type": "transcript_segment",
                        "text": text,
                        "timestamp": datetime.now().isoformat()
                    })

    def _transcribe(self, audio_data: np.ndarray) -> str | None:
        """Transcribe audio. Returns text or None."""
        try:
            audio_float = audio_data.astype(np.float32).flatten() / 32768.0
            segments, _ = self._model.transcribe(audio_float, language="en", beam_size=5)
            text = " ".join([s.text.strip() for s in segments]).strip()
            return text if text else None
        except Exception as e:
            print(f"⚠️  Transcription error: {e}")
            return None

    def _write_segment(self, text: str) -> None:
        """Append segment to session file."""
        try:
            elapsed = (datetime.now() - self._session_start_time).total_seconds()
            ts = self._format_duration(elapsed)
            with open(self._session_file, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {text}\n")
        except Exception as e:
            print(f"⚠️  Write failed: {e}")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        s = int(seconds)
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# === MAIN ===

def main():
    global continuous_transcriber

    print("🎙️  Voice Inject")
    print("   Meeting mode: Click Record in browser (VAD-based, saves transcript)")
    print("   Command mode: Double-tap Right Option ⌥ (transcribe → paste)")
    print("   Press Ctrl+C to quit.\n")

    load_whisper()
    print("✅ Whisper model loaded\n")

    start_websocket_thread()

    def sigint_handler(signum, frame):
        print("\n⏹️  Shutting down...")
        if meeting_active and continuous_transcriber:
            continuous_transcriber.end_session()
        sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)

    continuous_transcriber = ContinuousTranscriber(
        model=whisper_model,
        sample_rate=SAMPLE_RATE,
        message_queue=message_queue,
    )

    print("🎙️  Ready\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="int16", callback=audio_callback):
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


if __name__ == "__main__":
    main()
