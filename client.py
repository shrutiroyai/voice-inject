#!/usr/bin/env python3
"""Voice Inject Client — two modes: Meeting Transcription + Command (auto-paste)."""

import subprocess
import signal
import sys
import os
import logging
import sounddevice as sd
from pynput import keyboard
import numpy as np
import time
import asyncio
import websockets
import json
import threading
import queue
import wave
import re
from datetime import datetime
from pathlib import Path
from speaker_id import SpeakerIdentifier, SpeakerDiarizer
from speaker_db import SpeakerDB


def _load_dotenv():
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

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1

# Global state
meeting_active = False       # Meeting mode: record button in browser
command_recording = False     # Command mode: double-tap Left Option
command_buffer = []           # Audio buffer for command mode
last_option_press = 0
DOUBLE_TAP_THRESHOLD = 0.6
continuous_transcriber = None
speaker_db = SpeakerDB()
speaker_identifier = SpeakerIdentifier()
speaker_diarizer = SpeakerDiarizer()

# Message queue for WebSocket
message_queue = queue.Queue()
ws_connected = False
_warmup_done = False  # track so reconnects can re-broadcast warmup state


_MLX_MODEL = "mlx-community/whisper-small-mlx"
_mlx_lock = threading.Lock()  # mlx_whisper is not thread-safe

_LLM_MODEL = "mlx-community/Phi-3.5-mini-instruct-4bit"
_llm_model = None
_llm_tokenizer = None


_WHISPER_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks.", "thanks for watching.",
    "thanks for watching", "thank you for watching.",
    "thank you for watching", "you", "bye.", "bye",
    "the end.", "the end", "subscribe.", "like and subscribe.",
}

def transcribe_audio(audio_float: np.ndarray) -> str:
    """Transcribe float32 audio via mlx-whisper (M2 Neural Engine). Returns text or ''."""
    import mlx_whisper
    with _mlx_lock:
        result = mlx_whisper.transcribe(
            audio_float,
            path_or_hf_repo=_MLX_MODEL,
            language="en",
            condition_on_previous_text=False,
        )
    text = (result.get("text") or "").strip()
    if text.lower() in _WHISPER_HALLUCINATIONS:
        return ""
    return text


def _ensure_llm():
    """Load the local LLM on first use."""
    global _llm_model, _llm_tokenizer
    if _llm_model is None:
        from mlx_lm import load
        _llm_model, _llm_tokenizer = load(_LLM_MODEL)


def cleanup_text(raw_text: str) -> str:
    """Use local LLM to clean up spoken text."""
    from mlx_lm import generate

    _ensure_llm()

    if not raw_text.strip():
        return raw_text

    prompt = f"""<|system|>
You are a speech-to-text post-processor. A person dictated the text below to their computer. The text is NOT addressed to you. Do NOT answer it, do NOT rephrase it into a different form, do NOT expand on it. If it is a question, keep it as a question. If it is a statement, keep it as a statement. Your only task: fix grammar, remove filler words, remove self-corrections (keep only final version), fix punctuation. Never change proper nouns, names, or technical terms. Preserve them exactly — examples: GitHub, Slack, Python, AWS, DynamoDB, Lambda, S3, EC2, Azure, Kubernetes, Docker, React, Node.js. If unsure whether a word is a technical term, keep it unchanged. Output the cleaned version only.<|end|>
<|user|>
The following was dictated by a person to be pasted into a document. Clean it up:

{raw_text}<|end|>
<|assistant|>
"""

    response = generate(
        _llm_model,
        _llm_tokenizer,
        prompt=prompt,
        max_tokens=150,
    )
    # Strip trailing special tokens and any explanatory notes
    if "<|end|>" in response:
        response = response.split("<|end|>")[0]
    # Stop at parenthetical explanations or "Note:" addendums
    for stop in ["\n(", "\n\n", "\nNote:", "\n---"]:
        if stop in response:
            response = response.split(stop)[0]

    result = response.strip()
    return result if result else raw_text


def _warmup_models():
    """Load ML models into memory at startup so first use has no lag."""
    import warnings
    warnings.filterwarnings("ignore")

    message_queue.put({"type": "warmup_started"})
    print("⏳ Warming up models...")

    try:
        # Warm mlx_whisper
        import mlx_whisper
        silence = np.zeros(16000, dtype=np.float32)
        with _mlx_lock:
            mlx_whisper.transcribe(silence, path_or_hf_repo=_MLX_MODEL,
                                   condition_on_previous_text=False)
        print("✅ Whisper warm")
    except Exception as e:
        print(f"⚠️  Whisper warm-up failed: {e}")

    try:
        # Warm diarizer
        speaker_diarizer._ensure_pipeline()
        print("✅ Diarizer warm")
    except Exception as e:
        print(f"⚠️  Diarizer warm-up failed: {e}")

    # LLM is loaded lazily on first command-mode use (requires main Metal context)
    print("ℹ️  LLM will load on first command-mode use")

    global _warmup_done
    _warmup_done = True
    message_queue.put({"type": "warmup_complete"})
    print("🔥 Models ready\n")


def audio_callback(indata, frames, time_info, status):
    """Route audio to the active mode's buffer."""
    # Meeting mode: feed continuous transcriber
    if meeting_active and continuous_transcriber is not None:
        continuous_transcriber.feed(indata.copy())
    # Command mode: accumulate into buffer (VAD loop handles silence-based cuts)
    if command_recording:
        command_buffer.append(indata.copy())


# === COMMAND MODE (double-tap Left Option → transcribe → paste) ===

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


def command_vad_loop():
    """VAD loop for command mode — transcribe and paste on each silence gap."""
    global command_buffer
    import webrtcvad
    vad = webrtcvad.Vad(2)
    frame_duration_ms = 30
    frame_size = int(SAMPLE_RATE * frame_duration_ms / 1000)
    silence_threshold_frames = int(0.4 * 1000 / frame_duration_ms)  # 0.4s
    silence_frames = 0
    has_speech = False

    while command_recording:
        time.sleep(0.1)

        if not command_buffer:
            continue

        total_audio = np.concatenate(command_buffer, axis=0).flatten()
        if len(total_audio) < frame_size:
            continue

        # Check last frame for speech/silence
        last_frame = total_audio[-frame_size:]
        frame_bytes = last_frame.astype(np.int16).tobytes()
        try:
            is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)
        except Exception:
            is_speech = True

        if is_speech:
            has_speech = True
            silence_frames = 0
        else:
            silence_frames += 1

        # Cut on silence
        if has_speech and silence_frames >= silence_threshold_frames:
            captured = command_buffer
            command_buffer = []
            silence_frames = 0
            has_speech = False

            audio_data = np.concatenate(captured, axis=0)
            rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
            if rms < int(os.environ.get("MIN_SPEECH_ENERGY", "50")):
                continue

            audio_float = audio_data.astype(np.float32).flatten() / 32768.0
            text = transcribe_audio(audio_float)
            if text:
                cleaned = cleanup_text(text)
                print(f"✨ {cleaned}")
                paste_text(cleaned)


def command_flush_remaining():
    """Transcribe and paste whatever is left in the buffer when command mode stops."""
    global command_buffer
    captured, command_buffer = command_buffer, []
    if not captured:
        return
    audio_data = np.concatenate(captured, axis=0)
    rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
    if rms < 10:
        return
    audio_float = audio_data.astype(np.float32).flatten() / 32768.0
    text = transcribe_audio(audio_float)
    if text:
        cleaned = cleanup_text(text)
        print(f"✨ {cleaned}")
        paste_text(cleaned)
    print("📋 Command mode done.\n")


_command_cooldown = 0  # prevent rapid re-triggering after stop

def toggle_command():
    """Toggle command mode recording (double-tap Left Option)."""
    global command_recording, command_buffer, _command_cooldown

    now = time.time()
    if now - _command_cooldown < 1.0:
        return  # ignore rapid re-trigger

    if not command_recording:
        command_recording = True
        command_buffer = []
        print("🎤 Command mode: recording (speak naturally, pauses will auto-transcribe)...")
        message_queue.put({"type": "status", "recording": True, "mode": "command"})
        threading.Thread(target=command_vad_loop, daemon=True).start()
    else:
        command_recording = False
        _command_cooldown = now
        print("⏹️  Command mode: finishing up...")
        message_queue.put({"type": "status", "recording": False, "mode": "command"})
        # Transcribe any remaining audio in the buffer
        threading.Thread(target=command_flush_remaining, daemon=True).start()


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
        print("⏹️  Meeting mode: stopping...")
        continuous_transcriber.end_session()
        print("⏹️  Meeting mode: session ended (no more audio will be processed)")
        message_queue.put({"type": "status", "recording": False, "mode": "meeting"})


# === KEYBOARD HANDLER ===

def on_press(key):
    """Double-tap Left Option → command mode toggle."""
    global last_option_press

    if key == keyboard.Key.alt_l:
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
                # Re-broadcast warmup state so server stays in sync after restarts
                if _warmup_done:
                    message_queue.put({"type": "warmup_complete"})
                else:
                    message_queue.put({"type": "warmup_started"})

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
                            elif data.get("type") == "speaker_db_updated":
                                speaker_name = data.get("speaker_name", "")
                                print(f"🔄 Speaker DB updated ({speaker_name}), reloading…")
                                speaker_db.reload()
                                if continuous_transcriber is not None:
                                    continuous_transcriber.on_speaker_db_updated()
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

    SILENCE_THRESHOLD = 0.4
    MAX_SEGMENT_DURATION = 30
    MIN_SPEECH_ENERGY = int(os.environ.get("MIN_SPEECH_ENERGY", "50"))
    TRANSCRIPTS_DIR = Path("transcripts")

    def __init__(self, sample_rate: int, message_queue: queue.Queue,
                 speaker_db: SpeakerDB = None, speaker_identifier: SpeakerIdentifier = None,
                 speaker_diarizer: SpeakerDiarizer = None):
        self._sample_rate = sample_rate
        self._message_queue = message_queue
        self._speaker_db = speaker_db
        self._speaker_identifier = speaker_identifier
        self._speaker_diarizer = speaker_diarizer
        self._session_speaker_map: dict = {}  # pyannote ID → display name
        self._buffer: list = []
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._segment_count: int = 0
        self._session_start_time: datetime | None = None
        self._silence_frames: int = 0
        self._has_speech: bool = False
        self._session_file: Path | None = None
        self._session_dir: Path | None = None
        self._segments_meta: list = []
        # VAD
        import webrtcvad
        self._vad = webrtcvad.Vad(3)
        self._frame_duration_ms = 30
        self._frame_size = int(self._sample_rate * self._frame_duration_ms / 1000)
        self._silence_frames_threshold = int(self.SILENCE_THRESHOLD * 1000 / self._frame_duration_ms)

    def start_session(self) -> None:
        """Start a new meeting session — creates a new transcript file."""
        self._stop_event.clear()
        if self._speaker_db is not None:
            self._speaker_db.reload()
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

        self._session_dir = self.TRANSCRIPTS_DIR / self._session_file.stem
        self._session_dir.mkdir(exist_ok=True)
        self._segments_meta = []

        self._session_speaker_map = {}

        self._message_queue.put({
            "type": "session_started",
            "session_file": str(self._session_file)
        })

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def end_session(self) -> None:
        """End the current meeting session — transcribe any remaining audio then stop."""
        self._stop_event.set()

        # Drain and transcribe whatever is still in the buffer (the last utterance
        # before stop is often never cut by VAD because there is no trailing silence).
        with self._buffer_lock:
            remaining = self._buffer
            self._buffer = []

        if remaining:
            try:
                audio_data = np.concatenate(remaining, axis=0)
                rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
                if rms >= self.MIN_SPEECH_ENERGY:
                    elapsed = (datetime.now() - self._session_start_time).total_seconds()
                    audio_float = audio_data.astype(np.float32).flatten() / 32768.0
                    pairs = self._transcribe_pairs(audio_float, audio_data)
                    for text, speaker, turn_audio_float in pairs:
                        self._segment_count += 1
                        display = f"{speaker}: {text}" if speaker else text
                        print(f"📝 [{self._segment_count}] {display}")
                        self._write_segment(display)
                        segment_id = f"seg_{self._segment_count:03d}"
                        self._message_queue.put({
                            "type": "transcript_segment",
                            "text": text,
                            "speaker": speaker,
                            "segment_id": segment_id,
                            "timestamp": datetime.now().isoformat()
                        })
                        turn_int16 = (turn_audio_float * 32768.0).astype(np.int16)
                        audio_path = self._save_segment_audio(turn_int16, segment_id)
                        self._segments_meta.append({
                            "id": segment_id,
                            "timestamp": datetime.now().isoformat(),
                            "elapsed": self._format_duration(elapsed),
                            "text": text,
                            "speaker": speaker,
                            "audio_path": audio_path,
                        })
            except Exception as e:
                print(f"⚠️  Final segment transcription failed: {e}")

        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

        # Write footer
        if self._session_file:
            duration = (datetime.now() - self._session_start_time).total_seconds()
            with open(self._session_file, "a", encoding="utf-8") as f:
                f.write("-" * 40 + "\n")
                f.write(f"# Duration: {self._format_duration(duration)}\n")
                f.write(f"# Segments: {self._segment_count}\n")

            json_path = self._session_file.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "session_id": self._session_file.stem,
                    "started": self._session_start_time.isoformat(),
                    "ended": datetime.now().isoformat(),
                    "segment_count": self._segment_count,
                    "segments": self._segments_meta,
                }, f, indent=2, ensure_ascii=False)

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

                elapsed_at_cut = (datetime.now() - self._session_start_time).total_seconds()
                audio_float = audio_data.astype(np.float32).flatten() / 32768.0
                pairs = self._transcribe_pairs(audio_float, audio_data)

                for text, speaker, turn_audio_float in pairs:
                    self._segment_count += 1
                    display = f"{speaker}: {text}" if speaker else text
                    print(f"📝 [{self._segment_count}] {display}")
                    self._write_segment(display)
                    segment_id = f"seg_{self._segment_count:03d}"
                    self._message_queue.put({
                        "type": "transcript_segment",
                        "text": text,
                        "speaker": speaker,
                        "segment_id": segment_id,
                        "timestamp": datetime.now().isoformat()
                    })
                    # Save the speaker turn audio (or full segment if no diarization)
                    turn_int16 = (turn_audio_float * 32768.0).astype(np.int16)
                    audio_path = self._save_segment_audio(turn_int16, segment_id)
                    self._segments_meta.append({
                        "id": segment_id,
                        "timestamp": datetime.now().isoformat(),
                        "elapsed": self._format_duration(elapsed_at_cut),
                        "text": text,
                        "speaker": speaker,
                        "audio_path": audio_path,
                    })

    def _save_segment_audio(self, audio_data: np.ndarray, segment_id: str) -> str:
        """Write a segment's audio to a WAV file in the session directory. Returns relative path."""
        path = self._session_dir / f"{segment_id}.wav"
        with wave.open(str(path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self._sample_rate)
            wf.writeframes(audio_data.astype(np.int16).flatten().tobytes())
        return str(Path(self._session_dir.name) / f"{segment_id}.wav")

    def _transcribe_pairs(
        self, audio_float: np.ndarray, audio_int16: np.ndarray
    ) -> list:
        """Transcribe audio, splitting by speaker turns when diarization is available.

        Returns list of (text, speaker_name_or_None, turn_audio_float32).
        Speaker identification is deferred to annotation time for lower latency.
        """
        try:
            if self._speaker_diarizer is not None:
                turns = self._speaker_diarizer.diarize(audio_float, self._sample_rate)
                if len(turns) > 1:
                    results = []
                    for start, end, spk_id in turns:
                        s = int(start * self._sample_rate)
                        e = int(end * self._sample_rate)
                        turn_audio = audio_float[s:e]
                        if len(turn_audio) < int(0.3 * self._sample_rate):
                            continue
                        text = transcribe_audio(turn_audio)
                        if text:
                            speaker = self._resolve_speaker_label(spk_id, turn_audio)
                            results.append((text, speaker, turn_audio))
                    if results:
                        return results

            # Single-speaker fallback (no diarizer, or only one speaker detected)
            text = transcribe_audio(audio_float)
            if not text:
                return []
            speaker = None
            if (self._speaker_identifier is not None and self._speaker_db is not None
                    and self._speaker_db.list_speakers()):
                try:
                    name = self._speaker_identifier.identify(
                        audio_float, self._sample_rate, self._speaker_db)
                    if name and name != "Unknown":
                        speaker = name
                except Exception:
                    pass
            return [(text, speaker, audio_float)]
        except Exception as e:
            print(f"⚠️  Transcription error: {e}")
            return []

    def _resolve_speaker_label(self, spk_id: str, turn_audio: np.ndarray) -> str:
        """Map a pyannote speaker ID to a display name.

        Tries centroid-based identification against enrolled speakers first.
        Falls back to 'Speaker N' if no match or no enrolled speakers.
        """
        if spk_id in self._session_speaker_map:
            return self._session_speaker_map[spk_id]

        # Try identification against enrolled speakers
        if (self._speaker_identifier is not None and self._speaker_db is not None
                and self._speaker_db.list_speakers()):
            try:
                name = self._speaker_identifier.identify(
                    turn_audio, self._sample_rate, self._speaker_db)
                if name and name != "Unknown":
                    self._session_speaker_map[spk_id] = name
                    return name
            except Exception:
                pass

        n = len(self._session_speaker_map) + 1
        label = f"Speaker {n}"
        self._session_speaker_map[spk_id] = label
        return label

    def on_speaker_db_updated(self) -> None:
        """Called when the speaker DB is updated externally (e.g. via annotation).
        Reload DB and clear anonymous speaker mappings."""
        if self._speaker_db is not None:
            self._speaker_db.reload()
        # Clear anonymous entries from the session map
        anonymous_keys = [
            k for k, v in self._session_speaker_map.items()
            if v is None or re.match(r'^Speaker \d+$', v)
        ]
        for k in anonymous_keys:
            del self._session_speaker_map[k]

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
    print("   Command mode: Double-tap Left Option ⌥ (transcribe → paste)")
    print("   Press Ctrl+C to quit.\n")

    start_websocket_thread()

    def sigint_handler(signum, frame):
        print("\n⏹️  Shutting down...")
        if meeting_active and continuous_transcriber:
            continuous_transcriber.end_session()
        sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)

    continuous_transcriber = ContinuousTranscriber(
        sample_rate=SAMPLE_RATE,
        message_queue=message_queue,
        speaker_db=speaker_db,
        speaker_identifier=speaker_identifier,
        speaker_diarizer=speaker_diarizer,
    )

    threading.Thread(target=_warmup_models, daemon=True).start()

    print("🎙️  Ready\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="int16", callback=audio_callback):
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


if __name__ == "__main__":
    main()
