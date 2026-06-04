# Build a Voice-to-Text Dictation Tool That Pastes AI-Cleaned Text Into Any App

*Hold Control, speak, release — cleaned text appears wherever your cursor is. Let's build it in ~200 lines of Python.*

---

## The Problem

I type a lot. Slack messages, documents, code comments, emails. But sometimes my thoughts flow faster than my fingers. I wanted a tool where I could just *talk* and have polished text appear — no "um"s, no grammar issues, correct spelling of domain-specific terms.

Commercial dictation tools exist, but they're either subscription-based, don't understand my company's acronyms, or can't be customized. So I built my own.

## The Architecture

```
┌─────────────┐     ┌──────────────────────────┐     ┌────────────────┐
│  Microphone │────▶│  AWS Transcribe Streaming │────▶│ Amazon Bedrock │
│  (16kHz PCM)│     │  (real-time WebSocket)    │     │ (Claude Haiku) │
└─────────────┘     └──────────────────────────┘     └───────┬────────┘
                                                              │
      ┌───────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────┐     ┌──────────────────────┐
│ Custom Vocab │     │  Auto-paste into      │
│ (YAML rules) │────▶│  any focused app      │
└──────────────┘     └──────────────────────┘
```

Three distinct stages, each solving a different problem:

1. **Hearing** — Convert speech to raw text (AWS Transcribe)
2. **Understanding** — Clean, format, and apply domain vocabulary (Bedrock LLM)
3. **Acting** — Put the result where the user needs it (clipboard + paste)

Let me teach you each concept.

---

## Concept 1: Streaming Transcription (Why It Matters)

Most speech-to-text demos work like this: record audio → upload file → get text back. That adds seconds of latency.

**Streaming transcription** is different. You open a persistent WebSocket connection and send audio chunks *while the user is still speaking*. AWS Transcribe processes them in real-time and returns partial results that refine into final results.

```python
# Audio is captured in 100ms chunks and sent immediately
SAMPLE_RATE = 16000  # 16kHz — standard for speech
CHUNK_SAMPLES = int(SAMPLE_RATE * 0.1)  # 100ms = 1600 samples

# The stream stays open the entire time you're holding the key
stream = await client.start_stream_transcription(
    language_code="en-US",
    media_sample_rate_hz=SAMPLE_RATE,
    media_encoding="pcm",
)
```

**The key insight:** By the time you release the Control key, most of your speech has *already been transcribed*. You're only waiting for the last 1-2 seconds to finalize.

---

## Concept 2: LLM as a Post-Processor (Not a Transcriber)

Here's a common misconception: "Why not just use an LLM for transcription?"

Because LLMs aren't real-time. They need the full audio file, process it in one shot, and are slower for pure transcription. But they're *brilliant* at understanding context and cleaning text.

So we separate concerns:
- **Transcribe** does the heavy lifting of converting sound waves to words (fast, streaming)
- **Bedrock Claude** does the light lifting of fixing grammar and applying rules (fast, one API call)

```python
# Normal mode: minimal cleanup
system_prompt = (
    "You are a transcription formatter doing MINIMAL cleanup.\n"
    "ONLY remove filler words, add punctuation, capitalize, and apply vocab corrections.\n"
    "DO NOT rephrase or restructure. Preserve the speaker's exact words."
)
```

**Example:**
- Raw: *"um so like we need to look at the y o y numbers for um gen bi"*
- Cleaned: *"We need to look at the YoY numbers for GenBI."*

---

## Concept 3: Command Mode with Wake Word

The system has two modes:

1. **Normal dictation** — Light cleanup only (remove fillers, add punctuation, apply vocab)
2. **Command mode** — Say "Molly" to trigger full LLM assistance

```python
is_command_mode = raw_text.lower().strip().startswith("molly")

if is_command_mode:
    # Examples: "Molly, make this more formal", "Molly, summarize this"
    system_prompt = "You are Molly, a helpful dictation assistant..."
else:
    # Normal dictation
    system_prompt = "Minimal cleanup only..."
```

**Examples:**
- "the TY numbers look good" → "The TY numbers look good."
- "Molly, make this more formal: the numbers look good" → "The metrics are performing well."

---

## Concept 4: Custom Vocabulary Without Fine-Tuning

Every domain has jargon. Instead of fine-tuning a model (expensive, slow, overkill), we inject vocabulary rules directly into the prompt:

```yaml
# ~/.voice-inject/vocab.yaml
corrections:
  - hear: ["year over year", "y o y", "yoy"]
    use: "YoY"
  - hear: ["t y", "tv", "tee why", "this year"]
    use: "TY"
  - hear: ["gen bi", "jenbi"]
    use: "GenBI"
```

These get appended to the system prompt dynamically:

```
VOCABULARY RULES (always use these exact spellings):
- "year over year" / "y o y" / "yoy" → YoY
- "t y" / "tv" / "tee why" → TY
```

**No model retraining. No restart needed.** Edit the YAML, and the next dictation uses the new rules.

---

## Concept 5: Push-to-Talk with Global Key Hooks

Instead of always-on listening (creepy, battery-draining), we use a physical key as the trigger:

- **Ctrl press** → Start recording + open Transcribe stream
- **Ctrl release** → Stop recording, wait for final transcript, clean, paste

The `pynput` library gives us global keyboard hooks that work even when the terminal isn't focused:

```python
from pynput import keyboard

def on_press(key):
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        start_recording()

def on_release(key):
    if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        stop_and_process()
```

---

## The Full Pipeline in Action

```
1. [Ctrl pressed]  → Mic starts, WebSocket opens
2. [Speaking...]   → Audio chunks stream to AWS (100ms each)
3. [Ctrl released] → Audio stops, stream closes, final transcript arrives
4. [~1 second]    → LLM cleans text + applies vocab
5. [Instant]      → Text copied to clipboard + Cmd+V auto-pasted
```

Total latency: **~2 seconds** from releasing the key to text appearing.

---

## Cost Analysis

**For 100 dictations/day (avg 20 seconds each):**
- AWS Transcribe Streaming: ~$33/month
- Bedrock Haiku cleanup: ~$3/month
- **Total: ~$36/month**

**Why it costs more than consumer apps:**
- You're paying AWS retail prices (no volume discount)
- Streaming transcription costs 4x more than file-based
- You get real-time response and complete data privacy

**Cheaper alternatives:**
- Azure Speech-to-Text: ~$24/month (33% cheaper, same features)
- OpenAI Whisper API (file-based): ~$11/month (loses real-time)
- Self-hosted Whisper: ~$0 (runs locally, uses CPU)

---

## What I Learned

1. **Separate hearing from understanding.** Streaming STT + LLM post-processing beats either alone.
2. **Prompt injection > fine-tuning** for vocabulary. It's faster to ship and easier to iterate.
3. **Two-mode design is powerful.** Normal dictation for speed, command mode for creativity.
4. **Threading is the hard part.** The async/threading interaction between Transcribe streaming and the rest of the app was where all the bugs lived.
5. **Haiku over Sonnet for latency-sensitive tasks.** Claude 3.5 Haiku gives 90% of the quality at 5x the speed for simple cleanup tasks.

---

## Try It Yourself

The full project is ~200 lines of Python across two files. You need:
- macOS (for `pbcopy` and `osascript` paste)
- Python 3.12+
- AWS credentials with Transcribe + Bedrock access

```bash
git clone <repo> && cd voice-inject
./install.sh
python voice_inject.py
```

Hold Control, speak, release — your cleaned text appears!

---

*Built with: Python, AWS Transcribe Streaming, Amazon Bedrock (Claude 3.5 Haiku), pynput, sounddevice*
