#!/usr/bin/env python3
"""Voice Inject Server — runs on dev desktop, handles Transcribe + LLM cleaning."""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml
import json
import asyncio
import sys
from pathlib import Path
import boto3
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from amazon_transcribe.auth import StaticCredentialResolver

app = FastAPI()

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config paths
CONFIG_DIR = Path.home() / ".voice-inject"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
VOCAB_PATH = CONFIG_DIR / "vocab.yaml"

CONFIG_DIR.mkdir(exist_ok=True)

# AWS config
BEDROCK_MODEL = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
AWS_REGION = "us-west-2"


def load_config():
    """Load configuration from config.yaml."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {
        "creativity_level": 1,
        "tone": "neutral",
        "model": BEDROCK_MODEL,
        "region": AWS_REGION
    }


def save_config(config: dict):
    """Save configuration to config.yaml."""
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_vocab():
    """Load vocabulary from vocab.yaml."""
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH) as f:
            data = yaml.safe_load(f) or {}
            corrections = data.get("corrections", [])
            print(f"[DEBUG] Loaded {len(corrections)} vocab corrections from {VOCAB_PATH}", flush=True)
            sys.stdout.flush()
            return corrections
    print(f"[DEBUG] Vocab file not found at {VOCAB_PATH}", flush=True)
    sys.stdout.flush()
    return []


def save_vocab(corrections: list):
    """Save vocabulary to vocab.yaml."""
    with open(VOCAB_PATH, "w") as f:
        yaml.dump({"corrections": corrections}, f, default_flow_style=False, allow_unicode=True)


def load_vocab_prompt() -> str:
    """Generate vocabulary prompt section."""
    corrections = load_vocab()
    if not corrections:
        return ""
    lines = ["VOCABULARY RULES (always use these exact spellings):"]
    for entry in corrections:
        variants = " / ".join(f'"{h}"' for h in entry["hear"])
        lines.append(f'- {variants} → {entry["use"]}')
    return "\n".join(lines)


def get_credential_resolver():
    """Get AWS credentials for Transcribe streaming."""
    session = boto3.Session(region_name=AWS_REGION)
    creds = session.get_credentials().get_frozen_credentials()
    return StaticCredentialResolver(
        access_key_id=creds.access_key,
        secret_access_key=creds.secret_key,
        session_token=creds.token or "",
    )


def get_bedrock_client():
    """Create a fresh Bedrock client with instance profile credentials."""
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def clean_with_llm(raw_text: str, creativity_level: int = 1, tone: str = "neutral") -> str:
    """Clean transcribed text using Bedrock LLM."""
    if not raw_text.strip():
        return ""
    
    # Check if command mode
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
        prompt_text = raw_text
    else:
        # Normal mode: use XML tag wrapping to prevent conversational responses
        if creativity_level == 1:
            system_prompt = "Extract and correct the text inside <dictation> tags. Fix grammar, punctuation, and spelling. Remove ONLY hesitation filler words (um, uh, like, you know). PRESERVE intentional expressions like: laughter (ha ha, haha), reactions (LOL, OMG), and other meaningful interjections. Output ONLY the corrected text."
        elif creativity_level == 2:
            system_prompt = f"Extract text from <dictation> tags. Fix grammar and spelling. Make concise. Remove ONLY hesitation fillers (um, uh, like, you know) but PRESERVE intentional expressions (ha ha, LOL, OMG). {tone.capitalize()} tone. Output ONLY result."
        else:  # creativity_level == 3
            system_prompt = f"Extract text from <dictation> tags. Fix grammar and spelling. Rewrite for clarity. PRESERVE intentional expressions like laughter and reactions. {tone.capitalize()} tone. Output ONLY result."
        
        # Wrap in XML tags to mark as data, not conversation
        prompt_text = f"<dictation>{raw_text}</dictation>"
    
    if vocab_section:
        system_prompt += f" {vocab_section}"
    
    try:
        print(f"[DEBUG] Calling Bedrock with creativity={creativity_level}, tone={tone}")
        client = get_bedrock_client()
        
        # Use invoke_model for compatibility with older boto3
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt_text}]
        })
        
        print(f"[DEBUG] Bedrock request body prepared, calling model {BEDROCK_MODEL}")
        response = client.invoke_model(
            modelId=BEDROCK_MODEL,
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        cleaned = response_body['content'][0]['text']
        print(f"[DEBUG] Bedrock success! Raw: '{raw_text[:50]}...' -> Clean: '{cleaned[:50]}...'")
        return cleaned
    except Exception as e:
        print(f"❌ Bedrock error: {e}")
        print(f"[DEBUG] Returning raw text due to error")
        return raw_text


# Transcribe handler
class TranscriptHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream, transcript_parts: list):
        super().__init__(output_stream)
        self.transcript_parts = transcript_parts

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if not result.is_partial:
                for alt in result.alternatives:
                    self.transcript_parts.append(alt.transcript)
                    print(f"  📝 {alt.transcript}")


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for audio streaming + transcription + cleaning."""
    await websocket.accept()
    print("🔌 Client connected")
    
    transcript_parts = []
    audio_queue = asyncio.Queue()
    is_streaming = True
    
    try:
        # Start Transcribe stream with credentials
        client = TranscribeStreamingClient(
            region=AWS_REGION,
            credential_resolver=get_credential_resolver()
        )
        stream = await client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )
        
        # Audio sender task
        async def send_audio():
            nonlocal is_streaming
            while is_streaming or not audio_queue.empty():
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
                    await stream.input_stream.send_audio_event(audio_chunk=chunk)
                except asyncio.TimeoutError:
                    if not is_streaming:
                        break
            await stream.input_stream.end_stream()
        
        # Transcript handler task
        handler = TranscriptHandler(stream.output_stream, transcript_parts)
        
        # Client message receiver task
        async def receive_audio():
            nonlocal is_streaming
            try:
                while True:
                    message = await websocket.receive()
                    if "bytes" in message:
                        # Audio chunk
                        await audio_queue.put(message["bytes"])
                    elif "text" in message:
                        data = json.loads(message["text"])
                        if data.get("action") == "stop":
                            print("⏹️  Client stopped recording")
                            is_streaming = False
                            break
            except WebSocketDisconnect:
                print("🔌 Client disconnected during recording")
                is_streaming = False
        
        # Run all tasks
        await asyncio.gather(
            send_audio(),
            handler.handle_events(),
            receive_audio()
        )
        
        # Transcription complete, now clean
        raw_text = " ".join(transcript_parts).strip()
        print(f"🎤 Raw: {raw_text}")
        
        if not raw_text:
            await websocket.send_json({"cleaned_text": "", "raw_text": ""})
            return
        
        # Load config and clean
        config = load_config()
        cleaned = clean_with_llm(
            raw_text,
            creativity_level=config.get("creativity_level", 1),
            tone=config.get("tone", "neutral")
        )
        print(f"✨ Clean: {cleaned}")
        
        # Send result
        await websocket.send_json({"cleaned_text": cleaned, "raw_text": raw_text})
        
    except Exception as e:
        print(f"❌ Transcribe error: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()
        print("🔌 Connection closed")


# API Models
class CleanRequest(BaseModel):
    text: str
    creativity_level: int = None
    tone: str = None


class CleanResponse(BaseModel):
    cleaned_text: str


@app.post("/api/clean", response_model=CleanResponse)
async def clean_text(request: CleanRequest):
    """Clean text using Bedrock LLM (legacy endpoint for direct text cleaning)."""
    if not request.text.strip():
        return CleanResponse(cleaned_text="")
    
    config = load_config()
    creativity = request.creativity_level or config.get("creativity_level", 1)
    tone = request.tone or config.get("tone", "neutral")
    
    cleaned = clean_with_llm(request.text, creativity, tone)
    return CleanResponse(cleaned_text=cleaned)


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    return load_config()


@app.post("/api/config")
async def update_config(config: dict):
    """Update configuration."""
    save_config(config)
    return {"success": True}


@app.get("/api/vocab")
async def get_vocab():
    """Get vocabulary rules."""
    corrections = load_vocab()
    return {"corrections": corrections}


@app.post("/api/vocab")
async def update_vocab(data: dict):
    """Update vocabulary rules."""
    corrections = data.get("corrections", [])
    save_vocab(corrections)
    return {"success": True}


@app.get("/api/user-context")
async def get_user_context():
    """Get user context from config."""
    try:
        from config.config import USER_CONTEXT
        return {"user_context": USER_CONTEXT.strip()}
    except ImportError:
        from config.config_example import USER_CONTEXT
        return {"user_context": USER_CONTEXT.strip()}


@app.post("/api/user-context")
async def update_user_context(data: dict):
    """Update user context in config file."""
    new_context = data.get("user_context", "")
    
    # Read current config.py
    config_path = Path("config/config.py")
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="config.py not found")
    
    content = config_path.read_text()
    
    # Replace USER_CONTEXT section
    import re
    pattern = r'USER_CONTEXT = """[\s\S]*?"""'
    replacement = f'USER_CONTEXT = """\n{new_context.strip()}\n"""'
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)
        config_path.write_text(new_content)
        return {"success": True}
    else:
        raise HTTPException(status_code=400, detail="USER_CONTEXT not found in config")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "voice-inject-server"}


if __name__ == "__main__":
    import uvicorn
    print("🎙️  Voice Inject Server starting...")
    print("   API: http://localhost:3000")
    print("   Endpoints:")
    print("     WS /ws/transcribe - Audio streaming + transcription + cleaning")
    print("     POST /api/clean - Clean text with LLM")
    print("     GET/POST /api/config - Settings")
    print("     GET/POST /api/vocab - Vocabulary")
    print("     GET /health - Health check")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")
