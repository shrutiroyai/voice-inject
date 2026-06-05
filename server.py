#!/usr/bin/env python3
"""Voice Inject Server — runs on dev desktop, handles Transcribe + LLM cleaning."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml
import json
import sys
import logging
from pathlib import Path
import boto3

# Configure logging (force=True overrides uvicorn's logging config)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/voice-inject-server.log', mode='w')
    ],
    force=True
)
logger = logging.getLogger(__name__)

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
BEDROCK_MODEL = "openai.gpt-oss-120b-1:0"
AWS_REGION = "us-west-2"


def load_config():
    """Load configuration from config.yaml."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {
        "creativity_level": 0.3,  # Temperature 0-1
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
            logger.info(f"Loaded {len(corrections)} vocab corrections from {VOCAB_PATH}")
            return corrections
    logger.info(f"Vocab file not found at {VOCAB_PATH}")
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
        # Normal mode: single prompt, temperature controls creativity
        system_prompt = (
            f"Extract and correct the text inside <dictation> tags. "
            f"Fix grammar, punctuation, and spelling. "
            f"Remove ONLY hesitation filler words (um, uh, like, you know). "
            f"PRESERVE intentional expressions like: laughter (ha ha, haha), reactions (LOL, OMG), and other meaningful interjections. "
            f"{tone.capitalize()} tone. "
            f"Do NOT include reasoning, explanations, or any tags. "
            f"Output ONLY the corrected text, nothing else."
        )
        # Wrap in XML tags to mark as data, not conversation
        prompt_text = f"<dictation>{raw_text}</dictation>"
    
    if vocab_section:
        system_prompt += f" {vocab_section}"
    
    try:
        logger.info(f"Calling Bedrock with creativity={creativity_level}, tone={tone}")
        client = get_bedrock_client()
        
        # Use OpenAI-compatible format for GPT models
        # creativity_level is now temperature (0-1 range)
        temperature = float(creativity_level) if creativity_level <= 1 else 0.3
        body = json.dumps({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            "max_tokens": 1024,
            "temperature": temperature
        })
        
        logger.info(f"Bedrock request prepared, calling model {BEDROCK_MODEL}")
        response = client.invoke_model(
            modelId=BEDROCK_MODEL,
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        cleaned = response_body['choices'][0]['message']['content']
        
        # Strip out reasoning tags if present (GPT models sometimes include them)
        import re
        cleaned = re.sub(r'<reasoning>.*?</reasoning>\s*', '', cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()
        
        logger.info(f"Bedrock success! Raw: '{raw_text[:50]}...' -> Clean: '{cleaned[:50]}...'")
        return cleaned
    except Exception as e:
        logger.error(f"Bedrock error: {e}", exc_info=True)
        logger.warning("Returning raw text due to Bedrock error")
        return raw_text


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
    logger.info(f"Received clean request: text='{request.text[:50]}...'")
    
    if not request.text.strip():
        return CleanResponse(cleaned_text="")
    
    config = load_config()
    creativity = request.creativity_level or config.get("creativity_level", 1)
    tone = request.tone or config.get("tone", "neutral")
    
    logger.info(f"Calling clean_with_llm with creativity={creativity}, tone={tone}")
    
    cleaned = clean_with_llm(request.text, creativity, tone)
    
    logger.info(f"Got result: '{cleaned[:50]}...'")
    
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
    print("     POST /api/clean - Clean text with Bedrock LLM")
    print("     GET/POST /api/config - Settings")
    print("     GET/POST /api/vocab - Vocabulary")
    print("     GET /health - Health check")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")
