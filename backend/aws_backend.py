#!/usr/bin/env python3
"""AWS Backend for Voice Inject — handles Transcribe and Bedrock API calls."""

import asyncio
import boto3
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from amazon_transcribe.auth import StaticCredentialResolver

try:
    from config.config import AWS_REGION, BEDROCK_MODEL_ID, SAMPLE_RATE, LANGUAGE_CODE
except ImportError:
    from config.config_example import AWS_REGION, BEDROCK_MODEL_ID, SAMPLE_RATE, LANGUAGE_CODE


class TranscriptHandler(TranscriptResultStreamHandler):
    """Handler for AWS Transcribe streaming events."""
    
    def __init__(self, output_stream, transcript_parts: list):
        super().__init__(output_stream)
        self.transcript_parts = transcript_parts

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        """Process transcript events and collect final results."""
        for result in transcript_event.transcript.results:
            if not result.is_partial:
                for alt in result.alternatives:
                    self.transcript_parts.append(alt.transcript)
                    print(f"  📝 {alt.transcript}")


class AWSBackend:
    """AWS backend for speech-to-text and LLM text cleaning."""
    
    def __init__(self, region: str = None, model_id: str = None):
        """
        Initialize AWS backend.
        
        Args:
            region: AWS region for Transcribe and Bedrock (defaults to config.AWS_REGION)
            model_id: Bedrock model ID (defaults to config.BEDROCK_MODEL_ID)
        """
        self.region = region or AWS_REGION
        self.model_id = model_id or BEDROCK_MODEL_ID
    
    def _get_credential_resolver(self):
        """Get AWS credentials for Transcribe streaming."""
        session = boto3.Session(region_name=self.region)
        creds = session.get_credentials().get_frozen_credentials()
        return StaticCredentialResolver(
            access_key_id=creds.access_key,
            secret_access_key=creds.secret_key,
            session_token=creds.token or "",
        )
    
    def _get_bedrock_client(self):
        """Create a Bedrock runtime client."""
        return boto3.client("bedrock-runtime", region_name=self.region)
    
    async def transcribe_stream(self, audio_queue, is_recording_fn) -> list[str]:
        """
        Stream audio to AWS Transcribe and return transcript.
        
        Args:
            audio_queue: Queue containing audio chunks (bytes)
            is_recording_fn: Function that returns True while recording
            
        Returns:
            List of transcript text segments
        """
        transcript_parts = []
        
        client = TranscribeStreamingClient(
            region=self.region,
            credential_resolver=self._get_credential_resolver()
        )
        
        stream = await client.start_stream_transcription(
            language_code=LANGUAGE_CODE,
            media_sample_rate_hz=SAMPLE_RATE,
            media_encoding="pcm",
        )
        
        async def send_audio():
            """Send audio chunks to Transcribe."""
            while is_recording_fn() or not audio_queue.empty():
                try:
                    chunk = audio_queue.get(timeout=0.2)
                    await stream.input_stream.send_audio_event(audio_chunk=chunk)
                except:
                    if not is_recording_fn():
                        break
            await stream.input_stream.end_stream()
        
        handler = TranscriptHandler(stream.output_stream, transcript_parts)
        await asyncio.gather(send_audio(), handler.handle_events())
        
        return transcript_parts
    
    def clean_text(self, raw_text: str, system_prompt: str) -> str:
        """
        Clean text using Bedrock LLM with prompt caching.
        
        Args:
            raw_text: Raw transcript text to clean
            system_prompt: System prompt for the LLM (will be cached)
            
        Returns:
            Cleaned text
        """
        if not raw_text.strip():
            return ""
        
        client = self._get_bedrock_client()
        
        # Use prompt caching: system prompt gets cached for faster subsequent calls
        response = client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": raw_text}]}],
            system=[
                {"text": system_prompt},
                {"cache_control": {"type": "ephemeral"}}  # Cache everything above
            ],
        )
        
        return response["output"]["message"]["content"][0]["text"]
