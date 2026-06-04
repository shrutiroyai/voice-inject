#!/usr/bin/env python3
"""
Configuration template for Voice Inject.

Copy this file to config.py and customize the values.
DO NOT commit config.py to version control.
"""

# AWS Configuration
AWS_REGION = "us-west-2"  # AWS region for Transcribe and Bedrock
BEDROCK_MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"  # Bedrock model ID

# Audio Configuration
SAMPLE_RATE = 16000  # Audio sample rate in Hz (16kHz is standard for speech)
CHANNELS = 1  # Mono audio
LANGUAGE_CODE = "en-US"  # Language for transcription

# Command Mode Configuration
COMMAND_WAKE_WORD = "molly"  # Say this word to trigger command mode

# Keyboard Configuration
TRIGGER_KEY = "ctrl"  # Options: "ctrl", "shift", "alt", "cmd"

# User Context (for better LLM understanding)
USER_CONTEXT = """
Data scientist and software developer working with AWS, Python, and AI/ML.
Frequently uses: GitHub, GenAI, LLM, Bedrock, SageMaker, API, Docker, 
Kubernetes, Lambda, RAG, embeddings, inference, CI/CD, PR, repo, commit, 
merge, deploy, pipeline, model, training, NLP, Transcribe.
"""
