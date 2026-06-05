#!/usr/bin/env python3
"""
Example configuration for Voice Inject.

Copy this file to config.py and customize with your own terminology.
"""

# User Context - Add your frequently used terms, acronyms, and domain vocabulary
# This helps Bedrock LLM better understand and format your dictation
USER_CONTEXT = """
Add your professional context here. Include:
- Your role/industry (e.g., software engineer, researcher, writer)
- Frequently used technical terms and acronyms
- Project-specific terminology
- Names of tools, frameworks, or products you mention often

Example:
Software engineer working on cloud infrastructure.
Frequently uses: AWS, Kubernetes, Docker, Terraform, Python, React, 
API, microservices, CI/CD, S3, Lambda, CloudWatch, monitoring, deployment.
"""
