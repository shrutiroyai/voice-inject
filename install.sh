#!/bin/bash
set -e

echo "🔧 Setting up Voice Inject..."

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Seed vocab if not exists
VOCAB_DIR="$HOME/.voice-inject"
mkdir -p "$VOCAB_DIR"
if [ ! -f "$VOCAB_DIR/vocab.yaml" ]; then
    cp default_vocab.yaml "$VOCAB_DIR/vocab.yaml"
    echo "📝 Seeded ~/.voice-inject/vocab.yaml with default terms"
fi

echo ""
echo "✅ Done! To run:"
echo "   source .venv/bin/activate"
echo "   python voice_inject.py"
echo ""
echo "📝 Manage vocabulary:"
echo "   python vocab.py list"
echo "   python vocab.py add 'GenBI' 'gen bi' 'jenbi'"
echo "   python vocab.py remove 'GenBI'"
echo ""
echo "⚠️  macOS: Grant Accessibility + Microphone permissions to Terminal/iTerm"
