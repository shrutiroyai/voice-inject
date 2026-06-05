#!/bin/bash
set -e

echo "🎙️  Voice Inject - Unified Setup & Launcher"
echo "============================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Change to script directory (resolve to absolute path)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down all services...${NC}"
    [ ! -z "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null
    [ ! -z "$UI_PID" ] && kill $UI_PID 2>/dev/null
    [ ! -z "$CLIENT_PID" ] && kill $CLIENT_PID 2>/dev/null
    pkill -P $$ 2>/dev/null || true
    echo -e "${GREEN}✓ All services stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# ============================================
# SETUP PHASE - Auto-install if needed
# ============================================

echo -e "${YELLOW}Checking setup...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found!${NC}"
    echo "Install from https://www.python.org/"
    exit 1
fi

# Check/create config directory
CONFIG_DIR="$HOME/.voice-inject"
if [ ! -d "$CONFIG_DIR" ]; then
    echo -e "${YELLOW}Creating ~/.voice-inject directory...${NC}"
    mkdir -p "$CONFIG_DIR"
fi

# Check/create config files
if [ ! -f "config/config.py" ]; then
    echo -e "${YELLOW}Creating config/config.py from example...${NC}"
    cp config/config_example.py config/config.py
fi

# Check/create Python venv
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment (first time)...${NC}"
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install Python dependencies into venv
if ! python -c "import fastapi, boto3, yaml, pynput, sounddevice, whisper" 2>/dev/null; then
    echo -e "${YELLOW}Installing Python dependencies into venv (first time)...${NC}"
    pip install -q -r requirements.txt
fi

# Check ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}❌ ffmpeg not found!${NC}"
    echo "Install with: brew install ffmpeg"
    exit 1
fi

# Check Node.js for UI
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found!${NC}"
    echo "Install from https://nodejs.org/"
    exit 1
fi

# Check/install UI dependencies
if [ ! -d "ui/node_modules" ]; then
    echo -e "${YELLOW}Installing UI dependencies (first time)...${NC}"
    cd ui && npm install --silent && cd ..
fi

echo -e "${GREEN}✓ Setup complete${NC}"
echo ""

# ============================================
# STARTUP PHASE
# ============================================

# Step 1: Clean up any existing processes
echo -e "${YELLOW}1. Cleaning up existing processes...${NC}"
lsof -ti :3000 | xargs kill -9 2>/dev/null || true
lsof -ti :5173 | xargs kill -9 2>/dev/null || true
pkill -9 -f "python.*server\|python.*client" 2>/dev/null || true
sleep 1
echo -e "${GREEN}✓ Ports clear${NC}"

# Step 2: Start server (using venv python)
echo -e "${YELLOW}2. Starting server...${NC}"
python server.py > /tmp/voice-inject-server.log 2>&1 &
SERVER_PID=$!
sleep 3

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}❌ Server failed to start!${NC}"
    echo "Check logs: tail /tmp/voice-inject-server.log"
    exit 1
fi

# Test server is responding
if ! curl -s http://localhost:3000/health > /dev/null; then
    echo -e "${RED}❌ Server not responding!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Server running${NC} (http://localhost:3000)"

# Step 3: Start UI
echo -e "${YELLOW}3. Starting UI...${NC}"
cd ui && npm run dev > /tmp/voice-inject-ui.log 2>&1 &
UI_PID=$!
cd ..
sleep 3

if ! kill -0 $UI_PID 2>/dev/null; then
    echo -e "${RED}❌ UI failed to start!${NC}"
    echo "Check logs: tail /tmp/voice-inject-ui.log"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

echo -e "${GREEN}✓ UI running${NC} (http://localhost:5173)"

# Step 4: Start voice capture client (using venv python)
echo -e "${YELLOW}4. Starting voice capture client...${NC}"
cd "$SCRIPT_DIR"
python "$SCRIPT_DIR/client.py" > /tmp/voice-inject-client.log 2>&1 &
CLIENT_PID=$!
sleep 2

if ! kill -0 $CLIENT_PID 2>/dev/null; then
    echo -e "${RED}❌ Client failed to start!${NC}"
    echo "Check logs: tail /tmp/voice-inject-client.log"
else
    echo -e "${GREEN}✓ Client running${NC}"
fi

# Step 5: Open browser
echo -e "${YELLOW}5. Opening browser...${NC}"
sleep 2
open http://localhost:5173 2>/dev/null || echo "(Open http://localhost:5173 in your browser)"
echo -e "${GREEN}✓ Browser opened${NC}"

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 Voice Inject is ready!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Services:${NC}"
echo -e "  🖥️  Server:  ${GREEN}http://localhost:3000${NC}"
echo -e "  🎨 UI:      ${GREEN}http://localhost:5173${NC}"
echo -e "  🎤 Client:  ${GREEN}Listening for Fn+Fn${NC}"
echo ""
echo -e "${BLUE}Usage:${NC}"
echo -e "  1. Press Fn+Fn to start recording"
echo -e "  2. Speak your text"
echo -e "  3. Press Fn+Fn again to stop"
echo -e "  4. Cleaned text appears in your app"
echo ""
echo -e "${BLUE}Logs:${NC}"
echo -e "  Server: /tmp/voice-inject-server.log"
echo -e "  UI:     /tmp/voice-inject-ui.log"
echo -e "  Client: /tmp/voice-inject-client.log"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep script running
wait
