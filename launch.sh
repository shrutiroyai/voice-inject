#!/bin/bash
# Launch Voice Inject with UI
# This script starts the backend server, UI dev server, and client

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping server (PID: $SERVER_PID)"
        kill $SERVER_PID 2>/dev/null || true
    fi
    if [ ! -z "$UI_PID" ]; then
        echo "Stopping UI dev server (PID: $UI_PID)"
        kill $UI_PID 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo -e "${BLUE}=== Voice Inject Launcher ===${NC}\n"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found. Run ./install.sh first.${NC}"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if UI dependencies are installed
if [ ! -d "ui/node_modules" ]; then
    echo -e "${YELLOW}Installing UI dependencies...${NC}"
    cd ui
    npm install
    cd ..
fi

# Start the backend server
echo -e "${GREEN}Starting backend server...${NC}"
python server.py > /tmp/voice-inject-server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to start..."
for i in {1..10}; do
    if curl -s http://localhost:3000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is ready${NC}"
        break
    fi
    if [ $i -eq 10 ]; then
        echo -e "${RED}✗ Server failed to start. Check /tmp/voice-inject-server.log${NC}"
        exit 1
    fi
    sleep 1
done

# Start the UI dev server
echo -e "${GREEN}Starting UI dev server...${NC}"
cd ui
npm run dev > /tmp/voice-inject-ui.log 2>&1 &
UI_PID=$!
cd ..
echo "UI dev server PID: $UI_PID"

# Wait a moment for UI to start
sleep 2

echo -e "${GREEN}✓ UI dev server started${NC}"
echo -e "${BLUE}UI available at: http://localhost:5173${NC}\n"

# Start the client (this runs in foreground)
echo -e "${GREEN}Starting Voice Inject client...${NC}"
echo -e "${YELLOW}Hold Control to dictate, release to paste. Press Esc to quit.${NC}\n"
python client.py

# Cleanup will be called automatically on exit
