#!/bin/bash
#
# Voice Inject - One-Command Installer & Launcher
# Usage: ./install.sh or "voice" (after alias is registered)
#

# === COLOR CONSTANTS ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# === PATH RESOLUTION ===
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# === CONFIGURATION ===
HEALTH_TIMEOUT=10
CLIENT_CHECK_DELAY=2
KILL_TIMEOUT=5
SHUTDOWN_TIMEOUT=10

# === PID TRACKING ===
SERVER_PID=""
CLIENT_PID=""

# === FUNCTIONS ===

cleanup() {
    echo -e "${YELLOW}Shutting down...${NC}"

    # Send SIGTERM to tracked PIDs (only if non-empty and process exists)
    for pid in $CLIENT_PID $SERVER_PID; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null
        fi
    done

    # Wait up to SHUTDOWN_TIMEOUT seconds for processes to exit
    local elapsed=0
    while [ $elapsed -lt $SHUTDOWN_TIMEOUT ]; do
        local all_exited=true
        for pid in $CLIENT_PID $SERVER_PID; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                all_exited=false
                break
            fi
        done
        if $all_exited; then
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    # Send SIGKILL to any remaining processes
    for pid in $CLIENT_PID $SERVER_PID; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    done

    # Catch orphaned subprocesses
    pkill -P $$ 2>/dev/null

    echo -e "${GREEN}All services stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

check_prerequisites() {
    local missing=()

    if ! command -v python3 &>/dev/null; then
        missing+=("  - python3 (Install: brew install python3)")
    fi

    if ! command -v ffmpeg &>/dev/null; then
        missing+=("  - ffmpeg (Install: brew install ffmpeg)")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}❌ Missing required tools:${NC}"
        for tool in "${missing[@]}"; do
            echo -e "${RED}${tool}${NC}"
        done
        echo ""
        echo "Please install the missing tools and try again."
        exit 1
    fi

    echo -e "${GREEN}✓ All prerequisites found${NC}"
}

install_python_deps() {
    if [ ! -d "$SCRIPT_DIR/.venv" ]; then
        echo -e "${BLUE}Creating Python virtual environment...${NC}"
        python3 -m venv "$SCRIPT_DIR/.venv"
    fi

    # Activate venv
    source "$SCRIPT_DIR/.venv/bin/activate"

    # Check if key modules can be imported
    if "$SCRIPT_DIR/.venv/bin/python" -c "import fastapi, yaml, pynput, sounddevice, faster_whisper" 2>/dev/null; then
        echo -e "${GREEN}✓ Python dependencies already satisfied${NC}"
    else
        echo -e "${BLUE}Installing Python dependencies...${NC}"
        if ! pip install -r "$SCRIPT_DIR/requirements.txt" --quiet; then
            echo -e "${RED}❌ Failed to install Python dependencies${NC}"
            exit 2
        fi
        echo -e "${GREEN}✓ Python dependencies installed${NC}"
    fi
}

bootstrap_config() {
    # Create ~/.voice-inject directory if absent
    if [ ! -d "$HOME/.voice-inject" ]; then
        mkdir -p "$HOME/.voice-inject"
        chmod 755 "$HOME/.voice-inject"
        echo -e "${GREEN}✓ Created ~/.voice-inject directory${NC}"
    fi

    echo -e "${GREEN}✓ Configuration ready${NC}"
}

kill_port() {
    local port=$1
    local pid

    # Check if anything is on that port
    pid=$(lsof -ti :"$port" 2>/dev/null)
    if [ -z "$pid" ]; then
        return 0
    fi

    # Kill the process(es) occupying the port
    lsof -ti :"$port" | xargs kill -9 2>/dev/null

    # Wait up to KILL_TIMEOUT seconds for port to become free
    local elapsed=0
    while [ $elapsed -lt $KILL_TIMEOUT ]; do
        sleep 1
        elapsed=$((elapsed + 1))
        if ! lsof -ti :"$port" &>/dev/null; then
            return 0
        fi
    done

    # Port still occupied after timeout
    echo -e "${RED}❌ Port $port remains occupied after ${KILL_TIMEOUT}s — cannot start services${NC}"
    exit 4
}

open_browser() {
    local url="http://localhost:3000"

    if command -v open &>/dev/null; then
        open "$url"
    elif command -v xdg-open &>/dev/null; then
        xdg-open "$url"
    else
        echo -e "${BLUE}Open ${url} in your browser${NC}"
    fi
}

register_alias() {
    local SHELL_CONFIG=""
    local ALIAS_LINE="alias voice=\"$SCRIPT_DIR/install.sh\""

    # Determine shell config file based on $SHELL
    if echo "$SHELL" | grep -q "zsh"; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif echo "$SHELL" | grep -q "bash"; then
        SHELL_CONFIG="$HOME/.bashrc"
    fi

    if [ -n "$SHELL_CONFIG" ]; then
        # Remove any old voice() function definition that would shadow the alias
        if grep -q "^voice()" "$SHELL_CONFIG" 2>/dev/null; then
            # Remove the old voice function block (from "voice()" to its closing "}")
            sed -i '' '/^# Voice Inject/,/^}/d' "$SHELL_CONFIG" 2>/dev/null
            sed -i '' '/^voice()/,/^}/d' "$SHELL_CONFIG" 2>/dev/null
            # Also remove voice-stop if present
            sed -i '' '/^# Stop voice/,/^}/d' "$SHELL_CONFIG" 2>/dev/null
            sed -i '' '/^voice-stop()/,/^}/d' "$SHELL_CONFIG" 2>/dev/null
            echo -e "${YELLOW}Removed old voice() function from $SHELL_CONFIG${NC}"
        fi

        # Check if alias already exists
        if grep -qF "$ALIAS_LINE" "$SHELL_CONFIG" 2>/dev/null; then
            echo -e "${GREEN}✓ voice command already registered${NC}"
        else
            # Attempt to append alias
            if echo "" >> "$SHELL_CONFIG" && echo "# Voice Inject — launch with 'voice'" >> "$SHELL_CONFIG" && echo "$ALIAS_LINE" >> "$SHELL_CONFIG" 2>/dev/null; then
                echo -e "${GREEN}✓ Registered 'voice' command in $SHELL_CONFIG${NC}"
            else
                echo -e "${YELLOW}⚠️  Could not write to $SHELL_CONFIG (permission denied)${NC}"
                echo -e "${YELLOW}   Add this line manually to your shell config:${NC}"
                echo -e "${YELLOW}   $ALIAS_LINE${NC}"
                return 0
            fi
        fi

        # Determine which source command to suggest
        local RC_NAME
        RC_NAME=$(basename "$SHELL_CONFIG")
        echo -e "${BLUE}Run 'source ~/$RC_NAME' or restart terminal to use 'voice' command${NC}"
    else
        # Non-bash/non-zsh shell: try symlink fallback
        if ln -sf "$SCRIPT_DIR/install.sh" /usr/local/bin/voice 2>/dev/null; then
            echo -e "${GREEN}✓ Created symlink /usr/local/bin/voice${NC}"
        else
            echo -e "${YELLOW}⚠️  Could not create symlink in /usr/local/bin (permission denied)${NC}"
            echo -e "${YELLOW}   Add this alias manually to your shell config:${NC}"
            echo -e "${YELLOW}   $ALIAS_LINE${NC}"
        fi
    fi
}

wait_for_health() {
    local url=$1
    local timeout=$2
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        if curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null | grep -q "200"; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    return 1
}

wait_for_port() {
    local port=$1
    local timeout=$2
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        if nc -z localhost "$port" 2>/dev/null; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    return 1
}

start_services() {
    # Free port before starting
    kill_port 3000

    # Start server
    echo -e "${BLUE}Starting server...${NC}"
    "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/server.py" > /tmp/voice-inject-server.log 2>&1 &
    SERVER_PID=$!

    # Wait for server health check
    if ! wait_for_health "http://localhost:3000/health" "$HEALTH_TIMEOUT"; then
        echo -e "${RED}❌ Server failed to start within ${HEALTH_TIMEOUT}s${NC}"
        echo -e "${RED}   Check logs: /tmp/voice-inject-server.log${NC}"
        kill "$SERVER_PID" 2>/dev/null
        exit 4
    fi
    echo -e "${GREEN}✓ Server is healthy${NC}"

    # Start client
    echo -e "${BLUE}Starting client...${NC}"
    "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/client.py" > /tmp/voice-inject-client.log 2>&1 &
    CLIENT_PID=$!

    # Check if client is still alive after delay (non-fatal)
    sleep "$CLIENT_CHECK_DELAY"
    if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Client process exited early — check logs: /tmp/voice-inject-client.log${NC}"
    else
        echo -e "${GREEN}✓ Client is running${NC}"
    fi
}

# === MAIN ===

echo ""
echo -e "${BLUE}🎙️  Voice Inject - One-Command Installer${NC}"
echo "=========================================="
echo ""

# Phase 1: Prerequisite validation
check_prerequisites

echo ""

# Phase 2: Dependency installation
install_python_deps

echo ""

# Phase 3: Configuration bootstrap
bootstrap_config

echo ""

# Phase 4: Service startup
echo -e "${BLUE}Starting services...${NC}"
start_services

# Browser launch
open_browser

echo ""

# Phase 5: Alias registration
register_alias

echo ""

# Final status summary
echo -e "${GREEN}=========================================="
echo -e "✅ Voice Inject is running!"
echo -e "   UI:      http://localhost:3000"
echo -e "   Press Ctrl+C to stop all services"
echo -e "==========================================${NC}"

# Keep script running until interrupted
wait
