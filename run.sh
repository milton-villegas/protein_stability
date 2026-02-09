#!/bin/bash
# ============================================
# Protein Stability DoE Suite Launcher
# Run this script to start the application
# ============================================

# Change to the script's directory
cd "$(dirname "$0")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  Protein Stability DoE Suite"
echo "=========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    echo "Please install Python 3 using your package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "  Fedora: sudo dnf install python3 python3-pip"
    echo "  Arch: sudo pacman -S python python-pip"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo -e "${GREEN}Found:${NC} $PYTHON_VERSION"

# Check for tkinter (often missing on Linux)
if ! python3 -c "import tkinter" &> /dev/null; then
    echo -e "${RED}Error: tkinter is not installed.${NC}"
    echo "Please install it using your package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3-tk"
    echo "  Fedora: sudo dnf install python3-tkinter"
    echo "  Arch: sudo pacman -S tk"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Virtual environment directory
VENV_DIR=".venv"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo -e "${YELLOW}First run detected. Setting up environment...${NC}"
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to create virtual environment.${NC}"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo -e "${GREEN}Virtual environment created.${NC}"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Check if dependencies are installed by looking for a marker file
DEPS_MARKER="$VENV_DIR/.deps_installed"

if [ ! -f "$DEPS_MARKER" ]; then
    echo ""
    echo "Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip -q
    pip install -r requirements.txt

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to install dependencies.${NC}"
        read -p "Press Enter to exit..."
        exit 1
    fi

    # Create marker file
    touch "$DEPS_MARKER"
    echo -e "${GREEN}Dependencies installed successfully.${NC}"
fi

# Run the application
echo ""
echo -e "${GREEN}Launching application...${NC}"
echo ""
python3 main.py

# Deactivate virtual environment when done
deactivate
