#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
chmod +x "./run.command"

# ========= HELPERS =========

download() {
    curl -L "$1" -o "$2"
}

open_editor() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open -W -e "$1"
    else
        xdg-open "$1"
    fi
}

install_python311() {
    PY_PKG_URL="https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg"
    PKG_FILE="python311.pkg"

    echo "Downloading Python 3.11 pkg installer..."
    curl -L "$PY_PKG_URL" -o "$PKG_FILE"

    echo "Installing Python 3.11 (silent)..."
    sudo installer -pkg "$PKG_FILE" -target /

    rm -f "$PKG_FILE"

    # Make sure the new python3 is in PATH immediately
    if [[ -d "/Library/Frameworks/Python.framework/Versions/3.11/bin" ]]; then
        export PATH="/Library/Frameworks/Python.framework/Versions/3.11/bin:$PATH"
    fi

    echo "Python 3.11 installed."
}

install_python310() {
    PY_PKG_URL="https://www.python.org/ftp/python/3.10.11/python-3.10.11-macos11.pkg"
    PKG_FILE="python310.pkg"

    echo "Downloading Python 3.10 pkg installer..."
    curl -L "$PY_PKG_URL" -o "$PKG_FILE"

    echo "Installing Python 3.10 (silent)..."
    sudo installer -pkg "$PKG_FILE" -target /

    rm -f "$PKG_FILE"

    # Make sure the new python3 is in PATH immediately
    if [[ -d "/Library/Frameworks/Python.framework/Versions/3.10/bin" ]]; then
        export PATH="/Library/Frameworks/Python.framework/Versions/3.10/bin:$PATH"
    fi

    echo "Python 3.10 installed."
}


# ========= BASIC SYSTEM REQUIREMENTS =========

# Xcode CLT
if ! xcode-select -p >/dev/null 2>&1; then
    echo "Installing Xcode Command Line Tools..."
    xcode-select --install || true
fi

# Homebrew (ARM only)
if ! command -v brew >/dev/null 2>&1; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    if [[ -d "/opt/homebrew/bin" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

# Ensure Brew is on PATH for the session
if [[ -d "/opt/homebrew/bin" ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# curl
if ! command -v curl >/dev/null 2>&1; then
    echo "Installing curl..."
    brew install curl
fi

# unzip
if ! command -v unzip >/dev/null 2>&1; then
    echo "Installing unzip..."
    brew install unzip
fi

# git
if ! command -v git >/dev/null 2>&1; then
    echo "Installing git..."
    brew install git
fi

# portaudio
if ! brew list portaudio >/dev/null 2>&1; then
    echo "Installing portaudio..."
    brew install portaudio
fi


# ========= SETTINGS =========

REPO_URL="https://github.com/RVirmoors/llm-actor"
REPO_ZIP="https://github.com/RVirmoors/llm-actor/archive/refs/heads/main.zip"
TARGET_DIR="llm-actor"
ASSET_DIR="$TARGET_DIR/assets"

KOKORO_ONNX_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_BIN_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


# ========= PYTHON VERSION CHECK =========


# ===== PYTHON REQUIREMENT: only 3.10 or 3.11 allowed =====

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")

echo "Detected Python version: $PY_VER"

if [[ "$PY_VER" != "3.10" && "$PY_VER" != "3.11" ]]; then
    echo "Python 3.10 or 3.11 not found — installing 3.11..."
    install_python311
fi

PYTHON_PATH=$(command -v python3 || true)
echo "Using Python at: $PYTHON_PATH"


# ========= PYTHON CERTIFICATES =========

PYTHON_REAL=$(python3 -c "import os,sys; print(os.path.realpath(sys.executable))")
FRAMEWORK_ROOT=$(dirname "$(dirname "$PYTHON_REAL")")
CERT_SCRIPT="$FRAMEWORK_ROOT/Install Certificates.command"

if [[ -f "$CERT_SCRIPT" ]]; then
    echo "Running Python certificates installation..."
    bash "$CERT_SCRIPT"
fi


# ========= FETCH PROJECT =========

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Project directory not found. Downloading project..."
    if command -v git >/dev/null 2>&1; then
        git clone "$REPO_URL" "$TARGET_DIR"
    else
        download "$REPO_ZIP" project.zip
        unzip project.zip
        rm project.zip
        mv "${TARGET_DIR}-main" "$TARGET_DIR"
    fi
fi

echo ""
echo "Repository ready in $TARGET_DIR."
echo ""


# ========= KOKORO MODELS =========

mkdir -p "$ASSET_DIR"

if [[ ! -f "$ASSET_DIR/kokoro-v1.0.onnx" ]]; then
    echo "This project uses Kokoro for local text-to-speech."
    echo "You can choose between Kokoro (default) or Deepgram TTS in settings.ini later."
    echo ""
    echo "Downloading Kokoro model files..."

    download "$KOKORO_ONNX_URL" "$ASSET_DIR/kokoro-v1.0.onnx"
    download "$KOKORO_BIN_URL" "$ASSET_DIR/voices-v1.0.bin"
fi


# ========= PYTHON ENV =========

echo ""
echo "Setting up the project Python environment..."
cd "$TARGET_DIR"

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install .


# ========= ENV FILE =========

if [[ ! -f ".env" ]]; then
    cp .env.example .env
fi

NEEDS_API_KEYS=false
if grep -q "DEEPGRAM_API_KEY=your-deepgram-api-key" .env; then
    NEEDS_API_KEYS=true
fi

if $NEEDS_API_KEYS; then
    echo ""
    echo "API keys[s] not yet configured."
    echo ""
    echo "Switch to your web browser to see two new tabs."
    echo "Please register or log in, create API keys and copy them,"
    echo "then switch to your text editor where the .env file is open,"
    echo "and paste them into the relevant *_API_KEY fields."
    echo ""
    echo "When you're done, save and close the text editor (Cmd+Q)."
    echo ""
    open "https://console.deepgram.com/"
    open "https://console.groq.com/"
    open_editor ".env"
fi


# ========= SOUNDDEVICE + SETTINGS =========

echo ""
echo "Note the input and output device indices" 
echo "from the list below, and edit them into"
echo "BASIC_PROJECT\settings.ini as needed."
echo ""
echo "When you're done, save and close the text editor (Cmd+Q)."
echo ""

python3 -m sounddevice || true
open_editor "./BASIC_PROJECT/settings.ini"


echo ""
echo "Setup complete. You can now run the project by executing:"
echo ""
echo "    run.command"
echo ""
echo "When MacOS asks to enable Accessibility for the terminal,"
echo "please allow it in System Preferences >  Privacy & Security > Accessibility."
echo ""
echo "To change any settings, run this setup again"
echo "or edit the .env and settings.ini files directly."
echo ""
