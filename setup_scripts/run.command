#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/llm-actor"

source venv/bin/activate
sudo python3 BASIC_PROJECT/boot.py
