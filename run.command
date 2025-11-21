#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

source .venv/bin/activate
sudo python3 BASIC_PROJECT/boot.py
