#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .

if [ ! -f .env ]; then
    cp .env.example .env
fi

# print sounddevice info
echo "==== Sound device info: ===="
python -m sounddevice

# open .env in default editor
if [[ "$OSTYPE" == "darwin"* ]]; then
    open -e .env
    open -e ./BASIC_PROJECT/settings.ini
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open .env
    xdg-open ./BASIC_PROJECT/settings.ini
fi