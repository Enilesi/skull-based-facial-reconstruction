#!/usr/bin/env bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p assets/flame assets data/landmarks