#!/bin/bash

python3 -m venv llm-eng

source llm-eng/bin/activate

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126