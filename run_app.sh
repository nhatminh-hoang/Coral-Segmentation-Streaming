#!/usr/bin/env bash
source ~/miniconda3/bin/activate
conda activate coral_streaming

echo "Starting CoralScapes Production Monitor..."
echo "Server will be available at http://0.0.0.0:7860"
python server.py