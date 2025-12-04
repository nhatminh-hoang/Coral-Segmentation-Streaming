#!/usr/bin/env bash
source ~/miniconda3/bin/activate
conda activate coral_streaming

uvicorn app2:app --host 0.0.0.0 --port 7860
