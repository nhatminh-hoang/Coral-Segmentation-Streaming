#!/usr/bin/env bash
source ~/miniconda3/bin/activate
conda activate coral_streaming

echo "=============================================="
echo "🪸 CoralScapes Production Monitor"
echo "=============================================="

mkdir -p chunks

# 1. Define cleanup function
cleanup() {
    echo "🛑 Stopping Segmenter Worker..."
    kill $SEGMENTER_PID 2>/dev/null
    echo "✅ All processes stopped."
}

# 2. Trap signals (INT=Ctrl+C, TERM=kill command, EXIT=script exit)
trap cleanup INT TERM EXIT

echo "🔨 Starting Segmenter Worker (background process)..."
python segmenter_worker.py &
SEGMENTER_PID=$!
echo "   Segmenter PID: $SEGMENTER_PID"

echo "🌐 Starting Web Server at http://0.0.0.0:7860"
python server.py

# No need for manual cleanup at the bottom; the 'trap' handles it automatically when server.py exits.