#!/usr/bin/env bash
source ~/miniconda3/bin/activate
conda activate coral_streaming

echo "=============================================="
echo "🪸 CoralScapes Production Monitor"
echo "=============================================="

# Create chunks directory if it doesn't exist
mkdir -p chunks

# Start the segmenter worker in the background
echo "🔨 Starting Segmenter Worker (background process)..."
python segmenter_worker.py &
SEGMENTER_PID=$!
echo "   Segmenter PID: $SEGMENTER_PID"

# # Give segmenter a moment to start
# sleep 2

# Start the web server
echo "🌐 Starting Web Server at http://0.0.0.0:7860"
python server.py

# Cleanup: kill segmenter when server stops
echo "🛑 Stopping Segmenter Worker..."
kill $SEGMENTER_PID 2>/dev/null
echo "✅ All processes stopped."