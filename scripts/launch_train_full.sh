#!/bin/bash
# Launch T2.16 real training in background.
# Output to logs/train_full.log
set -e
cd /Users/xuan/Work/dxhome/llm-compliance
mkdir -p logs

# Run the training. Use PYTHONUNBUFFERED for live output, MPS_HIGH_WATERMARK
# to allow full memory, and -u for unbuffered stdout.
PYTHONUNBUFFERED=1 \
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
python -u scripts/train.py \
    --config configs/full.yaml \
    --preload-dataset \
    --max-train-seconds 9000 \
    --save-every 20 \
    --partial-name lora_partial.safetensors \
    --checkpoint-name lora_full.safetensors \
    > logs/train_full.log 2>&1 &
TRAIN_PID=$!
echo "[launch] training PID: $TRAIN_PID"
echo "[launch] log: $(pwd)/logs/train_full.log"
echo $TRAIN_PID > logs/train_full.pid
disown
echo "[launch] detached"
