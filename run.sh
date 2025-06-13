#!/bin/bash

set -e

SIZE=5000
EPOCHS=1
BATCH_SIZE=8

echo "=== Training with HALF dataset (${SIZE}/2 = $((SIZE/2)) images) ==="
python detector/train.py --use_half --size $SIZE --epochs $EPOCHS --batch_size $BATCH_SIZE --save_path detector_half.pth

echo "=== Training with FULL dataset (${SIZE} images) ==="
python detector/train.py --size $SIZE --epochs $EPOCHS --batch_size $BATCH_SIZE --save_path detector_full.pth
