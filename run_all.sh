#!/usr/bin/env bash
# Full pipeline. Override epochs for a quick run:  EPOCHS=5 CLS_EPOCHS=5 ./run_all.sh
set -e
cd "$(dirname "$0")"
EPOCHS=${EPOCHS:-30}
CLS_EPOCHS=${CLS_EPOCHS:-20}
BASE=${BASE:-32}

python src/download_data.py
python src/prepare_data.py

for P in raw zscore clahe; do
  python src/train_seg.py --preproc $P --epochs $EPOCHS --base $BASE
done
python src/train_cls.py --preproc clahe --epochs $CLS_EPOCHS

python src/evaluate.py runs/seg_raw runs/seg_zscore runs/seg_clahe runs/cls_clahe
