#!/usr/bin/env bash
set -euo pipefail

# Example: numeric-prior warm-up on ETTh1 with 10% training windows and H=96.
PYTHON_BIN="${PYTHON:-python}"
DATASET="ETTh1"
SEQ_LEN=96
PRED_LEN=96
TRAIN_RATIO=0.10
TEXT_CSV="generated/${DATASET}/${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}_text_M_semantic_v1.csv"
TEXT_NPZ="generated/${DATASET}/${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}_semantic_v1_text_features.npz"

${PYTHON_BIN} run.py \
  --is_training 1 \
  --model_id "${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}_r10_numeric" \
  --model CARE_Forecast \
  --method_profile numeric_only \
  --data "${DATASET}" \
  --root_path . \
  --data_path "dataset/${DATASET}.csv" \
  --text_path "${TEXT_CSV}" \
  --text_feature_path "${TEXT_NPZ}" \
  --features M \
  --target OT \
  --freq h \
  --split_mode ett_standard \
  --seq_len "${SEQ_LEN}" \
  --label_len 48 \
  --pred_len "${PRED_LEN}" \
  --text_dim 256 \
  --hidden_dim 256 \
  --sem_dim 128 \
  --moving_avg_kernel 7 \
  --frft_init_alpha 0.4 \
  --train_ratio "${TRAIN_RATIO}" \
  --train_ratio_seed 2026 \
  --train_epochs 10 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --learning_rate 5e-4 \
  --weight_decay 1e-4 \
  --patience 5 \
  --des numeric_warmup \
  --checkpoints checkpoints \
  --output_dir outputs
