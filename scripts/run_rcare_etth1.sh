#!/usr/bin/env bash
set -euo pipefail

# Example: RCARE-Forecast teacher-student training on ETTh1 with 10% training windows and H=96.
PYTHON_BIN="${PYTHON:-python}"
DATASET="ETTh1"
SEQ_LEN=96
PRED_LEN=96
TRAIN_RATIO=0.10
TEXT_CSV="generated/${DATASET}/${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}_text_M_semantic_v1.csv"
TEXT_NPZ="generated/${DATASET}/${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}_semantic_v1_text_features.npz"
NUMERIC_CKPT="${NUMERIC_CKPT:-}"

EXTRA_ARGS=()
if [[ -n "${NUMERIC_CKPT}" ]]; then
  EXTRA_ARGS+=(--pretrained_numeric_checkpoint "${NUMERIC_CKPT}" --freeze_numeric_backbone 1)
fi

${PYTHON_BIN} run.py \
  --is_training 1 \
  --model_id "${DATASET}_sl${SEQ_LEN}_pl${PRED_LEN}_r10_rcare" \
  --model CARE_Forecast \
  --method_profile privileged_bridge \
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
  --student_text_cols llm_history_text llm_history_prior_text \
  --teacher_text_cols llm_future_text llm_residual_text \
  --hidden_dim 256 \
  --sem_dim 128 \
  --moving_avg_kernel 7 \
  --frft_init_alpha 0.4 \
  --residual_planner_type cross_attn \
  --train_ratio "${TRAIN_RATIO}" \
  --train_ratio_seed 2026 \
  --train_epochs 12 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --learning_rate 5e-4 \
  --numeric_learning_rate 5e-5 \
  --weight_decay 1e-4 \
  --patience 5 \
  --des rcare \
  --checkpoints checkpoints \
  --output_dir outputs \
  "${EXTRA_ARGS[@]}"
