#!/usr/bin/env bash
set -euo pipefail

# Clean RCARE-Forecast launcher for Flight.
# Usage examples:
#   bash scripts/Flight.sh
#   RUN_STAGE=numeric bash scripts/Flight.sh
#   RUN_STAGE=full PRED_LENS="96 192" RATIOS="0.10" bash scripts/Flight.sh
#   NUMERIC_CKPT="checkpoints/<numeric-setting>/checkpoint.pth" bash scripts/Flight.sh

PYTHON_BIN="${PYTHON:-python}"
DATASET="Flight"
SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-48}"
TARGET="${TARGET:-OT}"
FEATURES="${FEATURES:-M}"
SPLIT_MODE="${SPLIT_MODE:-ratio}"
FREQ="${FREQ:-h}"
TEXT_DIM="${TEXT_DIM:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
SEM_DIM="${SEM_DIM:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
NUMERIC_EPOCHS="${NUMERIC_EPOCHS:-10}"
RCARE_EPOCHS="${RCARE_EPOCHS:-12}"
PATIENCE="${PATIENCE:-5}"
LR="${LR:-5e-4}"
NUMERIC_LR="${NUMERIC_LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-2026}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"
RATIOS="${RATIOS:-0.05 0.10 0.20}"
RUN_STAGE="${RUN_STAGE:-both}"  # numeric, full, or both
ROOT_PATH="${ROOT_PATH:-.}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

run_numeric() {
  local pred_len="$1"
  local ratio="$2"
  local ratio_tag="${ratio/./p}"
  local text_csv="generated/${DATASET}/${DATASET}_sl${SEQ_LEN}_pl${pred_len}_text_M_semantic_v1.csv"
  local text_npz="generated/${DATASET}/${DATASET}_sl${SEQ_LEN}_pl${pred_len}_semantic_v1_text_features.npz"

  "${PYTHON_BIN}" run.py \
    --is_training 1 \
    --model_id "${DATASET}_sl${SEQ_LEN}_pl${pred_len}_r${ratio_tag}_numeric" \
    --model CARE_Forecast \
    --method_profile numeric_only \
    --data "${DATASET}" \
    --root_path "${ROOT_PATH}" \
    --data_path "dataset/${DATASET}.csv" \
    --text_path "${text_csv}" \
    --text_feature_path "${text_npz}" \
    --features "${FEATURES}" \
    --target "${TARGET}" \
    --freq "${FREQ}" \
    --split_mode "${SPLIT_MODE}" \
    --seq_len "${SEQ_LEN}" \
    --label_len "${LABEL_LEN}" \
    --pred_len "${pred_len}" \
    --text_dim "${TEXT_DIM}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --sem_dim "${SEM_DIM}" \
    --moving_avg_kernel 7 \
    --frft_init_alpha 0.4 \
    --train_ratio "${ratio}" \
    --train_ratio_seed "${SEED}" \
    --seed "${SEED}" \
    --train_epochs "${NUMERIC_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --patience "${PATIENCE}" \
    --des numeric_warmup \
    --checkpoints "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}"
}

run_full() {
  local pred_len="$1"
  local ratio="$2"
  local ratio_tag="${ratio/./p}"
  local text_csv="generated/${DATASET}/${DATASET}_sl${SEQ_LEN}_pl${pred_len}_text_M_semantic_v1.csv"
  local text_npz="generated/${DATASET}/${DATASET}_sl${SEQ_LEN}_pl${pred_len}_semantic_v1_text_features.npz"
  local extra_args=()
  if [[ -n "${NUMERIC_CKPT:-}" ]]; then
    extra_args+=(--pretrained_numeric_checkpoint "${NUMERIC_CKPT}" --freeze_numeric_backbone 1)
  fi

  "${PYTHON_BIN}" run.py \
    --is_training 1 \
    --model_id "${DATASET}_sl${SEQ_LEN}_pl${pred_len}_r${ratio_tag}_rcare" \
    --model CARE_Forecast \
    --method_profile privileged_bridge \
    --data "${DATASET}" \
    --root_path "${ROOT_PATH}" \
    --data_path "dataset/${DATASET}.csv" \
    --text_path "${text_csv}" \
    --text_feature_path "${text_npz}" \
    --features "${FEATURES}" \
    --target "${TARGET}" \
    --freq "${FREQ}" \
    --split_mode "${SPLIT_MODE}" \
    --seq_len "${SEQ_LEN}" \
    --label_len "${LABEL_LEN}" \
    --pred_len "${pred_len}" \
    --text_dim "${TEXT_DIM}" \
    --student_text_cols llm_history_text llm_history_prior_text \
    --teacher_text_cols llm_future_text llm_residual_text \
    --hidden_dim "${HIDDEN_DIM}" \
    --sem_dim "${SEM_DIM}" \
    --moving_avg_kernel 7 \
    --frft_init_alpha 0.4 \
    --residual_planner_type cross_attn \
    --train_ratio "${ratio}" \
    --train_ratio_seed "${SEED}" \
    --seed "${SEED}" \
    --train_epochs "${RCARE_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --numeric_learning_rate "${NUMERIC_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --patience "${PATIENCE}" \
    --des rcare \
    --checkpoints "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    "${extra_args[@]}"
}

for pred_len in ${PRED_LENS}; do
  for ratio in ${RATIOS}; do
    echo "=== ${DATASET}: pred_len=${pred_len}, train_ratio=${ratio}, stage=${RUN_STAGE} ==="
    case "${RUN_STAGE}" in
      numeric) run_numeric "${pred_len}" "${ratio}" ;;
      full) run_full "${pred_len}" "${ratio}" ;;
      both) run_numeric "${pred_len}" "${ratio}"; run_full "${pred_len}" "${ratio}" ;;
      *) echo "Unknown RUN_STAGE=${RUN_STAGE}; use numeric, full, or both" >&2; exit 2 ;;
    esac
  done
done
