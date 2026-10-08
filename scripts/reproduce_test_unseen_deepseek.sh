#!/usr/bin/env bash
set -euo pipefail

# Full-split GeoNav evaluation with an OpenAI-compatible DeepSeek endpoint.
# Credentials and provider-specific model ids are supplied by the environment.
: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY in the environment}"
: "${DEEPSEEK_BASE_URL:?Set DEEPSEEK_BASE_URL in the environment}"
: "${GEONAV_VLM_MODEL:?Set GEONAV_VLM_MODEL to the provider's vision model id}"
: "${GEONAV_LLM_MODEL:?Set GEONAV_LLM_MODEL to the provider's language model id}"

export GEONAV_API_KEY="${GEONAV_API_KEY:-$DEEPSEEK_API_KEY}"
export GEONAV_VLM_BASE_URL="${GEONAV_VLM_BASE_URL:-$DEEPSEEK_BASE_URL}"
export GEONAV_LLM_BASE_URL="${GEONAV_LLM_BASE_URL:-$DEEPSEEK_BASE_URL}"

OUTPUT_DIR="${GEONAV_OUTPUT_DIR:-results/geonav_deepseek_full_test_unseen}"

python main_geonav.py \
  --mode eval \
  --deployment online \
  --split test_unseen \
  --altitude 50 \
  --map_size 240 \
  --map_meters 410 \
  --gsam_use_segmentation_mask \
  --gsam_box_threshold 0.20 \
  --gsam_use_map_cache \
  --eval_batch_size 50 \
  --eval_max_timestep 20 \
  --num_workers 8 \
  --output_dir "$OUTPUT_DIR"
