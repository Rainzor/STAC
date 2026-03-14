#!/bin/bash
set -e

WORKDIR='.'
OUTPUT_DIR="${WORKDIR}/eval_recon"

MODEL_NAME="causalvggt"
# base_model: 可由外部覆盖。用法: BASE_MODEL=streamvggt ./run_all.sh  或  ./run_all.sh streamvggt
BASE_MODEL="${1:-${BASE_MODEL:-stream3r}}"
KF_EVERY=5

echo "base_model: ${BASE_MODEL}"

DATASETS=("NRGBD" "7scenes")

for DATASET in "${DATASETS[@]}"; do 
    echo "Evaluating dataset: ${DATASET}"
    echo "=========================================="
    python eval/long_recon/launch.py \
        --output_dir "${OUTPUT_DIR}" \
        --size 518 \
        --kf_every ${KF_EVERY} \
        --model_name "${MODEL_NAME}" \
        --base_model "${BASE_MODEL}" \
        --dataset_type "${DATASET}" \
        --save_tag "baseline" \
        --vis_tag "causal" \
        --mode "window_kv" --streaming -win -1
    done
