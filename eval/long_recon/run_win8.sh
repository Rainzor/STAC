#!/bin/bash
set -e

WORKDIR='.'
OUTPUT_DIR="${WORKDIR}/eval_recon"

MODEL_NAME="causalvggt"
BASE_MODEL="stream3r"
KF_EVERY=5

echo "base_model: ${BASE_MODEL}"

DATASETS=("NRGBD" "7scenes")

for DATASET in "${DATASETS[@]}"; do

    echo "=========================================="
    echo "Evaluating dataset: ${DATASET}"
    echo "=========================================="
    python eval/long_recon/launch.py \
        --output_dir "${OUTPUT_DIR}" \
        --size 518 \
        --kf_every ${KF_EVERY} \
        --model_name "${MODEL_NAME}" \
        --base_model "${BASE_MODEL}" \
        --dataset_type "${DATASET}" \
        --save_tag "window" \
        --vis_tag "win8" \
        --mode "window_kv" --streaming -win 8
    
        echo "=========================================="

done
