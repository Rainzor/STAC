#!/bin/bash
set -e

WORKDIR='.'
OUTPUT_DIR="${WORKDIR}/eval_cam_results"

MODEL_NAME="causalvggt"
BASE_MODEL="stream3r"

SAVE_TAG="stac"
VIS_TAG="stac"

DATASETS=("tum" "scannet" "sintel")

for DATASET in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Evaluating dataset: ${DATASET}"
    echo "=========================================="
    python eval/cam_pose/launch.py \
        --output_dir "${OUTPUT_DIR}" \
        --size 518 \
        --model_name "${MODEL_NAME}" \
        --base_model "${BASE_MODEL}" \
        --dataset_type "${DATASET}" \
        --mode stac \
        --tag "${SAVE_TAG}" \
        --vis_tag "${VIS_TAG}"
done
