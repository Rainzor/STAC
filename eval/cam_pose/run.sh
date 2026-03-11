#!/bin/bash
set -e

WORKDIR='.'
OUTPUT_DIR="${WORKDIR}/eval_cam_results"

MODEL_NAME="causalvggt"
BASE_MODEL="stream3r"

MODE="window_chunk_merge"
WIN=4
HH=2
RET_SZ=2
CK=4

SAVE_TAG="stac"
VIS_TAG="stac_w4h2r2c4"

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
        --mode "${MODE}" \
        --streaming \
        -win ${WIN} \
        -hh ${HH} \
        -ret_sz ${RET_SZ} \
        -ret_buf \
        -ck ${CK} \
        --tag "${SAVE_TAG}" \
        --vis_tag "${VIS_TAG}"
done
