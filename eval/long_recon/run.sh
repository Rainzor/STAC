#!/bin/bash
set -e

WORKDIR='.'
OUTPUT_DIR="${WORKDIR}/eval_recon"

MODEL_NAME="causalvggt"
BASE_MODEL="stream3r"

MODE="window_chunk_merge"
WIN=4
HH=2
RET_SZ=2
CK=4
KF_EVERY=5

SAVE_TAG="stac"
VIS_TAG="w4h2r2c4"

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
        --save_tag "${SAVE_TAG}" \
        --vis_tag "${VIS_TAG}" \
        --mode "${MODE}" \
        --streaming \
        -ck ${CK} \
        -win ${WIN} \
        -hh ${HH} \
        -ret_sz ${RET_SZ} \
        -ret_buf
done
