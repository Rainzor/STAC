#!/bin/bash
set -e

WORKDIR='.'
OUTPUT_DIR="${WORKDIR}/eval_recon"

# External overrides (examples):
#   BASE_MODEL=streamvggt MODE=stac ATTN_BACKEND=triton ATTN_SUBSAMPLE=1.0 bash eval/long_recon/run.sh
MODEL_NAME="${MODEL_NAME:-causalvggt}"
BASE_MODEL="${BASE_MODEL:-stream3r}"
MODE="${MODE:-stac}"
KF_EVERY="${KF_EVERY:-5}"
ATTN_BACKEND="${ATTN_BACKEND:-cuda}"
ATTN_SUBSAMPLE="${ATTN_SUBSAMPLE:-1.0}"

DATASETS=("NRGBD" "7scenes")

# Auto tag by backend mode (override by VIS_TAG=...)
if [[ "${ATTN_BACKEND}" == "cuda" ]]; then
    AUTO_VIS_TAG="attn_cuda_sub${ATTN_SUBSAMPLE}"
else
    AUTO_VIS_TAG="attn_triton"
fi
VIS_TAG="${VIS_TAG:-${AUTO_VIS_TAG}}"

for DATASET in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Evaluating dataset: ${DATASET}"
    echo "MODEL_NAME=${MODEL_NAME} BASE_MODEL=${BASE_MODEL} MODE=${MODE} ATTN_BACKEND=${ATTN_BACKEND} ATTN_SUBSAMPLE=${ATTN_SUBSAMPLE} VIS_TAG=${VIS_TAG}"
    echo "=========================================="
    python eval/long_recon/launch.py \
        --output_dir "${OUTPUT_DIR}" \
        --size 518 \
        --kf_every ${KF_EVERY} \
        --model_name "${MODEL_NAME}" \
        --base_model "${BASE_MODEL}" \
        --dataset_type "${DATASET}" \
        --save_tag "stac" \
        --vis_tag "${VIS_TAG}" \
        --mode "${MODE}" \
        --attn_backend "${ATTN_BACKEND}" \
        --subsample "${ATTN_SUBSAMPLE}"
done
