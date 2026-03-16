#!/bin/bash
# One run = one SUBSAMPLE. Use in different tmux windows with different args.
# Usage:
#   ./run_subsample.sh 0.25
#   SUBSAMPLE=0.50 ./run_subsample.sh
set -e

if [ -n "${1:-}" ]; then
    SUBSAMPLE="$1"
elif [ -n "${SUBSAMPLE:-}" ]; then
    :  # use env SUBSAMPLE
else
    echo "Usage: $0 <SUBSAMPLE>   or   SUBSAMPLE=0.25 $0"
    exit 1
fi
WORKDIR='.'
OUTPUT_DIR="${WORKDIR}/eval_recon_subsample_392"

MODEL_NAME="causalvggt"
BASE_MODELS=(stream3r streamvggt)
KF_EVERY=5

DATASETS=("NRGBD" "7scenes")

# tag: 0.25 -> stac_cuda_samp_25, 1.0 -> stac_cuda_samp_100
SAMP_TAG=$(awk "BEGIN { printf \"%.0f\", ${SUBSAMPLE} * 100 }")
VIS_TAG="stac_cuda_samp_${SAMP_TAG}"

echo "=========================================="
echo "SUBSAMPLE=${SUBSAMPLE} -> vis_tag=${VIS_TAG}"
echo "=========================================="

for BASE_MODEL in "${BASE_MODELS[@]}"; do
    echo "---------- base_model: ${BASE_MODEL} ----------"
    for DATASET in "${DATASETS[@]}"; do
        echo "Evaluating dataset: ${DATASET}"
        SUBSAMPLE="${SUBSAMPLE}" python eval/long_recon/launch.py \
            --output_dir "${OUTPUT_DIR}" \
            --size 518 \
            --kf_every ${KF_EVERY} \
            --model_name "${MODEL_NAME}" \
            --base_model "${BASE_MODEL}" \
            --dataset_type "${DATASET}" \
            --save_tag "stac" \
            --vis_tag "${VIS_TAG}" \
            --mode stac
    done
done
