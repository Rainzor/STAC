#!/bin/bash
set -e

workdir='.'

default_datasets=('bonn' 'sintel' 'kitti')
datasets=("${@:-${default_datasets[@]}}")
base_models=('stream3r' 'streamvggt')
output_dir="${workdir}/exp_results/video_depth"

for model in "${base_models[@]}"; do
    echo "Evaluating model: $model"
    for data in "${datasets[@]}"; do
        causal_dir="${workdir}/exp_results/video_depth/${model}/${data}"
        echo "Saving depth results to: $causal_dir"
        python eval/video_depth/launch.py \
        --output_dir="$causal_dir" \
        --model_name causalvggt --base_model="$model" \
        --mode window_chunk_merge --streaming \
        --eval_dataset="$data" \
        -win 4 -hh 2 -ret_sz 2 -ck 4

        python eval/video_depth/eval_depth.py \
        --output_dir "$causal_dir" \
        --eval_dataset "$data" \
        --align "scale"
    done
done
