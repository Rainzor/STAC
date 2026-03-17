#!/bin/bash
set -e

workdir='.'

default_datasets=('bonn' 'sintel' 'kitti')
datasets=("${@:-${default_datasets[@]}}")
base_models=('stream3r')
output_dir="${workdir}/eval_depth/video_depth"

for model in "${base_models[@]}"; do
    echo "Evaluating model: $model"
    for data in "${datasets[@]}"; do
        causal_dir="${workdir}/eval_depth/video_depth/${model}/${data}"
        echo "Saving depth results to: $causal_dir"
        python eval/video_depth/launch.py \
        --output_dir="$causal_dir" \
        --size 518 \
        --model_name causalvggt --base_model="$model" \
        --mode stac \
        --eval_dataset="$data"

        python eval/video_depth/eval_depth.py \
        --output_dir "$causal_dir" \
        --eval_dataset "$data" \
        --align "scale"
    done
done
