#!/bin/bash
set -e

workdir='.'

datasets=$1
base_models=('stream3r' 'streamvggt')
output_dir="${workdir}/eval_depth/video_depth"

for model in "${base_models[@]}"; do
    echo "Evaluating model: $model"
    for data in "${datasets[@]}"; do
        causal_dir="${workdir}/eval_depth/video_depth/${model}/${data}"
        echo "Saving depth results to: $causal_dir"
        python eval/video_depth/launch.py \
        --output_dir="$causal_dir" \
        --model_name causalvggt --base_model="$model" \
        --mode causal --streaming \
        --eval_dataset="$data"

        python eval/video_depth/eval_depth.py \
        --output_dir "$causal_dir" \
        --eval_dataset "$data" \
        --align "scale"

        window_dir="${workdir}/eval_depth/video_depth/${model}-W/${data}"
        
        echo "Saving depth results to: $window_dir"

        python eval/video_depth/launch.py \
        --output_dir="$window_dir" \
        --model_name causalvggt --base_model="$model" \
        --mode window_kv --streaming \
        --eval_dataset="$data" \
        -win 8

        python eval/video_depth/eval_depth.py \
        --output_dir "$window_dir" \
        --eval_dataset "$data" \
        --align "scale"

        STAC_dir="${workdir}/eval_depth/video_depth/${model}-STAC/${data}"

        echo "Saving depth results to: $STAC_dir"

        python eval/video_depth/launch.py \
        --output_dir="$STAC_dir" \
        --model_name causalvggt --base_model="$model" \
        --mode stac \
        --eval_dataset="$data"

        python eval/video_depth/eval_depth.py \
        --output_dir "$STAC_dir" \
        --eval_dataset "$data" \
        --align "scale"
        
    done
done
