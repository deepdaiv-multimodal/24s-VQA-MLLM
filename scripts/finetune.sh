#!/bin/bash
# This script is used to finetune the pretrained BEiT3 model.

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2;;
    --task)
      TASK="$2"
      shift 2;;
    --pretrained_model)
      PRETRAINED_MODEL_PATH="$2"
      shift 2;;
    --version)
      VERSION="$2"
      shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

TASK=${TASK:-ok} # task name, one of ['ok', 'aok_val', 'aok_test'], default 'ok'
GPU=${GPU:-0} # GPU id(s) you want to use, default '0'
VERSION=${VERSION:-finetuning_okvqa} # version name, default 'finetuning_for_$TASK'

# Set the environment variable for CUDA
export CUDA_VISIBLE_DEVICES=$GPU
# CUDA_VISIBLE_DEVICES=$GPU

echo "Using GPU: $GPU"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# run python script
python main.py \
    --task $TASK --run_mode finetune \
    --cfg configs/finetune.yml \
    --version $VERSION \
    --gpu $GPU --seed 99 --grad_accu 2
