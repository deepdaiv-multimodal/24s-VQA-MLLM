#!/bin/bash
# This script is used to prompt InstructBLIP to generate final answers.

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2;;
    --version)
      VERSION="$2"
      shift 2;;
    --examples_path)
      EXAMPLES_PATH="$2"
      shift 2;;
    --candidates_path)
      CANDIDATES_PATH="$2"
      shift 2;;
    --captions_path)
      CAPTIONS_PATH="$2"
      shift 2;;
    --image_path)
      IMAGE_PATH="$2"
      shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

TASK=${TASK:-ok} # task name, one of ['ok', 'aok_val', 'aok_test'], default 'ok'
VERSION=${VERSION:-"prompt_okvqa"} # version name, default 'prompt_for_$TASK'
EXAMPLES_PATH=${EXAMPLES_PATH:-"assets/answer_aware_examples_okvqa.json"} # path to the examples, default is the result from our experiments
CANDIDATES_PATH=${CANDIDATES_PATH:-"assets/candidates_okvqa.json"} # path to the candidates, default is the result from our experiments
CAPTIONS_PATH=${CAPTIONS_PATH:-"assets/captions_okvqa.json"} # path to the captions, default is the result from our experiments
# IMAGE_PATH=${IMAGE_PATH:-"assets/images"} # path to the images folder, default is 'assets/images'
#GPU=${GPU:-0} # GPU id(s) you want to use, default '0'

#CUDA_VISIBLE_DEVICES=6 \
python main.py \
    --task $TASK --run_mode prompt \
    --version $VERSION \
    --cfg configs/prompt.yml \
    --examples_path $EXAMPLES_PATH \
    --candidates_path $CANDIDATES_PATH \
    --captions_path $CAPTIONS_PATH \
    #--image_path $IMAGE_PATH \
    # --image_path $IMAGE_PATH\
    #--gpu $GPU
