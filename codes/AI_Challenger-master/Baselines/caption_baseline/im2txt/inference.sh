#!/usr/bin/env bash

cd /home/pingan_ai/dxq/codes/AI_Challenger-master/Baselines/caption_baseline/im2txt/im2txt
CHECKPOINT_PATH="/media/pingan_ai/AI_Challenger/ImageCaption/model/train/train"
VOCAB_FILE="/media/pingan_ai/AI_Challenger/ImageCaption/ch2tf_word_counts/train/word_counts.txt"
export CUDA_VISIBLE_DEVICES="1"
IMAGE_DIR='/home/pingan_ai/dxq/project/my_test_img/'
OUTJSON='/home/pingan_ai/dxq/project/test_output/my_test_img.json'
python run_inference.py \
 	--checkpoint_path=${CHECKPOINT_PATH} \
  	--vocab_file=${VOCAB_FILE} \
  	--image_dir=${IMAGE_DIR}\
  	--out_predict_json=${OUTJSON}

