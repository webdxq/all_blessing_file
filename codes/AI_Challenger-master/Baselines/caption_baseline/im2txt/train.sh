#!/usr/bin/env bash
#outpath=/home/store-1-img/zhenghe/ai_challenger_caption_train_output/train.log
cd /home/pingan_ai/dxq/codes/AI_Challenger-master/Baselines/caption_baseline/im2txt/im2txt
TFRECORD_DIR="/media/pingan_ai/AI_Challenger/ImageCaption/TFrecords/train"
INCEPTION_CHECKPOINT="/media/pingan_ai/AI_Challenger/ImageCaption/inception_v3.ckpt"
MODEL_DIR="/media/pingan_ai/AI_Challenger/ImageCaption/model/train"
export CUDA_VISIBLE_DEVICES="1"
python train.py \
  --input_file_pattern="${TFRECORD_DIR}/train-?????-of-00280" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  # --number_of_steps=500 #> ${outpath} 2>&1 &