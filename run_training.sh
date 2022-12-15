#! bin/bash

export CUDA_VISIBLE_DEVICES=0,1
python train.py   --model_name_or_path="harshit345/xlsr-wav2vec-speech-emotion-recognition"\
    --output_dir="./save" \
    --dataset_path="/home/samuel/emotion-elderly/datasets" \