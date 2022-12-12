#! bin/bash

CUDA_VISIBLE_DEVICES=0 
python train.py --model_name_or_path="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition" \
    --output_dir="./save" \
    --dataset_path="/home/samuel/emotion-elderly/datasets" \