#! bin/bash

export CUDA_VISIBLE_DEVICES=0,1
python train.py  --model_name_or_path="harshit345/xlsr-wav2vec-speech-emotion-recognition"\
    --output_dir="./save" \
    --dataset_path="/home/samuel/emotion-elderly/datasets" \
    --training_language zho --training_age_group elderly \
    --per_device_train_batch_size=16 --per_device_eval_batch_size=16 \
    --dataloader_num_workers=16 --dataloader_pin_memory \
    --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
    --fp16 --fp16_backend=amp \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=100 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 \
    --metric_for_best_model=mer --greater_is_better=False --load_best_model_at_end=True
