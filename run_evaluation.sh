# CUDA_VISIBLE_DEVICES=1 python evaluation.py \
#     --output_dir="./save_eng" \
#     --dataset_path="/home/samuel/emotion-elderly/datasets" \
#     --label_column_name="language" \
#     # --max_eval_samples=52


CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python evaluation.py \
    --model_name_or_path facebook/wav2vec2-base \
    --dataset_path="/home/samuel/emotion-elderly/datasets" \
    --output_dir wav2vec2-base-ft-keyword-spotting \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_eval \
    --fp16 \
    --learning_rate 3e-5 \
    --max_length_seconds 1 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 16 \
    --eval_accumulation_steps 2 \
    --dataloader_num_workers 4 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --save_total_limit 3 \
    --seed 0