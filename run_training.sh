export CUDA_VISIBLE_DEVICES=1

# # ENG-ZHO
# python train.py  --model_name_or_path="facebook/wav2vec2-large-xlsr-53"\
#     --output_dir="./save" --cache_dir="./cache/"\
#     --dataset_path="/home/samuel/emotion-elderly/datasets" \
#     --training_language eng-zho --training_age_group all \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#     --gradient_accumulation_steps=1 --gradient_checkpointing \
#     --dataloader_num_workers=8 --dataloader_pin_memory \
#     --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
#     --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#     --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#     --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
#     --greater_is_better=False --load_best_model_at_end=True

# python train.py  --model_name_or_path="facebook/wav2vec2-large-xlsr-53"\
#     --output_dir="./save" --cache_dir="./cache/"\
#     --dataset_path="/home/samuel/emotion-elderly/datasets" \
#     --training_language eng-zho --training_age_group elderly \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#     --gradient_accumulation_steps=1 --gradient_checkpointing \
#     --dataloader_num_workers=8 --dataloader_pin_memory \
#     --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
#     --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#     --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#     --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
#     --greater_is_better=False --load_best_model_at_end=True
     
# python train.py  --model_name_or_path="facebook/wav2vec2-large-xlsr-53"\
#     --output_dir="./save" --cache_dir="./cache/"\
#     --dataset_path="/home/samuel/emotion-elderly/datasets" \
#     --training_language eng-zho --training_age_group others \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#     --gradient_accumulation_steps=1 --gradient_checkpointing \
#     --dataloader_num_workers=8 --dataloader_pin_memory \
#     --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
#     --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#     --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#     --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
#     --greater_is_better=False --load_best_model_at_end=True
     
# # ENG
# python train.py  --model_name_or_path="facebook/wav2vec2-large-xlsr-53"\
#     --output_dir="./save" --cache_dir="./cache/"\
#     --dataset_path="/home/samuel/emotion-elderly/datasets" \
#     --training_language eng --training_age_group all \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#     --gradient_accumulation_steps=1 --gradient_checkpointing \
#     --dataloader_num_workers=8 --dataloader_pin_memory \
#     --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
#     --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#     --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#     --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
#     --greater_is_better=False --load_best_model_at_end=True
     
# python train.py  --model_name_or_path="facebook/wav2vec2-large-xlsr-53"\
#     --output_dir="./save" --cache_dir="./cache/"\
#     --dataset_path="/home/samuel/emotion-elderly/datasets" \
#     --training_language eng --training_age_group elderly \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#     --gradient_accumulation_steps=1 --gradient_checkpointing \
#     --dataloader_num_workers=8 --dataloader_pin_memory \
#     --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
#     --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#     --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#     --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
#     --greater_is_better=False --load_best_model_at_end=True
     
# python train.py  --model_name_or_path="facebook/wav2vec2-large-xlsr-53"\
#     --output_dir="./save" --cache_dir="./cache/"\
#     --dataset_path="/home/samuel/emotion-elderly/datasets" \
#     --training_language eng --training_age_group others \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#     --gradient_accumulation_steps=1 --gradient_checkpointing \
#     --dataloader_num_workers=8 --dataloader_pin_memory \
#     --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
#     --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#     --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#     --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
#     --greater_is_better=False --load_best_model_at_end=True

# # ZHO
# python train.py  --model_name_or_path="facebook/wav2vec2-large-xlsr-53"\
#     --output_dir="./save" --cache_dir="./cache/"\
#     --dataset_path="/home/samuel/emotion-elderly/datasets" \
#     --training_language zho --training_age_group all \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#     --gradient_accumulation_steps=1 --gradient_checkpointing \
#     --dataloader_num_workers=8 --dataloader_pin_memory \
#     --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
#     --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#     --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#     --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
#     --greater_is_better=False --load_best_model_at_end=True
     
# python train.py  --model_name_or_path="facebook/wav2vec2-large-xlsr-53"\
#     --output_dir="./save" --cache_dir="./cache/"\
#     --dataset_path="/home/samuel/emotion-elderly/datasets" \
#     --training_language zho --training_age_group elderly \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#     --gradient_accumulation_steps=1 --gradient_checkpointing \
#     --dataloader_num_workers=8 --dataloader_pin_memory \
#     --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
#     --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#     --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#     --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
#     --greater_is_better=False --load_best_model_at_end=True
     
# python train.py  --model_name_or_path="facebook/wav2vec2-large-xlsr-53"\
#     --output_dir="./save" --cache_dir="./cache/"\
#     --dataset_path="/home/samuel/emotion-elderly/datasets" \
#     --training_language zho --training_age_group others \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#     --gradient_accumulation_steps=1 --gradient_checkpointing \
#     --dataloader_num_workers=8 --dataloader_pin_memory \
#     --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
#     --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#     --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#     --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
#     --greater_is_better=False --load_best_model_at_end=True

# YUE

python train.py  --model_name_or_path="facebook/wav2vec2-large-xlsr-53"\
    --output_dir="./save" --cache_dir="./cache/"\
    --dataset_path="/home/samuel/emotion-elderly/datasets" \
    --training_language yue --training_age_group all \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1 --gradient_checkpointing \
    --dataloader_num_workers=8 --dataloader_pin_memory \
    --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
    --greater_is_better=False --load_best_model_at_end=True

python train.py  --model_name_or_path="facebook/wav2vec2-large-xlsr-53"\
    --output_dir="./save" --cache_dir="./cache/"\
    --dataset_path="/home/samuel/emotion-elderly/datasets" \
    --training_language yue --training_age_group elderly \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1 --gradient_checkpointing \
    --dataloader_num_workers=8 --dataloader_pin_memory \
    --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
    --greater_is_better=False --load_best_model_at_end=True

python train.py  --model_name_or_path="facebook/wav2vec2-large-xlsr-53"\
    --output_dir="./save" --cache_dir="./cache/"\
    --dataset_path="/home/samuel/emotion-elderly/datasets" \
    --training_language yue --training_age_group others \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1 --gradient_checkpointing \
    --dataloader_num_workers=8 --dataloader_pin_memory \
    --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
    --greater_is_better=False --load_best_model_at_end=True

# # ENG-ZHO-YUE

# python train.py  --model_name_or_path="facebook/wav2vec2-large-xlsr-53"\
#     --output_dir="./save" --cache_dir="./cache/"\
#     --dataset_path="/home/samuel/emotion-elderly/datasets" \
#     --training_language eng-zho-yue --training_age_group all \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#     --gradient_accumulation_steps=1 --gradient_checkpointing \
#     --dataloader_num_workers=8 --dataloader_pin_memory \
#     --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
#     --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#     --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#     --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
#     --greater_is_better=False --load_best_model_at_end=True

# python train.py  --model_name_or_path="facebook/wav2vec2-large-xlsr-53"\
#     --output_dir="./save" --cache_dir="./cache/"\
#     --dataset_path="/home/samuel/emotion-elderly/datasets" \
#     --training_language eng-zho-yue --training_age_group elderly \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#     --gradient_accumulation_steps=1 --gradient_checkpointing \
#     --dataloader_num_workers=8 --dataloader_pin_memory \
#     --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
#     --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#     --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#     --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
#     --greater_is_better=False --load_best_model_at_end=True

# python train.py  --model_name_or_path="facebook/wav2vec2-large-xlsr-53"\
#     --output_dir="./save" --cache_dir="./cache/"\
#     --dataset_path="/home/samuel/emotion-elderly/datasets" \
#     --training_language eng-zho-yue --training_age_group others \
#     --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
#     --gradient_accumulation_steps=1 --gradient_checkpointing \
#     --dataloader_num_workers=8 --dataloader_pin_memory \
#     --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
#     --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
#     --evaluation_strategy=epoch --eval_steps=1 --eval_accumulation_steps=1 \
#     --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
#     --greater_is_better=False --load_best_model_at_end=True