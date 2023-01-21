export CUDA_VISIBLE_DEVICES=0

# ENG-ZHO
python evaluation.py  --model_name_or_path="./save/eng-zho_all_facebook/wav2vec2-large-xlsr-53"\
    --output_dir="./save/eng-zho_all_facebook/wav2vec2-large-xlsr-53" --cache_dir="./cache/eng-zho_all_facebook/wav2vec2-large-xlsr-53"\
    --dataset_path="/home/samuel/emotion-elderly/datasets" \
    --validation_language eng-zho --validation_age_group all \
    --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
    --gradient_accumulation_steps=4 --gradient_checkpointing \
    --dataloader_num_workers=8 --dataloader_pin_memory \
    --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=4 --eval_accumulation_steps=1 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
    --greater_is_better=False --load_best_model_at_end=True

python evaluation.py  --model_name_or_path="./save/eng-zho_elderly_facebook/wav2vec2-large-xlsr-53"\
    --output_dir="./save/eng-zho_elderly_facebook/wav2vec2-large-xlsr-53" --cache_dir="./cache/eng-zho_elderly_facebook/wav2vec2-large-xlsr-53"\
    --dataset_path="/home/samuel/emotion-elderly/datasets" \
    --validation_language eng-zho --validation_age_group all \
    --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
    --gradient_accumulation_steps=4 --gradient_checkpointing \
    --dataloader_num_workers=8 --dataloader_pin_memory \
    --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=4 --eval_accumulation_steps=1 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
    --greater_is_better=False --load_best_model_at_end=True
     
python evaluation.py  --model_name_or_path="./save/eng-zho_others_facebook/wav2vec2-large-xlsr-53"\
    --output_dir="./save/eng-zho_others_facebook/wav2vec2-large-xlsr-53" --cache_dir="./cache/eng-zho_others_facebook/wav2vec2-large-xlsr-53"\
    --dataset_path="/home/samuel/emotion-elderly/datasets" \
    --validation_language eng-zho --validation_age_group all \
    --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
    --gradient_accumulation_steps=4 --gradient_checkpointing \
    --dataloader_num_workers=8 --dataloader_pin_memory \
    --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=4 --eval_accumulation_steps=1 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
    --greater_is_better=False --load_best_model_at_end=True
     
# ENG
python evaluation.py  --model_name_or_path="./save/eng_all_facebook/wav2vec2-large-xlsr-53"\
    --output_dir="./save/eng_all_facebook/wav2vec2-large-xlsr-53" --cache_dir="./cache/eng_all_facebook/wav2vec2-large-xlsr-53"\
    --dataset_path="/home/samuel/emotion-elderly/datasets" \
    --validation_language eng-zho --validation_age_group all \
    --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
    --gradient_accumulation_steps=4 --gradient_checkpointing \
    --dataloader_num_workers=8 --dataloader_pin_memory \
    --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=4 --eval_accumulation_steps=1 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
    --greater_is_better=False --load_best_model_at_end=True
     
python evaluation.py  --model_name_or_path="./save/eng_elderly_facebook/wav2vec2-large-xlsr-53"\
    --output_dir="./save/eng_elderly_facebook/wav2vec2-large-xlsr-53" --cache_dir="./cache/eng_elderly_facebook/wav2vec2-large-xlsr-53"\
    --dataset_path="/home/samuel/emotion-elderly/datasets" \
    --validation_language eng-zho --validation_age_group all \
    --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
    --gradient_accumulation_steps=4 --gradient_checkpointing \
    --dataloader_num_workers=8 --dataloader_pin_memory \
    --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=4 --eval_accumulation_steps=1 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
    --greater_is_better=False --load_best_model_at_end=True
     
python evaluation.py  --model_name_or_path="./save/eng_others_facebook/wav2vec2-large-xlsr-53"\
    --output_dir="./save/eng_others_facebook/wav2vec2-large-xlsr-53" --cache_dir="./cache/eng_others_facebook/wav2vec2-large-xlsr-53"\
    --dataset_path="/home/samuel/emotion-elderly/datasets" \
    --validation_language eng-zho --validation_age_group all \
    --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
    --gradient_accumulation_steps=4 --gradient_checkpointing \
    --dataloader_num_workers=8 --dataloader_pin_memory \
    --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=4 --eval_accumulation_steps=1 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
    --greater_is_better=False --load_best_model_at_end=True

# ZHO
python evaluation.py  --model_name_or_path="./save/zho_all_facebook/wav2vec2-large-xlsr-53"\
    --output_dir="./save/zho_all_facebook/wav2vec2-large-xlsr-53" --cache_dir="./cache/zho_all_facebook/wav2vec2-large-xlsr-53"\
    --dataset_path="/home/samuel/emotion-elderly/datasets" \
    --validation_language eng-zho --validation_age_group all \
    --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
    --gradient_accumulation_steps=4 --gradient_checkpointing \
    --dataloader_num_workers=8 --dataloader_pin_memory \
    --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=4 --eval_accumulation_steps=1 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
    --greater_is_better=False --load_best_model_at_end=True
     
python evaluation.py  --model_name_or_path="./save/zho_elderly_facebook/wav2vec2-large-xlsr-53"\
    --output_dir="./save/zho_elderly_facebook/wav2vec2-large-xlsr-53" --cache_dir="./cache/zho_elderly_facebook/wav2vec2-large-xlsr-53"\
    --dataset_path="/home/samuel/emotion-elderly/datasets" \
    --validation_language eng-zho --validation_age_group all \
    --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
    --gradient_accumulation_steps=4 --gradient_checkpointing \
    --dataloader_num_workers=8 --dataloader_pin_memory \
    --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=4 --eval_accumulation_steps=1 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
    --greater_is_better=False --load_best_model_at_end=True
     
python evaluation.py  --model_name_or_path="./save/zho_others_facebook/wav2vec2-large-xlsr-53"\
    --output_dir="./save/zho_others_facebook/wav2vec2-large-xlsr-53" --cache_dir="./cache/zho_others_facebook/wav2vec2-large-xlsr-53"\
    --dataset_path="/home/samuel/emotion-elderly/datasets" \
    --validation_language eng-zho --validation_age_group all \
    --per_device_train_batch_size=2 --per_device_eval_batch_size=2 \
    --gradient_accumulation_steps=4 --gradient_checkpointing \
    --dataloader_num_workers=8 --dataloader_pin_memory \
    --seed=0 --num_train_epochs=20 --learning_rate=5e-5 \
    --logging_strategy=steps --logging_steps=10 --report_to=tensorboard \
    --evaluation_strategy=epoch --eval_steps=4 --eval_accumulation_steps=1 \
    --save_steps=1 --save_strategy=epoch --save_total_limit=1 --fp16 \
    --greater_is_better=False --load_best_model_at_end=True