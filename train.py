import os, sys
import logging
import numpy as np
import pandas as pd
import argparse
import torch
import torchaudio
from  transformers.utils.logging import set_verbosity, enable_default_handler, enable_explicit_format
import re
import json 
from utils.args_helper import ModelArguments, DataTrainingArguments, TrainingArguments
from datasets import DatasetDict, concatenate_datasets
from transformers import (
    AutoProcessor, AutoModelForAudioClassification, 
    Trainer, Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor, AutoConfig,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Config,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    HfArgumentParser,set_seed,
    EarlyStoppingCallback
)

import IPython.display as ipd

from transformers.trainer_utils import get_last_checkpoint
from utils import data_loader
from datasets import load_from_disk, set_caching_enabled
from utils.metrics import compute_metrics
from models.modeling_wav2vec2 import Wav2Vec2ForMultilabelSequenceClassification

set_caching_enabled(True)


logger = logging.getLogger(__name__)    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def eval_fn(training_args):

#####
# Entry Point
#####
def main():
    ###
    # Parsing & Initialization
    ###
    # Parse argument
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv)  == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args= parser.parse_args_into_dataclasses()

    # Set directory path
    training_args.output_dir="{}/{}_{}_{}".format(
        training_args.output_dir, data_args.training_language, data_args.training_age_group, model_args.model_name_or_path
    )
    cache_dir_path = "./{}/{}_{}_{}".format(
        model_args.cache_dir, data_args.training_language, data_args.training_age_group, model_args.model_name_or_path
    )
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs(cache_dir_path, exist_ok=True)
    
    # Set random seed
    set_seed(training_args.seed)
    
    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    ###
    # Prepare logger
    ###
    # Init logging
    os.makedirs("./log", exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    ###
    # Loading  Dataset
    ###
    # data_loader.load_dataset => return [
    #     {
    #         "lang": "eng",
    #         "group": "others",
    #         "data": (
    #             df_to_dataset(trn_en_others_df),
    #             df_to_dataset(val_en_others_df),
    #             {dset: df_to_dataset(df) for dset, df in tst_en_others_df.groupby('dataset')}
    #         )
    #     }, {
    #         "lang": "eng",
    #         "group": "elderly",
    #         "data": (
    #             df_to_dataset(trn_en_elderly_df),
    #             df_to_dataset(val_en_elderly_df),
    #             {dset: df_to_dataset(df) for dset, df in tst_en_elderly_df.groupby('dataset')}
    #         )
    #     }, {
    #         "lang": "zho",
    #         "group": "others",
    #         "data": (
    #             df_to_dataset(trn_zh_others_df),
    #             df_to_dataset(val_zh_others_df),
    #             {dset: df_to_dataset(df) for dset, df in tst_zh_others_df.groupby('dataset')}
    #         )
    #     }, {
    #         "lang": "zho",
    #         "group": "elderly",
    #         "data": (
    #             df_to_dataset(trn_zh_elderly_df),
    #             df_to_dataset(val_zh_elderly_df),
    #             {dset: df_to_dataset(df) for dset, df in tst_zh_elderly_df.groupby('dataset')}
    #         )
    #     },
    # ]

    print('Loading dataset...')
    dataset_dict = data_loader.load_dataset(data_args.dataset_path, mix_speakers=data_args.mix_speakers)
    # print(dataset_dict)

    train_dsets = []
    valid_dsets = []
    test_dsets = {}
    for group_dict in dataset_dict:
        for dset_name, dset in group_dict['data'][2].items():
            test_dsets[f"{group_dict['lang']}/{group_dict['group']}/{dset_name}"] = dset

        if not (data_args.training_language == 'eng-zho-yue' or
            (data_args.training_language == 'eng-zho' and (group_dict['lang'] == 'eng' or group_dict['lang'] == 'zho')) or 
            (data_args.training_language == 'eng' and group_dict['lang'] == 'eng') or 
            (data_args.training_language == 'zho' and group_dict['lang'] == 'zho') or
            (data_args.training_language == 'yue' and group_dict['lang'] == 'yue')):
            continue

        if not (data_args.training_age_group == 'all' or
            (data_args.training_age_group == 'elderly' and group_dict['group'] == 'elderly') or
            (data_args.training_age_group == 'others' and group_dict['group'] == 'others')):
            continue

        train_dsets.append(group_dict['data'][0])
        valid_dsets.append(group_dict['data'][1])

    train_dset = concatenate_datasets(train_dsets)
    valid_dset = concatenate_datasets(valid_dsets)
    test_dset_dict = DatasetDict(test_dsets)
    
    ###
    # Model Initialization
    ###
    label_list = [
        'sadness', 'fear', 'angry', 'happiness', 'disgust', 'neutral', 'surprise', 
        'positive', 'negative', 'excitement', 'frustrated', 'other', 'unknown'
    ]
    if 'wav2vec' in model_args.model_name_or_path:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)
        target_sampling_rate = feature_extractor.sampling_rate
        model = Wav2Vec2ForMultilabelSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=len(label_list)).to(device)
    else:
        raise('Not Implemented Error')
        # processor = WhisperProcessor.from_pretrained(model_args.model_name_or_path)
        # model = WhisperForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

    ###
    # Data Preprocessor
    ###
    # print('TRAIN')
    # print(train_dset)
    # print('VALID')
    # print(valid_dset)
    # print('TEST')
    # print(test_dset_dict)
    
    def data_transforms(batch):
        """Apply train_transforms across a batch."""
        output_batch = {"input_values": [], "labels": [], "labels_mask": []}
        for audio, labels in zip(batch['audio'], batch['labels']):
            wav = audio["array"]
            label_array = np.array([int(label) for label in labels])
            label_mask = label_array != -100
            label_mask = label_mask / label_mask.sum()
            
            output_batch["input_values"].append(wav)
            output_batch["labels"].append(label_array)
            output_batch["labels_mask"].append(label_mask)
        return output_batch

    # Set the dataset transforms
    train_dset.set_transform(data_transforms, columns=['audio', 'labels'], output_all_columns=False)
    valid_dset.set_transform(data_transforms, columns=['audio', 'labels'], output_all_columns=False)
    test_dset_dict.set_transform(data_transforms, columns=['audio', 'labels'], output_all_columns=False)
    
    ###
    # Training Phase
    ###
    
    # Initialize Trainer
    training_args.remove_unused_columns = False
    trainer = Trainer(
        train_dataset=train_dset,
        eval_dataset=valid_dset,
        model=model,
        args=training_args,
        # compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]        
    )

    print('*** Training Phase ***')

    # use last checkpoint if exist
    if os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.save_state()
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
#     ###
#     # Evaluation Phase
#     ###
#     results = {}
#     logger.info("*** Evaluation Phase ***")
    
#     for test_dset_key, test_dset in test_dset_dict.items():
#         trainer.compute_metrics = compute_metrics[test_dset_key]
#         metrics = trainer.evaluate(eval_dataset=test_dset)
#         metrics["eval_samples"] = len(test_dset)
        
#         keys = list(metrics.keys())
#         for key in keys:
#             metrics[key.replace('eval_',f'eval_{test_dset_key}')] = metrics[key]
#             del metrics[key]
#         metrics[f"eval_{test_dset_key}_samples"] = len(vectorized_datasets[subset])

#         trainer.log_metrics(f"eval_{test_dset_key}", metrics)
#         trainer.save_metrics(f"eval_{test_dset_key}", metrics)
    
    
if __name__ == '__main__':
    main()