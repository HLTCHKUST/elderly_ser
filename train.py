import os, sys
import logging
import numpy as np
import pandas as pd
import argparse

import torch

import re
import json 
from utils.args_helper import ModelArguments, DataTrainingArguments, TrainingArguments
# from datasets import DatasetDict
from transformers import (AutoProcessor, AutoModelForAudioClassification, Trainer,Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor, 
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
    Wav2Vec2Config,
    HfArgumentParser,set_seed)

from transformers.trainer_utils import get_last_checkpoint
from utils import data_loader


# processor = AutoProcessor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")




    



#####
# Entry Point
#####
def main():
    ###
    # Parsing & Initialization
    ###
    # Parse argument
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        print(sys.argv)
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[2].endswith(".json"):
        parser = HfArgumentParser((TrainingArguments))
        preprocessor_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[2]))
    else:
        model_args, data_args, training_args= parser.parse_args_into_dataclasses()

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
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(
            "./log/log__{}".format(model_args.model_name_or_path.replace("/", "_")), mode="w")],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to warn of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity(transformers.logging.WARNING)
    logger.info("Training/evaluation parameters %s", training_args)



    # Initialize our dataset and prepare it for the emotion classification task.
    data = data_loader.load_dataset(data_args.dataset_path)
    raw_datasets = {}
    for d in data:
        dset = datasets.DatasetDict()
        dset["train"] = d["data"][0]
        dset["validation"] = d["data"][1]
        for test_dset_name, test_dset in d["data"][-1].items():
            dset[f'test-{test_dset_name}'] = test_dset
        raw_datasets[f'{d["lang"]}-{d["group"]}'] = dset.copy()

    for k, v in raw_datasets.items():
        print(f'=== {k} ===')
        print(v)
        print()

    print(raw_datasets["zho-elderly"]["test-csed"])
    print()
    print(raw_datasets["zho-elderly"]["test-csed"][:3])

    train_fn(training_args)
    
if __name__ == '__main__':
    main()