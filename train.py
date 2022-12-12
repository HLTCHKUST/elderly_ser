import os, sys
import logging
import numpy as np
import pandas as pd
import argparse

import torch
from  transformers.utils.logging import    set_verbosity, enable_default_handler, enable_explicit_format
import re
import json 
from utils.args_helper import ModelArguments, DataTrainingArguments, TrainingArguments
from datasets import DatasetDict
from transformers import (AutoProcessor, AutoModelForAudioClassification, Trainer, Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor, AutoConfig,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
    Wav2Vec2Config,
    HfArgumentParser,set_seed)
import IPython.display as ipd

from transformers.trainer_utils import get_last_checkpoint
from utils import data_loader


logger = logging.getLogger(__name__)    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_fn(model, training_data, feature_extractor, model_args, data_args, training_args ):

    training_args.output_dir="{}/{}".format(training_args.output_dir, model_args.model_name_or_path)

    os.makedirs(training_args.output_dir, exist_ok=True)

    cache_dir_path = "./{}/{}".format(model_args.cache_dir, model_args.model_name_or_path)
    os.makedirs(cache_dir_path, exist_ok=True)

    # Initialize Trainer
    trainer = Trainer(
        train_dataset=training_data,
        eval_dataset=None,
        model=model,
        # data_collator=data_collator,
        args=training_args,
        # compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    ###
    # Training Phase
    ###
    print('*** Training Phase ***')
    
    # use last checkpoint if exist
    if os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    # Save the feature_extractor and the tokenizer
    # if is_main_process(training_args.local_rank):
    feature_extractor.save_pretrained(training_args.output_dir)

    metrics = train_result.metrics


    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    ###
    # Evaluation Phase
    ###
    results = {}
    logger.info("*** Evaluation Phase ***")
    # metrics = trainer.evaluate(eval_dataset=vectorized_datasets["valid"])
    # metrics["eval_samples"] = len(vectorized_datasets["valid"])

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    

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



    # Initialize our dataset and prepare it for the emotion classification task.
    data = data_loader.load_dataset(data_args.dataset_path)
    raw_datasets = {}
    for d in data:
        dset = DatasetDict()
        dset["train"] = d["data"][0]
        dset["validation"] = d["data"][1]
        for test_dset_name, test_dset in d["data"][-1].items():
            dset[f'test-{test_dset_name}'] = test_dset
        raw_datasets[f'{d["lang"]}-{d["group"]}'] = dset.copy()

    # for k, v in raw_datasets.items():
    #     print(f'=== {k} ===')
    #     print(v)
    #     print()

    # print(raw_datasets["zho-elderly"]["test-csed"])
    # print()
    # print(raw_datasets["zho-elderly"]["test-csed"][:3])
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)
    sampling_rate = feature_extractor.sampling_rate
    model = AutoModelForAudioClassification.from_pretrained(model_args.model_name_or_path).to(device)
    # processor = AutoProcessor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

    # model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    
    train_fn(model, raw_datasets["zho-elderly"]["test-csed"], feature_extractor, model_args, data_args, training_args )
    
if __name__ == '__main__':
    main()