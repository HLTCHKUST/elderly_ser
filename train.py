import os, sys
import logging
import numpy as np
import pandas as pd
import argparse
import torch
import torchaudio
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
    WhisperProcessor,
    WhisperForConditionalGeneration,
    HfArgumentParser,set_seed)
import IPython.display as ipd

from transformers.trainer_utils import get_last_checkpoint
from utils import data_loader
from datasets import load_from_disk, set_caching_enabled

set_caching_enabled(True)


logger = logging.getLogger(__name__)    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_fn(model, train_data, valid_data, feature_extractor, model_args, data_args, training_args ):

    training_args.output_dir="{}/{}".format(training_args.output_dir, model_args.model_name_or_path)

    os.makedirs(training_args.output_dir, exist_ok=True)

    cache_dir_path = "./{}/{}".format(model_args.cache_dir, model_args.model_name_or_path)
    os.makedirs(cache_dir_path, exist_ok=True)

    # Initialize Trainer
    trainer = Trainer(
        train_dataset=train_data,
        eval_dataset=valid_data,
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

    # metrics = train_result.metrics


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

    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)
    

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


    train_dataset=raw_datasets["eng-others"]["train"]
    eval_dataset=raw_datasets["eng-others"]["validation"]
    input_column="audio"
    label_list=['sadness', 'fear', 'angry', 'happiness', 'disgust', 'neutral']
    num_labels = len(label_list)
    pooling_mode = "mean"


    # Config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)}
    )
    setattr(config, 'pooling_mode', pooling_mode)

    # Model Prepare
    if 'wav2vec' in model_args.model_name_or_path:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)
        target_sampling_rate = feature_extractor.sampling_rate
        model = AutoModelForAudioClassification.from_pretrained(model_args.model_name_or_path).to(device)
    else:
        processor = WhisperProcessor.from_pretrained(model_args.model_name_or_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

# # load dummy dataset and read soundfiles
# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# input_features = processor(ds[0]["audio"]["array"], return_tensors="pt").input_features 

# # Generate logits
# logits = model(input_features, decoder_input_ids = torch.tensor([[50258]])).logits 
# # take argmax and decode
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = processor.batch_decode(predicted_ids)
# ['<|en|>']

    def label_to_id(label, label_list):

        if len(label_list) > 0:
            return label_list.index(label) if label in label_list else -1

        return label

    def preprocess_function(raw_datasets):
        speech_list = [item['array'] for item in raw_datasets[input_column]]
        # target_list = [label_to_id(label, label_list) for label in label_list]

        # # target_list = list(map(lambda x: x==1.0, raw_datasets[label_list]))
    
    # # print(eval_dataset['sadness'][0],
    # # eval_dataset['fear'][0],
    # # eval_dataset['angry'][0],
    # # eval_dataset['happiness'][0],
    # # eval_dataset['disgust'][0],
    # # eval_dataset['neutral'][0])
    # list_emo=[eval_dataset['sadness'],
    # eval_dataset['fear'],
    # eval_dataset['angry'],
    # eval_dataset['happiness'],
    # eval_dataset['disgust'],
    # eval_dataset['neutral']]
    # target_list=eval_dataset.index(1.0)


        result = feature_extractor(speech_list, sampling_rate=target_sampling_rate)
        result["labels"] = list(target_list)

        return result




    train_dataset = train_dataset.map(
        preprocess_function,
        batch_size=6,
        batched=True,
        num_proc=4
    )   
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batch_size=6,
        batched=True,
        num_proc=4
    )

    idx = 0
    print(f"Training input_values: {train_dataset[idx]['input_values']}")
    print(f"Training attention_mask: {train_dataset[idx]['attention_mask']}")
    print(f"Training labels: {train_dataset[idx]['labels']} - {train_dataset[idx]['emotion']}")
    
    # processor = AutoProcessor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

    # model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    if not os.path.exists("{}/preprocess_data.arrow".format(model_args.cache_dir_path)):
        vectorized_datasets.save_to_disk("{}/preprocess_data.arrow".format(model_args.cache_dir_path))
    else:
        print('Loading cached dataset...')
        vectorized_datasets = datasets.load_from_disk('{}/preprocess_data.arrow'.format(model_args.cache_dir_path))

    train_fn(model, train_dataset, eval_dataset, feature_extractor, model_args, data_args, training_args )
    
if __name__ == '__main__':
    main()