import datasets
import evaluate
import logging
import numpy as np
import os
import sys
import transformers

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from utils import data_loader
from utils.args_helper import DataTrainingArguments, ModelArguments
from utils.metrics import compute_metrics


logger = logging.getLogger(__name__)


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

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

    print(raw_datasets)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        return_attention_mask=model_args.attention_mask,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    def train_transforms(batch):
        """Apply train_transforms across a batch."""
        output_batch = {"input_values": []}
        for audio in batch[data_args.audio_column_name]:
            wav = random_subsample(
                audio["array"], max_length=data_args.max_length_seconds, sample_rate=feature_extractor.sampling_rate
            )
            output_batch["input_values"].append(wav)
        output_batch["labels"] = [int(label) for label in batch[data_args.label_column_name]]

        return output_batch

    def val_transforms(batch):
        """Apply val_transforms across a batch."""
        output_batch = {"input_values": []}
        for audio in batch[data_args.audio_column_name]:
            wav = audio["array"]
            output_batch["input_values"].append(wav)
        output_batch["labels"] = [int(label) for label in batch[data_args.label_column_name]]

        return output_batch

    # # Prepare label mappings.
    # # We'll include these in the model's config to get human readable labels in the Inference API.
    # labels = preprocessed_datasets["train"].features[data_args.label_column_name].names
    # label2id, id2label = dict(), dict()
    # for i, label in enumerate(labels):
    #     label2id[label] = str(i)
    #     id2label[str(i)] = label

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        # num_labels=len(labels),
        # label2id=label2id,
        # id2label=id2label,
        finetuning_task="audio-classification",
        # cache_dir=model_args.cache_dir,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForAudioClassification.from_pretrained(
        model_args.model_name_or_path,
        # from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        # cache_dir=model_args.cache_dir,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # freeze the convolutional waveform encoder
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    def evaluate_subset(raw_datasets, subset):
        preprocessed_datasets = raw_datasets[subset].copy()

        if data_args.audio_column_name not in preprocessed_datasets["train"].column_names:
            raise ValueError(
                f"--audio_column_name {data_args.audio_column_name} not found in dataset '{data_args.dataset_path}'. "
                "Make sure to set `--audio_column_name` to the correct audio column - one of "
                f"{', '.join(preprocessed_datasets['train'].column_names)}."
            )

        if data_args.label_column_name not in preprocessed_datasets["train"].column_names:
            raise ValueError(
                f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_path}'. "
                "Make sure to set `--label_column_name` to the correct text column - one of "
                f"{', '.join(preprocessed_datasets['train'].column_names)}."
            )

        # Setting `return_attention_mask=True` is the way to get a correctly masked mean-pooling over
        # transformer outputs in the classifier, but it doesn't always lead to better accuracy

        if training_args.do_train:
            if data_args.max_train_samples is not None:
                preprocessed_datasets["train"] = (
                    preprocessed_datasets["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
                )
            # Set the training transforms
            preprocessed_datasets["train"].set_transform(train_transforms, output_all_columns=False)

        if training_args.do_eval:
            if data_args.max_eval_samples is not None:
                preprocessed_datasets["validation"] = (
                    preprocessed_datasets["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
                )
            # Set the validation transforms
            preprocessed_datasets["validation"].set_transform(val_transforms, output_all_columns=False)

        # Initialize our trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=preprocessed_datasets["train"] if training_args.do_train else None,
            eval_dataset=preprocessed_datasets["validation"] if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=feature_extractor,
        )

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            print(preprocessed_datasets)
            print()
            print(preprocessed_datasets["validation"])
            print()
            print(preprocessed_datasets["validation"][0])
            print()
            metrics = trainer.evaluate()
            print(f"=== Validation {subset} ===")
            trainer.log_metrics("validation", metrics)
            trainer.save_metrics("validation", metrics)

        print()

    # evaluate_subset(raw_datasets, subset="eng-others")
    # evaluate_subset(raw_datasets, subset="eng-elderly")
    # evaluate_subset(raw_datasets, subset="zho-others")
    evaluate_subset(raw_datasets, subset="zho-elderly")

if __name__ == "__main__":
    main()