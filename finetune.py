import torch
from huggingsound import TrainingArguments, ModelArguments
from transformers import AutoProcessor, AutoModelForAudioClassification




def run():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", device=device)
    output_dir = "/checkpoints"
    processor = AutoProcessor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")


    # first of all, you need to define your model's token set
    labels = ['ANG','DIS','FEA','HAP','NEU','SAD' ]  # ager, disgust, fear, happiness, neutral, sadness
    label_set = 

    # the lines below will load the training and model arguments objects, 
    # you can check the source code (huggingsound.trainer.TrainingArguments and huggingsound.trainer.ModelArguments) to see all the available arguments
    training_args = TrainingArguments(
        learning_rate=3e-4,
        max_steps=1000,
        eval_steps=200,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
    )
    model_args = ModelArguments(
        activation_dropout=0.1,
        hidden_dropout=0.1,
    ) 

    # define your train/eval data
    train_data = [
        {"path": "/path/to/sagan.mp3", "transcription": "extraordinary claims require extraordinary evidence"},
        {"path": "/path/to/asimov.wav", "transcription": "violence is the last refuge of the incompetent"},
    ]
    eval_data = [
        {"path": "/path/to/sagan2.mp3", "transcription": "absence of evidence is not evidence of absence"},
        {"path": "/path/to/asimov2.wav", "transcription": "the true delight is in the finding out rather than in the knowing"},
    ]

    # and finally, fine-tune your model
    model.finetune(
        output_dir, 
        train_data=train_data, 
        eval_data=eval_data, # the eval_data is optional
        token_set=token_set, 
        training_args=training_args,
        model_args=model_args,
    )

