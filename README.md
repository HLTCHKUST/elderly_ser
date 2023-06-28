# Cross-Lingual Cross-Age Group Adaptation for Low-Resource Elderly Speech Emotion Recognition

In this work, we study the prospect of transferring emotion recognition ability over various age groups and languages through the utilization of multilingual pre-trained speech models. In this work, we develop two speech emotion recognition resources, i.e., BiMotion and YueMotion. 
- **BiMotion** is a bi-lingual bi-age-group speech emotion recognition benchmark that covers 6 adult and elderly speech emotion recognition datasets from English and Mandarin Chinese.
- **YueMotion** is a newly-constructed publicly-available Cantonese speech emotion recognition dataset.

## Dataset

YueMotion, which is the Cantonese speech emotion recognition dataset we collect, is available on [HuggingFace](https://huggingface.co/datasets/CAiRE/YueMotion).

## Fine-tuned Checkpoints

- English-Chinese all age: [HuggingFace](https://huggingface.co/CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-all-age)
- English-Chinese elderly: [HuggingFace](https://huggingface.co/CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-elderly)
- English-Chinese adults: [HuggingFace](https://huggingface.co/CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-adults)

### Publication

Our paper will be published at INTERSPEECH 2023. In the meantime, you can find our paper on [arXiv](https://arxiv.org/abs/2306.14517).
If you find our work useful, please consider citing our paper as follows:

```
@misc{cahyawijaya2023crosslingual,
      title={Cross-Lingual Cross-Age Group Adaptation for Low-Resource Elderly Speech Emotion Recognition}, 
      author={Samuel Cahyawijaya and Holy Lovenia and Willy Chung and Rita Frieske and Zihan Liu and Pascale Fung},
      year={2023},
      eprint={2306.14517},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### BiMotion Benchmark

- Crema -> Single label
- ElderReact -> Multi label
- ESD -> Single label
- Chinese-Speech-Emotion-Dataset (CSED) -> Single label
- Tress -> Single label
- IEMOCAP -> Single label
- YueMotion -> Single label
