from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
from transformers import EvalPrediction

import collections
import numpy as np


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    preds = np.array(preds).astype('int32')
    if type(p.label_ids) == tuple:
        p.label_ids = p.label_ids[0]
    label_ids = np.array(p.label_ids).astype('int32')
    
    print('preds')
    print(preds)
    print('p.label_ids')
    print(p.label_ids)

    report = classification_report(label_ids, preds, output_dict=True)
    metrics = flatten(report)

    return metrics

    # # Load the accuracy metric from the datasets package
    # metric = evaluate.load("accuracy")

    # # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    # def compute_metrics(eval_pred):
    #     """Computes accuracy on a batch of predictions"""
    #     predictions = np.argmax(eval_pred.predictions, axis=1)
    #     return metric.compute(predictions=predictions, references=eval_pred.label_ids)