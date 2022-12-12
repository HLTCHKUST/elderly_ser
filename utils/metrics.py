from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
from transformers import EvalPrediction

import collections
import numpy as np


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    preds = np.array(preds).astype('int32')
    label_ids = np.array(p.label_ids).astype('int32')

    classification_report = classification_report(label_ids, preds, output_dict=True)
    metrics = flatten_dict(classification_report)

    return metrics