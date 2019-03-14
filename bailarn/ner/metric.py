"""
Custom Metric
"""

from collections import OrderedDict

import numpy as np
from sklearn import metrics
import sklearn.metrics
from . import constant
import pandas as pd

def custom_metric(y_true, y_pred):
    """Calculate score with custom metric"""

    # Find score on each metric
    scores = OrderedDict(sorted({
        "f1_micro": 0.0,
        "f1_macro": 0.0,
        "precision": 0.0,
        "recall": 0.0
    }.items()))

    scores['f1_micro'] = sklearn.metrics.f1_score(
        y_pred=y_pred, y_true=y_true, average='micro')
    scores['f1_macro'] = sklearn.metrics.f1_score(
        y_pred=y_pred, y_true=y_true, average='macro')
    scores["precision"] = sklearn.metrics.precision_score(
        y_pred=y_pred, y_true=y_true, average='macro')
    scores["recall"] = sklearn.metrics.recall_score(
        y_pred=y_pred, y_true=y_true, average='macro')

    return scores