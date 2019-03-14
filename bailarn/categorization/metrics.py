from sklearn.metrics import precision_recall_fscore_support
from collections import OrderedDict


def custom_metric(y_true, y_pred):
    """Calculate score with custom metric"""

    # Find score on each metric
    scores = OrderedDict(sorted({
        "precision_macro": 0.0,
        "recall_macro": 0.0,
        "f1_macro": 0.0,
        "f1_micro": 0.0,
    }.items()))

    _, _, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro')

    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro')

    scores["precision_macro"] = precision
    scores["recall_macro"] = recall
    scores["f1_macro"] = f1_macro
    scores["f1_micro"] = f1_micro

    return scores
