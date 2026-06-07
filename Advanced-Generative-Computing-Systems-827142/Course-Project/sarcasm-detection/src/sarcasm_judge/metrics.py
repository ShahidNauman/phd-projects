from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(labels, predictions) -> dict[str, object]:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_non_sarcastic": float(precision[0]),
        "recall_non_sarcastic": float(recall[0]),
        "f1_non_sarcastic": float(f1[0]),
        "precision_sarcastic": float(precision[1]),
        "recall_sarcastic": float(recall[1]),
        "f1_sarcastic": float(f1[1]),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def trainer_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    metrics = compute_classification_metrics(labels, predictions)
    return {key: value for key, value in metrics.items() if isinstance(value, float)}
