from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np


def compute_metrics(
    p: tuple(list[str]), label2id: dict[str, int], id2label: dict[int, str]
) -> dict[str, dict[str, float]]:
    """
    Computes precision, recall f1 and accuracy for the given predictions.
    The metrics are computed both for the overall dataset and for each class separately

    Args:
        p (tuple(list[str], list[str])): predictions of the model and the correct labels
        label2id (dict[str]): mapping of labels to IDs
        id2label (dict[str]): mapping of IDs to labels

    Returns:
        dict[str]: contains the computed metrics
    """
    predictions, labels = p
    predictions = [
        [pred["entity"] for pred in prediction_sentence]
        for prediction_sentence in predictions
    ]

    predictions = [
        [label2id[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    predictions = [label for sentence in predictions for label in sentence]
    labels = [label for sentence in labels for label in sentence]

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    accuracy = accuracy_score(labels, predictions)

    results = {
        "overall": {
            "precision": np.average(precision, weights=support).item(),
            "recall": np.average(precision, weights=support).item(),
            "f1": np.average(precision, weights=support).item(),
            "accuracy": accuracy,
            "support": support.sum().item(),
        }
    }

    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        class_name = id2label[str(i)]
        results[class_name] = {
            "precision": p.item(),
            "recall": r.item(),
            "f1": f.item(),
            "support": s.item(),
        }

    return results
