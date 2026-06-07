from sarcasm_judge.metrics import compute_classification_metrics


def test_metrics_include_macro_f1_and_confusion_matrix():
    metrics = compute_classification_metrics([0, 1, 1, 0], [0, 1, 0, 0])

    assert "macro_f1" in metrics
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]

