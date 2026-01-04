import json
from sklearn.metrics import accuracy_score, precision_score, recall_score


def compute_and_save_metrics(preds, gts, num_ambiguous, num_total, save_path):
    """
    Computes classification metrics and saves them to a JSON file.
    """
    if len(preds) == 0:
        print("Warning: No predictions found to compute metrics.")
        return None

    metrics = {
        "accuracy": float(accuracy_score(gts, preds)),
        "precision": float(precision_score(gts, preds)),
        "recall": float(recall_score(gts, preds)),
        "num_samples": len(preds),
        "num_total_attempts": num_total,
        "num_ambiguous_raw": num_ambiguous,
        "ambiguity_rate": num_ambiguous / max(1, num_total)
    }

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics