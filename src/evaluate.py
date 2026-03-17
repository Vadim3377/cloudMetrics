import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)


def choose_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_f1 = -1.0

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = thr

    return best_threshold, best_f1


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def predict_lstm(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def save_metrics(metrics, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_pr_curve(y_true, y_prob, path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_timeline(memory, incident_labels, probs, threshold, split_start_idx, path):
    plt.figure(figsize=(14, 6))

    t = np.arange(len(memory))
    plt.plot(t, memory, label="Memory usage")

    incident_idx = np.where(incident_labels == 1)[0]
    if len(incident_idx) > 0:
        plt.scatter(
            incident_idx,
            memory[incident_idx],
            s=10,
            label="Incident points"
        )

    prob_t = np.arange(split_start_idx, split_start_idx + len(probs))
    scaled_probs = probs * 100.0
    plt.plot(prob_t, scaled_probs, label="Predicted probability x100")

    plt.axhline(90, linestyle="--", label="Incident threshold (90)")
    plt.axhline(threshold * 100, linestyle=":", label="Decision threshold x100")

    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("Memory Usage and Predicted Incident Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()