import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data import generate_memory_series, make_incident_labels, build_windows
from src.features import extract_features
from src.models import LSTMClassifier
from src.train import make_loader, train_lstm
from src.evaluate import (
    choose_best_threshold,
    compute_metrics,
    predict_lstm,
    save_metrics,
    plot_pr_curve,
    plot_timeline,
)


WINDOW_SIZE = 30
HORIZON = 10
THRESHOLD = 90.0


def time_split(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:],
        train_end, val_end
    )


def main():
    os.makedirs("results/figures", exist_ok=True)

    memory = generate_memory_series(n_steps=20000, seed=42)
    incident_labels = make_incident_labels(memory, threshold=THRESHOLD)
    X, y = build_windows(
        memory,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        threshold=THRESHOLD
    )

    print("Dataset shape:", X.shape, y.shape)
    print("Positive ratio:", y.mean())

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        train_end, val_end
    ) = time_split(X, y)

    # -----------------------------
    # Baseline: Logistic Regression
    # -----------------------------
    X_train_feat = extract_features(X_train)
    X_val_feat = extract_features(X_val)
    X_test_feat = extract_features(X_test)

    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    lr_model.fit(X_train_feat, y_train)

    val_prob_lr = lr_model.predict_proba(X_val_feat)[:, 1]
    test_prob_lr = lr_model.predict_proba(X_test_feat)[:, 1]

    best_thr_lr, _ = choose_best_threshold(y_val, val_prob_lr)
    metrics_lr = compute_metrics(y_test, test_prob_lr, best_thr_lr)

    # -----------------------------
    # LSTM
    # -----------------------------
    train_loader = make_loader(X_train, y_train, batch_size=64, shuffle=False)
    val_loader = make_loader(X_val, y_val, batch_size=64, shuffle=False)

    model = LSTMClassifier(input_size=1, hidden_size=32)
    model = train_lstm(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        lr=1e-3,
        device="cpu"
    )

    val_prob_lstm = predict_lstm(model, X_val, device="cpu")
    test_prob_lstm = predict_lstm(model, X_test, device="cpu")

    best_thr_lstm, _ = choose_best_threshold(y_val, val_prob_lstm)
    metrics_lstm = compute_metrics(y_test, test_prob_lstm, best_thr_lstm)

    all_metrics = {
        "logistic_regression": metrics_lr,
        "lstm": metrics_lstm,
    }

    print(all_metrics)
    save_metrics(all_metrics, "results/metrics.json")

    plot_pr_curve(y_test, test_prob_lstm, "results/figures/pr_curve_lstm.png")

    # approximate alignment of test probabilities to original memory timeline
    split_start_idx = (WINDOW_SIZE - 1) + val_end
    plot_timeline(
        memory,
        incident_labels,
        test_prob_lstm,
        best_thr_lstm,
        split_start_idx=split_start_idx,
        path="results/figures/timeline_lstm.png"
    )


if __name__ == "__main__":
    main()