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


# Number of past time steps given to the model
WINDOW_SIZE = 30

# Number of future time steps checked for an incident
HORIZON = 10

# Memory usage threshold above which we consider the system to be in incident state
THRESHOLD = 90.0


def time_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Split data into train, validation, and test parts in chronological order.

    This is important for time-series problems because random shuffling would
    leak future information into training.

    Returns:
        X_train, y_train
        X_val, y_val
        X_test, y_test
        train_end, val_end
    """
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
    """
    Full experiment pipeline.

    Steps:
        1) Generate synthetic memory usage data
        2) Create incident labels
        3) Convert time series into sliding-window samples
        4) Split data into train, validation, and test sets
        5) Train Logistic Regression baseline
        6) Train LSTM model
        7) Tune decision threshold on validation set
        8) Evaluate both models on test set
        9) Save metrics and plots
    """

    # Create output directory for saved figures and metrics
    os.makedirs("results/figures", exist_ok=True)

    # Generate synthetic memory usage series
    memory = generate_memory_series(n_steps=20000, seed=42)

    # Mark time steps where memory is above the critical threshold
    incident_labels = make_incident_labels(memory, threshold=THRESHOLD)

    # Build supervised dataset:
    # X contains past windows, y indicates whether an incident happens soon
    X, y = build_windows(
        memory,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        threshold=THRESHOLD
    )

    print("Dataset shape:", X.shape, y.shape)
    print("Positive ratio:", y.mean())

    # Split data in time order
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        train_end, val_end
    ) = time_split(X, y)


    # Classical models cannot directly consume sequences,
    # so we convert each window into handcrafted features.
    X_train_feat = extract_features(X_train)
    X_val_feat = extract_features(X_val)
    X_test_feat = extract_features(X_test)

    # Standardize features before Logistic Regression
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    lr_model.fit(X_train_feat, y_train)

    # Get predicted probabilities for validation and test sets
    val_prob_lr = lr_model.predict_proba(X_val_feat)[:, 1]
    test_prob_lr = lr_model.predict_proba(X_test_feat)[:, 1]

    # Choose threshold on validation set, then evaluate on test set
    best_thr_lr, _ = choose_best_threshold(y_val, val_prob_lr)
    metrics_lr = compute_metrics(y_test, test_prob_lr, best_thr_lr)


    # LSTM receives raw sequential windows directly.
    # Training loader is batched; validation loader is used for model selection.
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

    # Predict probabilities with trained LSTM
    val_prob_lstm = predict_lstm(model, X_val, device="cpu")
    test_prob_lstm = predict_lstm(model, X_test, device="cpu")

    # Tune threshold on validation set and evaluate on test set
    best_thr_lstm, _ = choose_best_threshold(y_val, val_prob_lstm)
    metrics_lstm = compute_metrics(y_test, test_prob_lstm, best_thr_lstm)

    # Collect all final results
    all_metrics = {
        "logistic_regression": metrics_lr,
        "lstm": metrics_lstm,
    }

    print(all_metrics)

    # Save metrics for later inspection and for README/report use
    save_metrics(all_metrics, "results/metrics.json")

    # Save precision-recall curve for LSTM
    plot_pr_curve(y_test, test_prob_lstm, "results/figures/pr_curve_lstm.png")

    # Align test predictions with the original time series for visualization.
    # Each window starts after WINDOW_SIZE - 1 steps, and test begins after val_end windows.
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