import numpy as np


def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Convert raw time-series windows into fixed-size feature vectors
    for classical machine learning models, since classical models cannot directly process sequences.
    We therefore summarize each window using statistics.

    Input:
        X shape: (n_samples, window_size, 1)

    Output:
        Feature matrix of shape (n_samples, n_features)

    """

    # Remove last dimension → shape becomes (n_samples, window_size)
    x = X[:, :, 0]

    # Basic statistics
    # Average memory usage over the window
    mean = x.mean(axis=1)

    # Standard deviation
    std = x.std(axis=1)

    # Minimum and maximum values
    min_ = x.min(axis=1)
    max_ = x.max(axis=1)


    # Current state features

    # Last value in the window, the strongest predictor of near-future incidents
    last = x[:, -1]

    # First value in the window, used to estimate overall change
    first = x[:, 0]

    # Total change across the window, where positive = increasing trend (possible leak)
    delta = last - first

    # Trend estimation

    # Estimate slope using linear regression for each window, which
    # captures whether memory is steadily increasing or decreasing
    slopes = []
    for row in x:
        t = np.arange(len(row))
        slope = np.polyfit(t, row, 1)[0]  # coefficient of linear trend
        slopes.append(slope)
    slopes = np.array(slopes)


    # Recent behavior (short-term)

    # Mean over last few steps (focuses on most recent activity)
    recent_mean = x[:, -5:].mean(axis=1)

    # Recent variability
    # detects instability just before potential incident
    recent_std = x[:, -5:].std(axis=1)


    # Combine all features

    return np.column_stack([
        mean,
        std,
        min_,
        max_,
        last,
        first,
        delta,
        slopes,
        recent_mean,
        recent_std
    ])