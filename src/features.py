import numpy as np


def extract_features(X: np.ndarray) -> np.ndarray:
    """
    X shape: (n_samples, window_size, 1)
    Returns feature matrix for classical ML models.
    """
    x = X[:, :, 0]

    mean = x.mean(axis=1)
    std = x.std(axis=1)
    min_ = x.min(axis=1)
    max_ = x.max(axis=1)
    last = x[:, -1]
    first = x[:, 0]
    delta = last - first

    slopes = []
    for row in x:
        t = np.arange(len(row))
        slope = np.polyfit(t, row, 1)[0]
        slopes.append(slope)
    slopes = np.array(slopes)

    recent_mean = x[:, -5:].mean(axis=1)
    recent_std = x[:, -5:].std(axis=1)

    return np.column_stack([
        mean, std, min_, max_, last, first, delta, slopes, recent_mean, recent_std
    ])