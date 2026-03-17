import numpy as np


def generate_memory_series(
    n_steps: int = 20000,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic memory usage data in [0, 100].

    Regimes:
    - stable baseline
    - temporary burst
    - memory leak
    """
    rng = np.random.default_rng(seed)

    memory = np.zeros(n_steps, dtype=np.float32)
    memory[0] = 50.0

    i = 1
    while i < n_steps:
        regime = rng.choice(
            ["stable", "burst", "leak"],
            p=[0.78, 0.15, 0.07]
        )

        if regime == "stable":
            duration = int(rng.integers(20, 120))
            target = rng.uniform(45, 65)
            for _ in range(duration):
                if i >= n_steps:
                    break
                noise = rng.normal(0, 1.5)
                memory[i] = 0.92 * memory[i - 1] + 0.08 * target + noise
                i += 1

        elif regime == "burst":
            duration = int(rng.integers(5, 20))
            burst_level = rng.uniform(70, 85)
            for _ in range(duration):
                if i >= n_steps:
                    break
                noise = rng.normal(0, 2.0)
                memory[i] = 0.75 * memory[i - 1] + 0.25 * burst_level + noise
                i += 1

        else:  # leak
            duration = int(rng.integers(20, 80))
            drift = rng.uniform(0.4, 1.2)
            for _ in range(duration):
                if i >= n_steps:
                    break
                noise = rng.normal(0, 1.0)
                memory[i] = memory[i - 1] + drift + noise
                i += 1

    memory = np.clip(memory, 0, 100)
    return memory


def make_incident_labels(memory: np.ndarray, threshold: float = 90.0) -> np.ndarray:
    """Pointwise incident labels."""
    return (memory >= threshold).astype(np.int64)


def build_windows(
    memory: np.ndarray,
    window_size: int,
    horizon: int,
    threshold: float = 90.0
):
    """
    Build sliding windows and future-incident labels.

    X shape: (n_samples, window_size, 1)
    y shape: (n_samples,)
    """
    X = []
    y = []

    for t in range(window_size - 1, len(memory) - horizon - 1):
        past_window = memory[t - window_size + 1:t + 1]
        future_window = memory[t + 1:t + horizon + 1]

        label = 1 if np.max(future_window) >= threshold else 0

        X.append(past_window.reshape(window_size, 1))
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)