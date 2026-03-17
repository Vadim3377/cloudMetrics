import numpy as np


def generate_memory_series(
    n_steps: int = 20000,
    seed: int = 42
) -> np.ndarray:
    """
    Generate a synthetic time series representing memory usage over time

    The goal is to simulate realistic system behavior so that we can train a model
    to predict future high memory usage.

    We simulate three types of regimes:

    1) Stable (memory fluctuates around a normal level)
    2) Burst (short-term spikes in memory usage)
    3) Leak (eventually leads to critical levels)

    The time series is generated sequentially, switching randomly between regimes
    """

    rng = np.random.default_rng(seed)

    # Initialize memory array
    memory = np.zeros(n_steps, dtype=np.float32)

    # Start from a reasonable baseline
    memory[0] = 50.0

    i = 1
    while i < n_steps:
        # Randomly choose which regime to simulate next
        regime = rng.choice(
            ["stable", "burst", "leak"],
            p=[0.78, 0.15, 0.07]  # mostly stable, rare leaks
        )


        # Stable regime
        if regime == "stable":
            duration = int(rng.integers(20, 120))  # how long this regime lasts
            target = rng.uniform(45, 65)  # equilibrium value

            for _ in range(duration):
                if i >= n_steps:
                    break

                # small noise around target level
                noise = rng.normal(0, 1.5)

                # smooth transition towards target
                memory[i] = 0.92 * memory[i - 1] + 0.08 * target + noise
                i += 1

        # Burst regime
        elif regime == "burst":
            duration = int(rng.integers(5, 20))
            burst_level = rng.uniform(70, 85)

            for _ in range(duration):
                if i >= n_steps:
                    break

                noise = rng.normal(0, 2.0)

                # faster movement towards higher memory usage
                memory[i] = 0.75 * memory[i - 1] + 0.25 * burst_level + noise
                i += 1

        # Leak regime
        else:  # leak
            duration = int(rng.integers(20, 80))

            # positive drift → memory steadily increases
            drift = rng.uniform(0.4, 1.2)

            for _ in range(duration):
                if i >= n_steps:
                    break

                noise = rng.normal(0, 1.0)

                # monotonic upward trend + noise
                memory[i] = memory[i - 1] + drift + noise
                i += 1

    # Ensure values stay within valid percentage range
    memory = np.clip(memory, 0, 100)

    return memory


def make_incident_labels(memory: np.ndarray, threshold: float = 90.0) -> np.ndarray:
    """
    Create pointwise incident labels.

    An incident is defined as memory usage exceeding a critical threshold

    These labels represent actual failures
    They are later used to construct supervised learning targets.

    Returns:
        array of 0/1 values (same length as memory)
    """
    return (memory >= threshold).astype(np.int64)


def build_windows(
    memory: np.ndarray,
    window_size: int,
    horizon: int,
    threshold: float = 90.0
):
    """
    Convert the time series into a supervised learning dataset.

    This is the key step where we transform a time-series problem into
    a classification problem.

    For each time step t:

        Input (X):
            memory[t - window_size + 1 : t + 1]
            => the past W values (what the system just did)

        Target (y):
            1 if any value in future window exceeds threshold:
                max(memory[t+1 : t+horizon+1]) >= threshold
            0 otherwise

    Interpretation:
        y = 1 => "An incident will happen soon"
        y = 0 => "System remains safe"

    This allows the model to learn early warning signals.

    Shapes:
        X: (n_samples, window_size, 1)
        y: (n_samples,)
    """

    X = []
    y = []

    # iterate over all valid positions where both past and future exist
    for t in range(window_size - 1, len(memory) - horizon - 1):

        # Past window (input to model)
        past_window = memory[t - window_size + 1:t + 1]

        # Future window (used only to create label)
        future_window = memory[t + 1:t + horizon + 1]

        # Label: will memory exceed threshold in the near future?
        label = 1 if np.max(future_window) >= threshold else 0

        X.append(past_window.reshape(window_size, 1))
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)