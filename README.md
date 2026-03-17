# Memory Incident Prediction

## Task description

The goal of this project is to predict whether an incident will occur within the next H time steps based on the previous W time steps of a time-series signal.

An incident is defined as a critical condition in the system. In this project, we model memory usage over time and define an incident as memory exceeding a fixed threshold.

The problem is formulated as a supervised binary classification task using a sliding window approach:

* Input: the previous W time steps of memory usage
* Output: 1 if an incident occurs within the next H steps, otherwise 0

This setup reflects a real-world monitoring scenario where the objective is to detect problems early and trigger alerts before failures occur.

---

## Approach

### Data generation

Since real monitoring data was not used, a synthetic time series was generated to simulate realistic system behavior. The data includes three regimes:

* Stable behavior with small fluctuations around a baseline
* Short bursts representing temporary load spikes
* Memory leak patterns with a gradual increase leading to critical levels

This allows the model to learn meaningful patterns such as trends and sustained growth.

---

### Problem formulation

The time series is converted into a supervised learning dataset using sliding windows:

* Each sample consists of W consecutive past values
* The label is determined by checking whether any value in the next H steps exceeds the incident threshold

This formulation allows the model to learn early warning signals from recent history.

---

## Code structure

### `data.py`

Responsible for data generation and preprocessing.

* `generate_memory_series` creates synthetic memory usage with realistic patterns
* `make_incident_labels` defines incidents based on a threshold
* `build_windows` converts the time series into input-output pairs using the sliding window approach

---

### `features.py`

Extracts fixed-size feature vectors from time windows for classical models.

Features include:

* statistical properties such as mean and standard deviation
* current level (last value)
* trend indicators such as slope and total change
* recent behavior over the last few time steps

These features allow simple models to capture important temporal patterns.

---

### `models.py`

Defines the LSTM model.

* The model processes sequences directly
* The final hidden state is used as a summary of the input window
* A linear layer maps this representation to a prediction

This model is designed to capture temporal dependencies automatically.

---

### `train.py`

Handles model training.

* Uses binary cross-entropy loss with logits
* Optimizes using Adam
* Tracks validation loss and keeps the best model

Training and validation are separated to ensure proper generalization.

---

### `evaluate.py`

Responsible for evaluation and visualization.

* `choose_best_threshold` selects the optimal decision threshold based on validation data
* `compute_metrics` calculates precision, recall, F1, ROC-AUC, and PR-AUC
* `plot_pr_curve` visualizes performance under different thresholds
* `plot_timeline` shows predictions alongside memory usage and true incidents

Special attention is given to threshold selection since this is critical in alerting systems.

---

### `main.py`

Orchestrates the full pipeline:

1. Generate data
2. Build dataset
3. Split into train, validation, and test sets
4. Train logistic regression baseline
5. Train LSTM model
6. Tune thresholds using validation data
7. Evaluate on test set
8. Save metrics and plots

---

## Results

The dataset is imbalanced, with approximately 3 percent of samples representing incidents, which reflects realistic monitoring scenarios.

Two models were evaluated:

### Logistic Regression

* High precision and recall
* Strong F1 score
* Performs well due to effective feature engineering

### LSTM

* Lower performance compared to the baseline
* Still achieves good ROC-AUC
* Less effective at capturing trends in this setup

### Key observation

The logistic regression model outperformed the LSTM. This indicates that the predictive signal is largely captured by simple features such as the current memory level and recent trend.

From the timeline visualization, the model tends to predict incidents close to when they occur, rather than significantly in advance.

---

## Design decisions

* Sliding window formulation was chosen to convert the time-series problem into supervised learning
* Synthetic data was used to control the structure of incidents and ensure reproducibility
* Logistic regression was used as a baseline due to its interpretability and strong performance on structured features
* LSTM was included to explore sequence-based modeling
* Time-based splitting was used to avoid data leakage
* Threshold tuning was performed on validation data to simulate real alerting systems

---

## Limitations

* The dataset is synthetic and may not capture all complexities of real systems
* Only a single metric (memory usage) is used, whereas real systems rely on multiple signals
* Incidents are defined by a simple threshold, which may not reflect real failure conditions
* The model tends to react to high values rather than anticipate long-term trends
* The LSTM model is relatively small and may not fully utilize temporal information

---

## Adaptation to real systems

To apply this approach in a real alerting system, several improvements would be required:

* Use real telemetry data from multiple metrics such as CPU, latency, and error rates
* Handle missing or noisy data
* Introduce more realistic incident definitions based on system behavior
* Continuously retrain or update the model to handle distribution shifts
* Calibrate thresholds based on operational constraints such as acceptable false alert rates
* Implement alert suppression and deduplication mechanisms

The overall framework remains applicable, but would need to be extended to handle production-scale complexity.
