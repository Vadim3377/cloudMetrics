import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM-based binary classifier for time-series windows.
    Given a sequence of past memory usage values (window),
    predict whether an incident will occur in the near future.

    Input:
        x shape: (batch_size, window_size, input_size)

        Example:
            batch_size = number of samples
            window_size = W (e.g. 30 time steps)
            input_size = number of features per timestep (here = 1 → memory)

    Output:
        logits shape: (batch_size,)
        → raw scores (before sigmoid)

    """

    def __init__(self, input_size: int = 1, hidden_size: int = 32):
        super().__init__()

        # LSTM layer:
        # Processes the sequence step-by-step and builds a hidden representation
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True  # input shape: (batch, seq, features)
        )

        # Fully connected layer:
        # Maps the final hidden state → single output (binary classification)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass.

        Steps:
        1) Pass the sequence through LSTM
        2) Extract final hidden state (summary of the sequence)
        3) Map to a single prediction (logit)

        We use only the LAST hidden state because it represents
        the accumulated information from the entire sequence.
        """

        # lstm_out is not needed → we only care about final hidden state
        _, (h_n, _) = self.lstm(x)

        # h_n shape: (num_layers, batch_size, hidden_size)
        # We take the last layer's hidden state
        last_hidden = h_n[-1]

        # Convert hidden representation → prediction score
        logits = self.fc(last_hidden)

        # Remove last dimension → shape becomes (batch_size,)
        return logits.squeeze(-1)