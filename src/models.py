import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        logits = self.fc(h_n[-1])
        return logits.squeeze(-1)