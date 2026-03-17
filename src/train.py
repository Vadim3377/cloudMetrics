import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X, y, batch_size=64, shuffle=False):
    """
    Convert numpy arrays into a PyTorch DataLoader.
    DataLoader handles batching and iteration during training.
    shuffle=False is important for time-series data to preserve temporal order
    Input:
        X: (n_samples, window_size, features)
        y: (n_samples,)

    Output:
        DataLoader yielding batches of (X_batch, y_batch)
    """

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)  # float because BCE loss expects floats
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_lstm(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cpu"):
    """
    Train an LSTM model for binary classification.

    Key components:
        - Loss: BCEWithLogitsLoss
            combines sigmoid + binary cross-entropy in a numerically stable way
            Training loss alone is not enough, since the model can overfit.
            Validation loss estimates generalization performance.
        - Optimizer: Adam
            adaptive learning rate, works well for most problems

        - Early model selection:
            we keep the model with the lowest validation loss

    Returns:
        Best-performing model (based on validation loss)
    """

    # Binary classification loss (expects raw logits)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Adam optimizer for parameter updates
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    # Track best model based on validation performance
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):

        # Training phase
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Reset gradients from previous step
            optimizer.zero_grad()

            # Forward pass
            logits = model(X_batch)

            # Compute loss
            loss = criterion(logits, y_batch)

            # Backpropagation (compute gradients)
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Accumulate weighted loss (for averaging later)
            train_loss += loss.item() * len(X_batch)

        # Average training loss over all samples
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0

        # No gradient tracking means faster and less memory
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * len(X_batch)

        # Average validation loss
        val_loss /= len(val_loader.dataset)

        # Print progress
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )


        # Model selection (early stopping logic)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save model weights (deep copy to CPU)
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

    # Restore best model after training
    if best_state is not None:
        model.load_state_dict(best_state)

    return model