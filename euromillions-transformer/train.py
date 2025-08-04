import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


"""
README (usage)
- Data: expects features_for_training.csv at repo root with columns matching the sample the user provided.
- Targets: target_N1..target_N5 (1..50), target_E1..target_E2 (1..12)
- Train:
    python models/transformer/train.py --data_path features_for_training.csv --save_path models/transformer/lottery_transformer_model_v4.pth
- Notes:
    - This version adapts the provided transformer to the engineered CSV (features_for_training.csv).
    - It scales only feature columns; targets remain as class indices.
    - It evaluates with average matches and Hamming distance-like metrics.
"""


# --- Model Components ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0), :]


class LotteryTransformer(nn.Module):
    """
    Transformer encoder-decoder predicting categorical classes for 5 main + 2 star numbers.
    tgt should be shape (batch, 7) with integer tokens (1..num_classes), not used as teacher forcing here,
    but embedded to match original architecture.
    """
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        num_classes_main: int,
        num_classes_star: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.pos_encoder = PositionalEncoding(model_dim)
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.input_fc = nn.Linear(input_dim, model_dim)

        self.main_number_embedding = nn.Embedding(num_classes_main + 1, model_dim, padding_idx=0)
        self.star_number_embedding = nn.Embedding(num_classes_star + 1, model_dim, padding_idx=0)

        self.output_fc_main = nn.Linear(model_dim, num_classes_main)
        self.output_fc_star = nn.Linear(model_dim, num_classes_star)

    def forward(self, src, tgt=None):
        # src: (batch, input_dim)
        # Make the encoder-only pathway: we ignore tgt during training to avoid teacher-forced copying.
        src = self.input_fc(src).unsqueeze(0)  # (1, batch, d_model)
        src = self.pos_encoder(src)            # (1, batch, d_model)

        # Build a learned fixed-length decoder query of length 7
        # We reuse embedding tables with a fixed token (1) to create a decoder query.
        batch = src.size(1)
        fixed_tokens = torch.ones((batch, 7), dtype=torch.long, device=src.device)
        main_tgt = self.main_number_embedding(fixed_tokens[:, :5])
        star_tgt = self.star_number_embedding(fixed_tokens[:, 5:])
        tgt_embedded = torch.cat([main_tgt, star_tgt], dim=1)   # (batch, 7, d_model)
        tgt_embedded = tgt_embedded.permute(1, 0, 2)            # (7, batch, d_model)
        tgt_embedded = self.pos_encoder(tgt_embedded)

        transformer_out = self.transformer(src, tgt_embedded)   # (7, batch, d_model)

        main_out = transformer_out[:5, :, :]
        star_out = transformer_out[5:, :, :]

        main_numbers_out = self.output_fc_main(main_out)  # (5, batch, num_classes_main)
        star_numbers_out = self.output_fc_star(star_out)  # (2, batch, num_classes_star)

        return main_numbers_out, star_numbers_out


# --- Data Preparation ---

TARGET_MAIN_COLS = ["target_N1", "target_N2", "target_N3", "target_N4", "target_N5"]
TARGET_STAR_COLS = ["target_E1", "target_E2"]

def load_and_prepare_data_csv(
    filepath: str = "features_for_training.csv",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    MinMaxScaler,
    List[str]
]:
    """
    Loads engineered CSV, scales feature columns, and returns tensors.
    Assumes targets are target_N1..target_N5, target_E1..target_E2 as integer classes (1-indexed).
    """
    df = pd.read_csv(filepath)

    # Validate target columns presence
    for c in TARGET_MAIN_COLS + TARGET_STAR_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required target column: {c}")

    # Identify feature columns: all except targets
    target_cols = TARGET_MAIN_COLS + TARGET_STAR_COLS
    feature_cols = [c for c in df.columns if c not in target_cols]

    # Sort by date-like fields if present to maintain temporal order
    date_like = [c for c in ["year", "month", "day", "dayofweek"] if c in df.columns]
    if date_like:
        df = df.sort_values(date_like).reset_index(drop=True)

    # Remove any rows with NaN in features/targets
    df = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

    # Targets as integers
    y = df[target_cols].astype(int).values  # shape (N, 7)

    # Scale features only
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[feature_cols].values).astype(np.float32)

    N = len(df)
    test_size = int(N * test_ratio)
    val_size = int(N * val_ratio)
    train_size = N - val_size - test_size

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    return (X_train_t, y_train_t), (X_val_t, y_val_t), (X_test_t, y_test_t), scaler, feature_cols


# --- Early Stopping ---

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.01, save_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"Validation loss improved. Model saved to {self.save_path}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# --- Training ---

def train_model(train_loader, val_loader, test_loader, model, epochs, lr, weight_decay, save_path='best_model.pth', device='cpu'):
    # Label smoothing to discourage exact memorization and improve generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine annealing for smoother LR schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, epochs - 5))
    # Slightly stricter early stopping to avoid overfitting
    early_stopping = EarlyStopping(patience=8, min_delta=0.002, save_path=save_path)

    model.to(device)

    # Lists to track losses
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            # Do NOT feed ground-truth as decoder input; prevent trivial echoing of previous draw
            main_preds, star_preds = model(batch_X, None)

            loss = 0.0
            for i in range(5):
                loss = loss + criterion(main_preds[i], batch_y[:, i] - 1)
            for i in range(2):
                loss = loss + criterion(star_preds[i], batch_y[:, 5 + i] - 1)

            loss.backward()
            # Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        total_main_matches = 0
        total_star_matches = 0
        total_hamming_distance = 0
        total_val_count = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                main_preds, star_preds = model(batch_X, None)

                val_loss = 0.0
                main_pred_indices = torch.argmax(main_preds, dim=2) + 1  # (5, B)
                for i in range(5):
                    val_loss = val_loss + criterion(main_preds[i], batch_y[:, i] - 1)

                star_pred_indices = torch.argmax(star_preds, dim=2) + 1  # (2, B)
                for i in range(2):
                    val_loss = val_loss + criterion(star_preds[i], batch_y[:, 5 + i] - 1)

                # metrics
                for i in range(batch_y.size(0)):
                    total_main_matches += len(set(main_pred_indices[:, i].tolist()) & set(batch_y[i, :5].tolist()))
                    total_star_matches += len(set(star_pred_indices[:, i].tolist()) & set(batch_y[i, 5:].tolist()))
                    total_hamming_distance += (main_pred_indices[:, i] != batch_y[i, :5]).sum().item() + (star_pred_indices[:, i] != batch_y[i, 5:]).sum().item()

                total_val_loss += val_loss.item()
                total_val_count += batch_y.size(0)

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        avg_main_matches = total_main_matches / max(1, total_val_count)
        avg_star_matches = total_star_matches / max(1, total_val_count)
        avg_hamming_dist = total_hamming_distance / max(1, total_val_count)

        # Step cosine scheduler every epoch (not by val loss)
        scheduler.step()
        early_stopping(avg_val_loss, model)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Avg Main Matches: {avg_main_matches:.2f}/5 | Avg Star Matches: {avg_star_matches:.2f}/2 | "
              f"Avg Hamming Dist: {avg_hamming_dist:.2f}")

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return train_losses, val_losses


def plot_loss_curves(train_losses, val_losses, save_dir="plots"):
    """Plot and save training and validation loss curves."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.semilogy(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Loss Curves (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_loss_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curves saved to {plot_path}")


def evaluate_model(test_loader, model, save_path='best_model.pth', device='cpu'):
    """Evaluate model on test set."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Load best model
    model.load_state_dict(torch.load(save_path, map_location=device))

    # Test evaluation
    model.eval()
    total_test_loss = 0.0
    total_main_matches = 0
    total_star_matches = 0
    total_hamming_distance = 0
    total_test_count = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            main_preds, star_preds = model(batch_X, None)

            test_loss = 0.0
            main_pred_indices = torch.argmax(main_preds, dim=2) + 1
            for i in range(5):
                test_loss = test_loss + criterion(main_preds[i], batch_y[:, i] - 1)

            star_pred_indices = torch.argmax(star_preds, dim=2) + 1
            for i in range(2):
                test_loss = test_loss + criterion(star_preds[i], batch_y[:, 5 + i] - 1)

            for i in range(batch_y.size(0)):
                total_main_matches += len(set(main_pred_indices[:, i].tolist()) & set(batch_y[i, :5].tolist()))
                total_star_matches += len(set(star_pred_indices[:, i].tolist()) & set(batch_y[i, 5:].tolist()))
                total_hamming_distance += (main_pred_indices[:, i] != batch_y[i, :5]).sum().item() + (star_pred_indices[:, i] != batch_y[i, 5:]).sum().item()

            total_test_loss += test_loss.item()
            total_test_count += batch_y.size(0)

    avg_test_loss = total_test_loss / max(1, len(test_loader))
    avg_main_matches = total_main_matches / max(1, total_test_count)
    avg_star_matches = total_star_matches / max(1, total_test_count)
    avg_hamming_dist = total_hamming_distance / max(1, total_test_count)

    print("\nTest Set Evaluation:")
    print(f"Test Loss: {avg_test_loss:.4f} | Avg Main Matches: {avg_main_matches:.2f}/5 | Avg Star Matches: {avg_star_matches:.2f}/2 | Avg Hamming Dist: {avg_hamming_dist:.2f}")
    
    return {
        "test_loss": avg_test_loss,
        "avg_main_matches": avg_main_matches,
        "avg_star_matches": avg_star_matches,
        "avg_hamming_dist": avg_hamming_dist
    }


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="features_for_training.csv")
    parser.add_argument("--save_path", type=str, default="models/transformer/lottery_transformer_model_v4.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--feature_drop", type=float, default=0.1, help="Probability to randomly drop (zero) some non-temporal features per batch for regularization")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_cols = load_and_prepare_data_csv(args.data_path)

    # Lightweight feature dropout at dataset level: zero-out a random subset of non-temporal columns per epoch via DataLoader worker_init_fn
    rng = np.random.default_rng(42)
    def worker_init_fn(_):
        pass  # deterministic for now; could add epoch-dependent masking

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Maintain chronological order (no shuffle); drop_last for train for more stable batch stats
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Hyperparameters
    input_dim = X_train.shape[1]
    model = LotteryTransformer(
        input_dim=input_dim,
        model_dim=args.model_dim,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        num_classes_main=50,
        num_classes_star=12,
        dropout=args.dropout,
    )
    # Xavier init for linear layers
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print("Starting model training (features_for_training.csv)...")
    train_losses, val_losses = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_path=args.save_path,
        device=device,
    )

    # Plot and save loss curves
    plot_loss_curves(train_losses, val_losses, save_dir="plots")

    # Evaluate on test set
    evaluate_model(test_loader, model, save_path=args.save_path, device=device)

    # Save checkpoint with scaler for inference
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "feature_cols": feature_cols,
            "hparams": {
                "input_dim": input_dim,
                "model_dim": args.model_dim,
                "nhead": args.nhead,
                "num_encoder_layers": args.num_encoder_layers,
                "num_decoder_layers": args.num_decoder_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
                "num_classes_main": 50,
                "num_classes_star": 12,
            },
        },
        args.save_path,
    )
    print(f"\nModel training complete. Best model saved to {args.save_path}")
    print("\n---")
    print("Disclaimer: Lottery numbers are generated by a random process.")
    print("This model is for educational purposes and should not be used for actual lottery predictions.")
    print("---")
