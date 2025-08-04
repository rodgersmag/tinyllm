import os
import math
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


"""
README (usage)
- Assumes a checkpoint produced by models/transformer/train.py that includes:
  - model_state_dict, scaler, feature_cols, hparams
- By default, reads the last row from features_for_training.csv to form the lag features for inference,
  then generates multiple candidate draws using temperature sampling and simple constraints.
- Run:
    python models/transformer/inference.py --data_path features_for_training.csv --model_path models/transformer/lottery_transformer_model_v4.pth --samples 10 --temperature 1.2
"""


# --- Model Components (must match train.py) ---

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
        # Match the encoder-only pathway used in training
        src = self.input_fc(src).unsqueeze(0)  # (1, batch, d_model)
        src = self.pos_encoder(src)

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


# --- Inference helpers ---

def load_checkpoint(model_path: str, device: torch.device):
    # PyTorch 2.6 defaults to weights_only=True, which breaks loading objects like sklearn scalers.
    # We trust the locally saved checkpoint from our training script, so explicitly set weights_only=False.
    # Alternatively, one could allowlist the class using torch.serialization.add_safe_globals, but we keep it simple here.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    required_keys = ["model_state_dict", "scaler", "feature_cols", "hparams"]
    for k in required_keys:
        if k not in checkpoint:
            raise ValueError(f"Checkpoint missing key: {k}")
    return checkpoint


def load_last_input_row(
    data_path: str,
    feature_cols: List[str],
    scaler: MinMaxScaler,
) -> torch.Tensor:
    df = pd.read_csv(data_path)
    # sort by date-like if exists
    date_like = [c for c in ["year", "month", "day", "dayofweek"] if c in df.columns]
    if date_like:
        df = df.sort_values(date_like).reset_index(drop=True)
    # take last row
    last_row = df.iloc[[-1]][feature_cols]
    scaled = scaler.transform(last_row.values).astype(np.float32)
    return torch.tensor(scaled, dtype=torch.float32)


def sample_without_repeats_sorted(probs: torch.Tensor, k: int, offset: int = 1) -> List[int]:
    """
    Samples k unique indices from probs (1D) without replacement and returns sorted numbers with 1-based offset.
    """
    probs = torch.softmax(probs, dim=-1)
    # Use top-p sampling alternative: simple multinomial without replacement by iterative masking
    chosen = []
    logits = torch.log(probs + 1e-12).clone()
    for _ in range(k):
        masked = logits.clone()
        if chosen:
            for idx in chosen:
                masked[idx - offset] = -1e9
        idx = torch.multinomial(torch.softmax(masked, dim=-1), 1).item()
        chosen.append(idx + offset)
    return sorted(chosen)


def generate_constrained_draw(
    model: LotteryTransformer,
    input_tensor: torch.Tensor,
    device: torch.device,
    temperature: float = 1.2,
) -> Tuple[List[int], List[int]]:
    """
    Generates a lottery draw using temperature sampling with simple non-duplicate constraints.
    """
    model.eval()
    with torch.no_grad():
        # Encoder-only forward; model ignores tgt
        main_logits, star_logits = model(input_tensor, None)  # shapes: (5, 1, 50), (2, 1, 12)

        # Temperature
        main_logits = main_logits.squeeze(1) / max(1e-6, temperature)  # (5, 50)
        star_logits = star_logits.squeeze(1) / max(1e-6, temperature)  # (2, 12)

        # Aggregate across positions by averaging logits to get overall distribution per category
        main_agg = main_logits.mean(dim=0)  # (50,)
        star_agg = star_logits.mean(dim=0)  # (12,)
        # Penalize previously drawn numbers from last input row (if present among lags) to avoid echoing low/min numbers
        # We attempt to read lag columns from the last row to downweight them.
        try:
            df = pd.read_csv(args.data_path)
            date_like = [c for c in ["year", "month", "day", "dayofweek"] if c in df.columns]
            if date_like:
                df = df.sort_values(date_like).reset_index(drop=True)
            last = df.iloc[-1]
            prev_mains = []
            prev_stars = []
            for c in ["N1_lag1","N2_lag1","N3_lag1","N4_lag1","N5_lag1"]:
                if c in df.columns:
                    prev_mains.append(int(last[c]))
            for c in ["E1_lag1","E2_lag1"]:
                if c in df.columns:
                    prev_stars.append(int(last[c]))
            if prev_mains:
                for n in prev_mains:
                    if 1 <= n <= 50:
                        main_agg[n-1] -= 1.0  # subtract logit to discourage repeats
            if prev_stars:
                for n in prev_stars:
                    if 1 <= n <= 12:
                        star_agg[n-1] -= 1.0
        except Exception:
            pass

        main_nums = sample_without_repeats_sorted(main_agg, k=5, offset=1)
        star_nums = sample_without_repeats_sorted(star_agg, k=2, offset=1)

    return main_nums, star_nums


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="features_for_training.csv")
    parser.add_argument("--model_path", type=str, default="models/transformer/lottery_transformer_model_v4.pth")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint and rebuild model with same hparams
    checkpoint = load_checkpoint(args.model_path, device)
    h = checkpoint["hparams"]
    model = LotteryTransformer(
        input_dim=h["input_dim"],
        model_dim=h["model_dim"],
        nhead=h["nhead"],
        num_encoder_layers=h["num_encoder_layers"],
        num_decoder_layers=h["num_decoder_layers"],
        dim_feedforward=h["dim_feedforward"],
        num_classes_main=h["num_classes_main"],
        num_classes_star=h["num_classes_star"],
        dropout=h["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    scaler: MinMaxScaler = checkpoint["scaler"]
    feature_cols: List[str] = checkpoint["feature_cols"]

    print(f"Model loaded successfully from: {args.model_path}")
    print(f"Using features: {feature_cols}")

    # Prepare single input tensor from last row
    input_tensor = load_last_input_row(args.data_path, feature_cols, scaler).to(device)

    print("\nGenerating predictions...")
    print("-" * 50)
    for i in range(args.samples):
        # Small temperature sweep to diversify outputs across iterations
        temp = args.temperature * (1.0 + 0.05 * i)
        main_nums, star_nums = generate_constrained_draw(
            model=model,
            input_tensor=input_tensor,
            device=device,
            temperature=temp,
        )
        main_str = " ".join(f"{n:02d}" for n in main_nums)
        star_str = " ".join(f"{n:02d}" for n in star_nums)
        print(f"Prediction {i+1:2}: Main Numbers: {main_str} | Stars: {star_str}")

    print("-" * 50)
    print("\nDisclaimer: These predictions are for educational and entertainment purposes only.")
    print("Adjust the 'temperature' parameter for more or less randomness.")
