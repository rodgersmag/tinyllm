#!/usr/bin/env python3
"""
compare.py - Real-world testing: Track predictions against actual draws

This script tests the trained model by generating predictions for each 
historical draw in the training data and comparing against actual results.
It also generates random predictions as a baseline comparison.

Usage:
    python models/transformer/compare.py --data_path features_for_training.csv --model_path models/transformer/lottery_transformer_model_v4.pth
"""

import argparse
import math
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# --- Model Components (must match train.py) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
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
        src = self.input_fc(src).unsqueeze(0)
        src = self.pos_encoder(src)
        
        batch = src.size(1)
        fixed_tokens = torch.ones((batch, 7), dtype=torch.long, device=src.device)
        main_tgt = self.main_number_embedding(fixed_tokens[:, :5])
        star_tgt = self.star_number_embedding(fixed_tokens[:, 5:])
        tgt_embedded = torch.cat([main_tgt, star_tgt], dim=1)
        tgt_embedded = tgt_embedded.permute(1, 0, 2)
        tgt_embedded = self.pos_encoder(tgt_embedded)

        transformer_out = self.transformer(src, tgt_embedded)
        
        main_out = transformer_out[:5, :, :]
        star_out = transformer_out[5:, :, :]
        
        main_numbers_out = self.output_fc_main(main_out)
        star_numbers_out = self.output_fc_star(star_out)
        
        return main_numbers_out, star_numbers_out


# --- Data Loading ---

def load_training_data(data_path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load training data and return dataframe with feature and target columns."""
    df = pd.read_csv(data_path)
    
    # Sort by date-like fields if present
    date_like = [c for c in ["year", "month", "day", "dayofweek"] if c in df.columns]
    if date_like:
        df = df.sort_values(date_like).reset_index(drop=True)
    
    target_cols = ["target_N1", "target_N2", "target_N3", "target_N4", "target_N5", 
                   "target_E1", "target_E2"]
    feature_cols = [c for c in df.columns if c not in target_cols]
    
    # Remove rows with NaN values
    df = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)
    
    return df, feature_cols, target_cols


def load_checkpoint(model_path: str, device: torch.device):
    """Load model checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    required_keys = ["model_state_dict", "scaler", "feature_cols", "hparams"]
    for k in required_keys:
        if k not in checkpoint:
            raise ValueError(f"Checkpoint missing key: {k}")
    return checkpoint


# --- Prediction Functions ---

def predict_for_sample(model, input_tensor: torch.Tensor, device: torch.device) -> Tuple[List[int], List[int]]:
    """Generate predictions for a single input sample."""
    model.eval()
    with torch.no_grad():
        main_logits, star_logits = model(input_tensor, None)
        
        # Get predicted numbers by taking argmax
        main_pred_indices = torch.argmax(main_logits, dim=2) + 1  # +1 for 1-based indexing
        star_pred_indices = torch.argmax(star_logits, dim=2) + 1
        
        # Convert to sorted lists
        main_numbers = sorted(main_pred_indices.squeeze(1).tolist())
        star_numbers = sorted(star_pred_indices.squeeze(1).tolist())
        
        return main_numbers, star_numbers


def generate_random_prediction() -> Tuple[List[int], List[int]]:
    """Generate a completely random prediction for baseline comparison."""
    main_numbers = sorted(random.sample(range(1, 51), 5))
    star_numbers = sorted(random.sample(range(1, 13), 2))
    return main_numbers, star_numbers


# --- Evaluation Functions ---

def count_exact_matches(pred_main: List[int], pred_star: List[int], 
                       actual_main: List[int], actual_star: List[int]) -> Dict[str, int]:
    """Count exact matches between prediction and actual numbers."""
    main_matches = len(set(pred_main) & set(actual_main))
    star_matches = len(set(pred_star) & set(actual_star))
    total_matches = main_matches + star_matches
    
    # Check if this is a jackpot (all numbers match)
    is_jackpot = (main_matches == 5 and star_matches == 2)
    
    return {
        'main_matches': main_matches,
        'star_matches': star_matches,
        'total_matches': total_matches,
        'is_jackpot': int(is_jackpot)
    }


def calculate_prize_tier(matches: Dict[str, int]) -> str:
    """Map match counts to prize tiers based on EuroMillions structure."""
    main = matches['main_matches']
    star = matches['star_matches']
    
    if main == 5 and star == 2:
        return "Jackpot"
    elif main == 5 and star == 1:
        return "Tier 2"
    elif main == 5 and star == 0:
        return "Tier 3"
    elif main == 4 and star == 2:
        return "Tier 4"
    elif main == 4 and star == 1:
        return "Tier 5"
    elif main == 3 and star == 2:
        return "Tier 6"
    elif main == 4 and star == 0:
        return "Tier 7"
    elif main == 2 and star == 2:
        return "Tier 8"
    elif main == 3 and star == 1:
        return "Tier 9"
    elif main == 3 and star == 0:
        return "Tier 10"
    elif main == 1 and star == 2:
        return "Tier 11"
    elif main == 2 and star == 1:
        return "Tier 12"
    elif main == 2 and star == 0:
        return "Tier 13"
    else:
        return "No Prize"


def run_comparison(data_path: str, model_path: str, device: torch.device) -> Dict:
    """Run the main comparison between model predictions and actual results."""
    
    # Load data
    print("Loading training data...")
    df, feature_cols, target_cols = load_training_data(data_path)
    print(f"Loaded {len(df)} training samples")
    
    # Load model
    print("Loading model...")
    checkpoint = load_checkpoint(model_path, device)
    hparams = checkpoint["hparams"]
    scaler = checkpoint["scaler"]
    
    # Initialize model
    model = LotteryTransformer(**hparams).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Scale features
    X_scaled = scaler.transform(df[feature_cols].values.astype(np.float32))
    
    # Initialize results tracking
    model_results = []
    random_results = []
    
    print("Running comparison...")
    
    # Process each training sample
    for idx in range(len(df)):
        # Get actual numbers
        actual_row = df.iloc[idx]
        actual_main = [int(actual_row[f"target_N{i}"]) for i in range(1, 6)]
        actual_star = [int(actual_row[f"target_E{i}"]) for i in range(1, 3)]
        
        # Model prediction
        input_tensor = torch.tensor(X_scaled[idx:idx+1], dtype=torch.float32).to(device)
        pred_main, pred_star = predict_for_sample(model, input_tensor, device)
        
        # Random prediction
        rand_main, rand_star = generate_random_prediction()
        
        # Count matches
        model_matches = count_exact_matches(pred_main, pred_star, actual_main, actual_star)
        random_matches = count_exact_matches(rand_main, rand_star, actual_main, actual_star)
        
        model_results.append(model_matches)
        random_results.append(random_matches)
    
    # Aggregate results
    model_summary = aggregate_results(model_results)
    random_summary = aggregate_results(random_results)
    
    return {
        'model_results': model_summary,
        'random_results': random_summary,
        'total_samples': len(df),
        'sample_results': list(zip(model_results, random_results))
    }


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate match statistics from all results."""
    total_samples = len(results)
    
    # Count jackpots and prize tiers
    jackpots = sum(r['is_jackpot'] for r in results)
    prize_counts = {}
    
    for result in results:
        tier = calculate_prize_tier(result)
        prize_counts[tier] = prize_counts.get(tier, 0) + 1
    
    # Calculate averages
    avg_main = np.mean([r['main_matches'] for r in results])
    avg_star = np.mean([r['star_matches'] for r in results])
    avg_total = np.mean([r['total_matches'] for r in results])
    
    return {
        'total_samples': total_samples,
        'jackpots': jackpots,
        'avg_main_matches': avg_main,
        'avg_star_matches': avg_star,
        'avg_total_matches': avg_total,
        'prize_counts': prize_counts
    }


def print_results(results: Dict):
    """Print comprehensive comparison results."""
    
    print("\n" + "="*60)
    print("LOTTERY PREDICTION COMPARISON RESULTS")
    print("="*60)
    print(f"Total samples analyzed: {results['total_samples']}")
    
    print("\n" + "-"*40)
    print("MODEL PREDICTIONS")
    print("-"*40)
    model = results['model_results']
    print(f"Average matches:")
    print(f"  Main numbers: {model['avg_main_matches']:.2f}/5")
    print(f"  Star numbers: {model['avg_star_matches']:.2f}/2")
    print(f"  Total: {model['avg_total_matches']:.2f}/7")
    print(f"Jackpots predicted: {model['jackpots']}")
    print(f"Jackpot rate: {model['jackpots']/model['total_samples']*100:.6f}%")
    
    print("\nPrize distribution:")
    for tier, count in sorted(model['prize_counts'].items(), key=lambda x: x[1], reverse=True):
        percentage = count/model['total_samples']*100
        print(f"  {tier}: {count} ({percentage:.2f}%)")
    
    print("\n" + "-"*40)
    print("RANDOM PREDICTIONS (BASELINE)")
    print("-"*40)
    random = results['random_results']
    print(f"Average matches:")
    print(f"  Main numbers: {random['avg_main_matches']:.2f}/5")
    print(f"  Star numbers: {random['avg_star_matches']:.2f}/2")
    print(f"  Total: {random['avg_total_matches']:.2f}/7")
    print(f"Jackpots predicted: {random['jackpots']}")
    print(f"Jackpot rate: {random['jackpots']/random['total_samples']*100:.6f}%")
    
    print("\nPrize distribution:")
    for tier, count in sorted(random['prize_counts'].items(), key=lambda x: x[1], reverse=True):
        percentage = count/random['total_samples']*100
        print(f"  {tier}: {count} ({percentage:.2f}%)")
    
    print("\n" + "-"*40)
    print("PERFORMANCE COMPARISON")
    print("-"*40)
    model_jackpot_rate = model['jackpots']/model['total_samples']*100
    random_jackpot_rate = random['jackpots']/random['total_samples']*100
    
    print(f"Model vs Random jackpot rate: {model_jackpot_rate:.6f}% vs {random_jackpot_rate:.6f}%")
    print(f"Model improvement: {model_jackpot_rate - random_jackpot_rate:.6f}%")
    
    print("\n" + "="*60)
    print("DISCLAIMER: These results are for educational purposes only.")
    print("Lottery numbers are randomly generated and past performance")
    print("does not indicate future results.")
    print("="*60)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare model predictions against historical lottery draws")
    parser.add_argument("--data_path", type=str, default="features_for_training.csv")
    parser.add_argument("--model_path", type=str, default="models/transformer/lottery_transformer_model_v4.pth")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Data path: {args.data_path}")
    print(f"Model path: {args.model_path}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run comparison
    results = run_comparison(args.data_path, args.model_path, device)
    
    # Print results
    print_results(results)