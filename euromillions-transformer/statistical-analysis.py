import itertools
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler


# --- Setup ---
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

MAIN_COLS = ["N1", "N2", "N3", "N4", "N5"]
STAR_COLS = ["E1", "E2"]
ALL_NUM_COLS = MAIN_COLS + STAR_COLS
DATE_COL = "DATE"
JACKPOT_COLS = ["JACKPOT", "JACKPOT_EUR", "JACKPOT_AMOUNT", "JACKPOT_AMOUNT_EUR"]
ROLLOVER_COLS = ["ROLLOVER", "ROLLOVERS", "IS_ROLLOVER"]


# --- Data Loading & Preparation (1. Data Preparation) ---
def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads and preprocesses the EuroMillions data.

    - Tries semicolon separator first, then comma as fallback.
    - Parses DATE column to datetime.
    - Drops rows with missing numeric values in N1..N5,E1,E2.
    - Casts numeric columns to int.
    """
    # Try semicolon, fallback to comma
    try:
        df = pd.read_csv(filepath, sep=";")
        if df.shape[1] == 1:
            # It likely wasn't ; separated
            df = pd.read_csv(filepath)
    except Exception:
        df = pd.read_csv(filepath)
    # Normalize column names strip/upper?
    df.columns = [c.strip().upper() for c in df.columns]
    # Ensure expected columns exist
    missing = [c for c in [DATE_COL] + ALL_NUM_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns missing in dataset: {missing}")
    # Parse date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    # Drop invalid dates
    df = df.dropna(subset=[DATE_COL])
    # Drop rows missing numbers
    df = df.dropna(subset=ALL_NUM_COLS).copy()
    # Cast numerics to int
    for col in ALL_NUM_COLS:
        if not is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=ALL_NUM_COLS).copy()
    for col in ALL_NUM_COLS:
        df[col] = df[col].astype(int)
    # Sort by date
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


# --- Helpers ---
def savefig(path: Path, tight: bool = True):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[saved] {path}")


def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def get_optional_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def stack_numbers(df: pd.DataFrame, cols: List[str], pool_name: str) -> pd.DataFrame:
    s = df[cols].stack().reset_index()
    s.columns = ["draw_index", "position", "number"]
    s["pool"] = pool_name
    return s


# --- 2. Basic Summary Statistics ---
def plot_summary_statistics(df: pd.DataFrame):
    print("2) Basic Summary Statistics:")
    summary = df[ALL_NUM_COLS].describe()
    print(summary.to_string())
    plt.figure(figsize=(10, 4))
    sns.heatmap(summary, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Summary Statistics of EuroMillions Numbers")
    savefig(PLOTS_DIR / "02_summary_statistics.png")
    print("Explanation: This table summarizes central tendency and dispersion (count, mean, std, min, quartiles, max) for each number position, helping identify skew and spread.")


# --- 3. Number Frequency Distribution ---
def plot_number_frequency(df: pd.DataFrame):
    print("3) Number Frequency Distribution:")
    # Main pool 1-50
    main_vals = pd.Series(df[MAIN_COLS].values.ravel("K"))
    main_freq = main_vals.value_counts().sort_index()
    print("Top 10 main numbers by frequency:\n", main_freq.sort_values(ascending=False).head(10).to_string())
    plt.figure(figsize=(12, 4))
    main_freq.plot(kind="bar", color="steelblue")
    plt.title("Main Numbers Frequency (1-50)")
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    savefig(PLOTS_DIR / "03_main_frequency.png")

    # Star pool 1-12
    star_vals = pd.Series(df[STAR_COLS].values.ravel("K"))
    star_freq = star_vals.value_counts().sort_index()
    print("Top 5 star numbers by frequency:\n", star_freq.sort_values(ascending=False).head(5).to_string())
    plt.figure(figsize=(8, 4))
    star_freq.plot(kind="bar", color="orange")
    plt.title("Star Numbers Frequency (1-12)")
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    savefig(PLOTS_DIR / "03_star_frequency.png")
    print("Explanation: Bars show how often each number appears, revealing hot (frequent) and cold (rare) numbers in each pool.")


# --- 4. Position-Specific Frequency ---
def plot_position_specific_frequency(df: pd.DataFrame):
    print("4) Position-Specific Frequency:")
    plt.figure(figsize=(12, 6))
    cols = ALL_NUM_COLS
    for i, col in enumerate(cols, 1):
        ax = plt.subplot(3, 3, i)
        vc = df[col].value_counts().sort_index()
        ax.bar(vc.index, vc.values, color="teal")
        ax.set_title(f"Frequency by Position {col}")
    plt.suptitle("Position-Specific Frequency", y=1.02, fontsize=12)
    savefig(PLOTS_DIR / "04_position_specific_frequency.png")
    print("Explanation: Compares distributions per position (N1..N5,E1,E2) to detect positional biases.")


# --- 5. Jackpot Analysis Over Time ---
def plot_jackpot_over_time(df: pd.DataFrame):
    jackpot_col = get_optional_column(df, JACKPOT_COLS)
    if jackpot_col is None:
        print("5) Jackpot Analysis Over Time: skipped (no jackpot column found).")
        return
    series = pd.to_numeric(df[jackpot_col], errors="coerce").dropna()
    if series.empty:
        print("5) Jackpot Analysis Over Time: skipped (jackpot values not numeric).")
        return
    plt.figure(figsize=(12, 4))
    plt.plot(df.loc[series.index, DATE_COL], series, color="purple")
    plt.title(f"Jackpot Over Time ({jackpot_col})")
    plt.xlabel("Date")
    plt.ylabel("Jackpot")
    savefig(PLOTS_DIR / "05_jackpot_over_time.png")
    print(f"5) Jackpot Analysis Over Time: plotted using column '{jackpot_col}'.")
    print("Explanation: Shows jackpot dynamics over time to highlight growth and spikes.")


# --- 6. Number Pair Frequency ---
def plot_number_pair_frequency(df: pd.DataFrame):
    # Count co-occurrence of pairs among main numbers only
    pair_counts: Dict[Tuple[int, int], int] = {}
    for _, row in df[MAIN_COLS].iterrows():
        nums = sorted(row.values.tolist())
        for a, b in itertools.combinations(nums, 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
    if not pair_counts:
        return
    # Build heatmap matrix 1..50
    max_main = 50
    mat = np.zeros((max_main + 1, max_main + 1), dtype=int)
    for (a, b), cnt in pair_counts.items():
        mat[a, b] = cnt
        mat[b, a] = cnt
    plt.figure(figsize=(10, 8))
    sns.heatmap(mat[1:, 1:], cmap="magma", cbar=True)
    plt.title("Co-occurrence Frequency of Main Number Pairs")
    plt.xlabel("Number")
    plt.ylabel("Number")
    savefig(PLOTS_DIR / "06_number_pair_heatmap.png")


# --- 7. Even-Odd Distribution ---
def plot_even_odd_distribution(df: pd.DataFrame):
    print("7) Even-Odd Distribution:")
    main_vals = pd.Series(df[MAIN_COLS].values.ravel("K"))
    counts = pd.Series({"Even": (main_vals % 2 == 0).sum(), "Odd": (main_vals % 2 == 1).sum()})
    print(counts.to_string())
    plt.figure(figsize=(5, 5))
    counts.plot(kind="pie", autopct="%1.1f%%", colors=["#66c2a5", "#fc8d62"])
    plt.title("Even vs Odd (Main Numbers)")
    plt.ylabel("")
    savefig(PLOTS_DIR / "07_even_odd_pie.png")

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["#66c2a5", "#fc8d62"])
    plt.title("Even vs Odd (Main Numbers)")
    plt.ylabel("Count")
    savefig(PLOTS_DIR / "07_even_odd_bar.png")
    print("Explanation: Compares parity composition to detect any preference towards even or odd numbers.")


# --- 8. High-Low Distribution ---
def plot_high_low_distribution(df: pd.DataFrame):
    print("8) High-Low Distribution:")
    main_vals = pd.Series(df[MAIN_COLS].values.ravel("K"))
    counts = pd.Series({"Low(1-25)": (main_vals <= 25).sum(), "High(26-50)": (main_vals >= 26).sum()})
    print(counts.to_string())
    plt.figure(figsize=(5, 5))
    counts.plot(kind="pie", autopct="%1.1f%%", colors=["#8da0cb", "#e78ac3"])
    plt.title("High vs Low (Main Numbers)")
    plt.ylabel("")
    savefig(PLOTS_DIR / "08_high_low_pie.png")
    print("Explanation: Splits the main pool into lower and upper halves to assess balance.")


# --- 9. Number Sum Analysis ---
def plot_number_sum_analysis(df: pd.DataFrame):
    print("9) Number Sum Analysis:")
    sums = df[MAIN_COLS].sum(axis=1)
    print(f"Sum stats: mean={sums.mean():.2f}, std={sums.std():.2f}, min={sums.min()}, max={sums.max()}")
    plt.figure(figsize=(8, 4))
    plt.hist(sums, bins=30, color="steelblue", edgecolor="black")
    plt.title("Distribution of Main Numbers Sum")
    plt.xlabel("Sum")
    plt.ylabel("Frequency")
    savefig(PLOTS_DIR / "09_sum_hist.png")

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=sums, color="steelblue")
    plt.title("Box Plot of Main Numbers Sum")
    plt.xlabel("Sum")
    savefig(PLOTS_DIR / "09_sum_box.png")
    print("Explanation: Evaluates overall magnitude of combinations; extreme sums may indicate unusual draws.")


# --- 10. Number Range Analysis ---
def plot_number_range_analysis(df: pd.DataFrame):
    print("10) Number Range Analysis:")
    ranges = df[MAIN_COLS].max(axis=1) - df[MAIN_COLS].min(axis=1)
    print(f"Range stats: mean={ranges.mean():.2f}, std={ranges.std():.2f}, min={ranges.min()}, max={ranges.max()}")
    plt.figure(figsize=(8, 4))
    plt.hist(ranges, bins=30, color="olive", edgecolor="black")
    plt.title("Distribution of Range (max-min) per Draw")
    plt.xlabel("Range")
    plt.ylabel("Frequency")
    savefig(PLOTS_DIR / "10_range_hist.png")

    plt.figure(figsize=(10, 4))
    plt.plot(df[DATE_COL], ranges, color="olive")
    plt.title("Range Over Time")
    plt.xlabel("Date")
    plt.ylabel("Range")
    savefig(PLOTS_DIR / "10_range_over_time.png")
    print("Explanation: Range captures spread within a draw; higher ranges imply more dispersed selections.")


# --- 11. Consecutive Numbers Analysis ---
def plot_consecutive_numbers(df: pd.DataFrame):
    print("11) Consecutive Numbers Analysis:")
    # Count number of consecutive pairs per draw among main numbers
    consec_counts = []
    for _, row in df[MAIN_COLS].iterrows():
        nums = sorted(row.values.tolist())
        count = sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1)
        consec_counts.append(count)
    consec_counts = pd.Series(consec_counts)
    print("Distribution of consecutive pair counts per draw:\n", consec_counts.value_counts().sort_index().to_string())
    plt.figure(figsize=(8, 4))
    consec_counts.value_counts().sort_index().plot(kind="bar", color="slateblue")
    plt.title("Frequency of Consecutive Pairs per Draw")
    plt.xlabel("Consecutive Pair Count in a Draw")
    plt.ylabel("Number of Draws")
    savefig(PLOTS_DIR / "11_consecutive_pairs.png")
    print("Explanation: Highlights how often adjacent numbers occur, indicating clustering on the number line.")


# --- 12. Time-Based Frequency Trends ---
def plot_time_based_frequency_trends(df: pd.DataFrame):
    print("12) Time-Based Frequency Trends:")
    # Monthly frequency of each main number
    tmp = df[[DATE_COL] + MAIN_COLS].copy()
    tmp["YEAR_MONTH"] = tmp[DATE_COL].dt.to_period("M").dt.to_timestamp()
    freq_records = []
    for _, row in tmp.iterrows():
        ym = row["YEAR_MONTH"]
        nums = row[MAIN_COLS].tolist()
        for n in nums:
            freq_records.append((ym, n))
    freq_df = pd.DataFrame(freq_records, columns=["YM", "NUMBER"])
    freq = freq_df.groupby(["YM", "NUMBER"]).size().reset_index(name="COUNT")
    pivot = freq.pivot(index="YM", columns="NUMBER", values="COUNT").fillna(0)
    # Plot a few lines to avoid extreme clutter (e.g., top 5 frequent numbers overall)
    overall = pd.Series(df[MAIN_COLS].values.ravel("K")).value_counts().head(5).index.tolist()
    print(f"Top 5 overall numbers used for trend lines: {list(map(int, overall))}")
    plt.figure(figsize=(12, 5))
    for n in overall:
        if n in pivot.columns:
            plt.plot(pivot.index, pivot[n], label=f"#{n}")
    plt.title("Monthly Frequency Trends (Top 5 Numbers)")
    plt.xlabel("Month")
    plt.ylabel("Frequency")
    plt.legend()
    savefig(PLOTS_DIR / "12_time_based_trends.png")
    print("Explanation: Tracks how frequently selected numbers occur over months, indicating temporal shifts.")


# --- 13. Seasonal Patterns ---
def plot_seasonal_patterns(df: pd.DataFrame):
    print("13) Seasonal Patterns:")
    records = []
    for _, row in df.iterrows():
        month = row[DATE_COL].month
        for n in row[MAIN_COLS].tolist():
            records.append((month, n))
    d = pd.DataFrame(records, columns=["MONTH", "NUMBER"])
    heat = d.groupby(["MONTH", "NUMBER"]).size().unstack(fill_value=0)
    print("Monthly totals (sum over numbers):\n", heat.sum(axis=1).to_string())
    plt.figure(figsize=(12, 5))
    sns.heatmap(heat, cmap="YlGnBu")
    plt.title("Seasonal Patterns: Frequency by Month and Number (Main)")
    plt.xlabel("Number")
    plt.ylabel("Month")
    savefig(PLOTS_DIR / "13_seasonal_heatmap.png")
    print("Explanation: Heatmap shows seasonality in number appearances by calendar month.")


# --- 14. Number Gaps Analysis ---
def plot_number_gaps(df: pd.DataFrame):
    print("14) Number Gaps Analysis:")
    # Average draws between appearances for each main number
    appearances: Dict[int, List[int]] = {n: [] for n in range(1, 51)}
    for idx, row in df[MAIN_COLS].iterrows():
        for n in set(row.values.tolist()):
            appearances[n].append(idx)
    avg_gaps = {}
    for n, idxs in appearances.items():
        if len(idxs) <= 1:
            avg_gaps[n] = np.nan
        else:
            gaps = np.diff(sorted(idxs))
            avg_gaps[n] = float(np.mean(gaps))
    ser = pd.Series(avg_gaps).sort_index()
    print("Numbers with largest average gaps:\n", ser.sort_values(ascending=False).head(10).to_string())
    plt.figure(figsize=(12, 4))
    ser.plot(kind="bar", color="gray")
    plt.title("Average Number of Draws Between Appearances (Main)")
    plt.xlabel("Number")
    plt.ylabel("Average Gap (draws)")
    savefig(PLOTS_DIR / "14_avg_gaps.png")
    print("Explanation: Highlights numbers that tend to have long or short waits between appearances.")


# --- 15. Hot and Cold Numbers ---
def plot_hot_cold_numbers(df: pd.DataFrame):
    print("15) Hot and Cold Numbers:")
    freq = pd.Series(df[MAIN_COLS].values.ravel("K")).value_counts().sort_values(ascending=False)
    hot = freq.head(10)
    cold = freq.tail(10)
    print("Hot (top 10):\n", hot.to_string())
    print("Cold (bottom 10):\n", cold.to_string())
    plt.figure(figsize=(10, 4))
    plt.bar(hot.index.astype(str), hot.values, color="#d73027")
    plt.title("Top 10 Hot Numbers (Main)")
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    savefig(PLOTS_DIR / "15_hot_numbers.png")

    plt.figure(figsize=(10, 4))
    plt.bar(cold.index.astype(str), cold.values, color="#4575b4")
    plt.title("Top 10 Cold Numbers (Main)")
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    savefig(PLOTS_DIR / "15_cold_numbers.png")
    print("Explanation: Hot numbers appear most frequently historically; cold appear least.")


# --- 16. Star Number Patterns ---
def plot_star_number_patterns(df: pd.DataFrame):
    print("16) Star Number Patterns:")
    plt.figure(figsize=(6, 6))
    plt.scatter(df["E1"], df["E2"], alpha=0.5, color="orange", edgecolor="k")
    plt.title("Star Number Scatter (E1 vs E2)")
    plt.xlabel("E1")
    plt.ylabel("E2")
    savefig(PLOTS_DIR / "16_star_scatter.png")

    # Star pair heatmap
    pair_counts: Dict[Tuple[int, int], int] = {}
    for _, row in df[STAR_COLS].iterrows():
        e1, e2 = sorted(row.values.tolist())
        pair_counts[(e1, e2)] = pair_counts.get((e1, e2), 0) + 1
    size = 12
    mat = np.zeros((size + 1, size + 1), dtype=int)
    for (a, b), cnt in pair_counts.items():
        mat[a, b] = cnt
        mat[b, a] = cnt
    plt.figure(figsize=(6, 5))
    sns.heatmap(mat[1:, 1:], cmap="Reds")
    plt.title("Star Pair Co-occurrence Heatmap")
    plt.xlabel("Star")
    plt.ylabel("Star")
    savefig(PLOTS_DIR / "16_star_pair_heatmap.png")
    print("Explanation: Scatter shows relationship between E1 and E2; heatmap shows frequent star pairs.")


# --- 17. Jackpot Rollover Analysis ---
def plot_jackpot_rollover(df: pd.DataFrame):
    jackpot_col = get_optional_column(df, JACKPOT_COLS)
    rollover_col = get_optional_column(df, ROLLOVER_COLS)
    if jackpot_col is None:
        print("17) Jackpot Rollover Analysis: skipped (no jackpot column found).")
        return
    jp = pd.to_numeric(df[jackpot_col], errors="coerce")
    if jp.isna().all():
        print("17) Jackpot Rollover Analysis: skipped (jackpot values not numeric).")
        return
    plt.figure(figsize=(12, 4))
    plt.plot(df[DATE_COL], jp, label="Jackpot", color="purple")
    if rollover_col and rollover_col in df.columns:
        roll = pd.to_numeric(df[rollover_col], errors="coerce")
        if not roll.isna().all():
            rollover_events = df[roll == 1]
            if not rollover_events.empty:
                plt.scatter(rollover_events[DATE_COL], rollover_events[jackpot_col], color="red", label="Rollover", zorder=3)
    plt.title("Jackpot with Rollover Events")
    plt.xlabel("Date")
    plt.ylabel("Jackpot")
    plt.legend()
    savefig(PLOTS_DIR / "17_jackpot_rollover.png")
    print(f"17) Jackpot Rollover Analysis: plotted jackpot vs time; rollover markers added if '{rollover_col}' present.")
    print("Explanation: Assesses whether rollovers coincide with jackpot jumps or trends.")


# --- 18. Number Clustering (PCA/2D scatter using simple features) ---
def plot_number_clustering(df: pd.DataFrame):
    print("18) Number Clustering:")
    # Simple feature vector: sorted main numbers + stars
    feats = []
    for _, row in df.iterrows():
        nums = sorted(row[MAIN_COLS].tolist())
        stars = sorted(row[STAR_COLS].tolist())
        feats.append(nums + stars)
    X = np.array(feats, dtype=float)
    # PCA 2D via numpy SVD (avoid extra deps)
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = U[:, :2] * S[:2]
    print("Explained variance proxy (singular values):", ", ".join(f"{v:.2f}" for v in S[:2]))
    plt.figure(figsize=(7, 6))
    plt.scatter(comps[:, 0], comps[:, 1], s=10, alpha=0.5)
    plt.title("Draw Clustering (PCA 2D on numbers)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    savefig(PLOTS_DIR / "18_clustering_pca.png")
    print("Explanation: Projects draws into 2D to visualize grouping by similar number patterns.")


# --- 19. Correlation Matrix ---
def plot_correlation_matrix(df: pd.DataFrame):
    print("19) Correlation Matrix:")
    cols = ALL_NUM_COLS.copy()
    jp_col = get_optional_column(df, JACKPOT_COLS)
    if jp_col:
        cols = cols + [jp_col]
    sub = df[cols].copy()
    for c in sub.columns:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    corr = sub.corr(numeric_only=True)
    print(corr.round(2).to_string())
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix (Numbers and Jackpot if present)")
    savefig(PLOTS_DIR / "19_correlation_matrix.png")
    print("Explanation: Heatmap shows linear relationships among positions and jackpot (if available).")


# --- 20. Autocorrelation Analysis ---
def plot_autocorrelation(df: pd.DataFrame):
    print("20) Autocorrelation Analysis:")
    # Use the sum of main numbers as a univariate series
    series = df[MAIN_COLS].sum(axis=1)
    plt.figure(figsize=(10, 4))
    plot_acf(series, ax=plt.gca(), lags=40)
    plt.title("ACF of Main Numbers Sum")
    savefig(PLOTS_DIR / "20_acf.png")

    plt.figure(figsize=(10, 4))
    plot_pacf(series, ax=plt.gca(), lags=40, method="ywm")
    plt.title("PACF of Main Numbers Sum")
    savefig(PLOTS_DIR / "20_pacf.png")
    print("Explanation: ACF/PACF reveal persistence or periodicity in the series of draw sums.")


# --- 21. Moving Averages ---
def plot_moving_averages(df: pd.DataFrame):
    print("21) Moving Averages:")
    # Moving average of number frequencies over time: track count of "top hot number" per month
    sums = df[MAIN_COLS].sum(axis=1)
    s = pd.Series(sums.values, index=df[DATE_COL])
    ma7 = s.rolling(7, min_periods=1).mean()
    print(f"Latest values - Sum: {s.iloc[-1]:.1f}, 7-d MA: {ma7.iloc[-1]:.1f}")
    plt.figure(figsize=(12, 4))
    plt.plot(s.index, s.values, label="Sum", alpha=0.5)
    plt.plot(ma7.index, ma7.values, label="7-d MA", color="red")
    plt.title("Main Sum with 7-d Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Sum")
    plt.legend()
    savefig(PLOTS_DIR / "21_moving_average.png")
    print("Explanation: Moving average smooths short-term noise to highlight trend.")


# --- 22. Distribution Fitting (overlay normal) ---
def plot_distribution_fitting(df: pd.DataFrame):
    print("22) Distribution Fitting:")
    data = df[MAIN_COLS].sum(axis=1).astype(float)
    mu, sigma = data.mean(), data.std(ddof=1)
    print(f"Normal fit parameters: mean={mu:.2f}, std={sigma:.2f}")
    plt.figure(figsize=(8, 4))
    sns.histplot(data, bins=30, stat="density", color="#80b1d3", edgecolor="black", alpha=0.7)
    # Normal curve
    xs = np.linspace(data.min(), data.max(), 200)
    norm_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((xs - mu) ** 2) / (2 * sigma**2 + 1e-9))
    plt.plot(xs, norm_pdf, color="red", label=f"N({mu:.1f},{sigma:.1f}²)")
    plt.title("Distribution Fit: Main Sum vs Normal")
    plt.legend()
    savefig(PLOTS_DIR / "22_distribution_fit.png")
    print("Explanation: Compares empirical distribution of sums to a normal curve.")


# --- 23. Chi-Square Test Visualization (uniform expectation) ---
def plot_chi_square_uniform(df: pd.DataFrame):
    print("23) Chi-Square Test Visualization:")
    # Observed frequencies for main numbers
    main_vals = pd.Series(df[MAIN_COLS].values.ravel("K"))
    obs = main_vals.value_counts().sort_index()
    obs = obs.reindex(range(1, 51), fill_value=0)
    expected = np.full(50, obs.mean())
    chi = ((obs.values - expected) ** 2 / (expected + 1e-9)).sum()
    print(f"Chi-square statistic against uniform expectation (unnormalized): {chi:.1f}")
    plt.figure(figsize=(12, 4))
    plt.bar(range(1, 51), obs.values, color="#a6cee3", label="Observed")
    plt.plot(range(1, 51), expected, color="red", label="Expected (Uniform)")
    plt.title(f"Chi-Square vs Uniform (sum={chi:.1f})")
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.legend()
    savefig(PLOTS_DIR / "23_chi_square_uniform.png")
    print("Explanation: Highlights deviations from a uniform distribution; larger deviations imply non-uniformity.")


# --- 24. Number Recurrence Analysis ---
def plot_number_recurrence(df: pd.DataFrame):
    print("24) Number Recurrence Analysis:")
    # For each number compute time between appearances (in draws), aggregate distribution
    gaps_all = []
    for n in range(1, 51):
        idxs = []
        for i, row in df[MAIN_COLS].iterrows():
            if n in row.values:
                idxs.append(i)
        if len(idxs) > 1:
            gaps = np.diff(idxs)
            gaps_all.extend(gaps.tolist())
    if not gaps_all:
        print("Insufficient recurrence events detected; skipping histogram.")
        return
    arr = np.array(gaps_all)
    print(f"Recurrence gaps stats (draws): mean={arr.mean():.2f}, median={np.median(arr):.2f}")
    plt.figure(figsize=(8, 4))
    plt.hist(gaps_all, bins=30, color="#b2df8a", edgecolor="black")
    plt.title("Distribution of Recurrence Times (Draw Gaps)")
    plt.xlabel("Gaps (draws)")
    plt.ylabel("Count")
    savefig(PLOTS_DIR / "24_recurrence_times.png")
    print("Explanation: Measures how quickly numbers tend to reappear across draws.")


# --- 25. Combination Analysis (Network Graph of frequent co-occurrences) ---
def plot_combination_network(df: pd.DataFrame, min_pair_count: int = 20):
    print("25) Combination Analysis (Network Graph):")
    # Build graph on main numbers; edges if pair count >= threshold
    pair_counts: Dict[Tuple[int, int], int] = {}
    for _, row in df[MAIN_COLS].iterrows():
        nums = sorted(row.values.tolist())
        for a, b in itertools.combinations(nums, 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
    edges = [(a, b, cnt) for (a, b), cnt in pair_counts.items() if cnt >= min_pair_count]
    if not edges:
        # Lower threshold adaptively if nothing meets it
        if pair_counts:
            m = np.percentile(list(pair_counts.values()), 95)
            edges = [(a, b, cnt) for (a, b), cnt in pair_counts.items() if cnt >= m]
        if not edges:
            print("No edges met the frequency threshold; skipping network plot.")
            return
    G = nx.Graph()
    G.add_nodes_from(range(1, 51))
    for a, b, w in edges:
        G.add_edge(a, b, weight=w)
    pos = nx.spring_layout(G, seed=42, k=0.25)
    plt.figure(figsize=(10, 8))
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_color="#1f78b4", node_size=200)
    nx.draw_networkx_edges(G, pos, width=[w / max(weights) * 5 for w in weights], alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white")
    plt.title("Combination Network (Main Number Co-occurrence)")
    plt.axis("off")
    savefig(PLOTS_DIR / "25_combination_network.png")
    print(f"Edges plotted: {len(edges)} (threshold={min_pair_count})")
    print("Explanation: Connects numbers that frequently co-occur in the same draw.")


# --- 26. Interactive Dashboard ---
# Skipped Plotly/Dash per constraints. We provide an "interactive-like" static grid summary.
def plot_static_dashboard(df: pd.DataFrame):
    print("26) Static Dashboard:")
    plt.figure(figsize=(12, 8))
    plt.suptitle("Static Summary Dashboard", fontsize=14, y=0.98)

    # Panel 1: Sum over time
    ax1 = plt.subplot(2, 2, 1)
    sums = df[MAIN_COLS].sum(axis=1)
    ax1.plot(df[DATE_COL], sums, color="tab:blue")
    ax1.set_title("Main Sum Over Time")

    # Panel 2: Range over time
    ax2 = plt.subplot(2, 2, 2)
    ranges = df[MAIN_COLS].max(axis=1) - df[MAIN_COLS].min(axis=1)
    ax2.plot(df[DATE_COL], ranges, color="tab:orange")
    ax2.set_title("Range Over Time")

    # Panel 3: Even vs Odd share
    ax3 = plt.subplot(2, 2, 3)
    main_vals = pd.Series(df[MAIN_COLS].values.ravel("K"))
    counts = pd.Series({"Even": (main_vals % 2 == 0).sum(), "Odd": (main_vals % 2 == 1).sum()})
    ax3.bar(counts.index, counts.values, color=["#66c2a5", "#fc8d62"])
    ax3.set_title("Even vs Odd (Main)")

    # Panel 4: Hot top 5
    ax4 = plt.subplot(2, 2, 4)
    freq = main_vals.value_counts().head(5)
    ax4.bar(freq.index.astype(str), freq.values, color="#d73027")
    ax4.set_title("Top 5 Hot Numbers")

    savefig(PLOTS_DIR / "26_static_dashboard.png")
    print("Explanation: Four-panel overview to quickly communicate core trends and distributions.")


# --- 27. Predictive Modeling Results ---
# Without training here, we visualize a placeholder: correlation of sum with time index
def plot_predictive_results(df: pd.DataFrame):
    print("27) Predictive Modeling Results (Proxy):")
    sums = df[MAIN_COLS].sum(axis=1)
    t = np.arange(len(sums))
    corr = np.corrcoef(t, sums)[0, 1]
    print(f"Correlation between time index and sums: {corr:.3f}")
    plt.figure(figsize=(6, 5))
    plt.scatter(t, sums, s=8, alpha=0.5)
    plt.title("Proxy Predictive Plot: Sum vs Time Index")
    plt.xlabel("Time Index")
    plt.ylabel("Sum")
    savefig(PLOTS_DIR / "27_predictive_proxy.png")
    print("Explanation: Illustrates a simple relationship used as a stand-in for model outputs.")


# --- 28. Anomaly Detection ---
def plot_anomaly_detection(df: pd.DataFrame):
    print("28) Anomaly Detection:")
    # Z-score anomalies on sums
    sums = df[MAIN_COLS].sum(axis=1).astype(float)
    z = (sums - sums.mean()) / (sums.std(ddof=1) + 1e-9)
    anomalies = np.where(np.abs(z) > 3)[0]
    print(f"Anomalies found (|z| > 3): {len(anomalies)}")
    plt.figure(figsize=(12, 4))
    plt.plot(df[DATE_COL], sums, label="Sum", alpha=0.7)
    if anomalies.size > 0:
        plt.scatter(df.loc[anomalies, DATE_COL], sums.iloc[anomalies], color="red", label="Anomaly", zorder=3)
    plt.title("Anomaly Detection on Main Sum (|z| > 3)")
    plt.xlabel("Date")
    plt.ylabel("Sum")
    plt.legend()
    savefig(PLOTS_DIR / "28_anomaly_detection.png")
    print("Explanation: Flags unusual draws that deviate strongly from typical sums.")


# --- 29. Monte Carlo Simulation ---
def plot_monte_carlo(df: pd.DataFrame, trials: int = 5000):
    print("29) Monte Carlo Simulation:")
    # Simulate random draws of 5 distinct numbers from 1..50, sum distribution vs actual
    rng = np.random.default_rng(42)
    sims = []
    for _ in range(trials):
        draw = rng.choice(np.arange(1, 51), size=5, replace=False)
        sims.append(draw.sum())
    sims = np.array(sims)
    actual = df[MAIN_COLS].sum(axis=1).values
    print(f"Simulated mean={sims.mean():.2f}, std={sims.std(ddof=1):.2f}; Actual mean={np.mean(actual):.2f}, std={np.std(actual, ddof=1):.2f}")
    plt.figure(figsize=(10, 5))
    sns.kdeplot(sims, label="Simulated Sums", color="gray")
    sns.kdeplot(actual, label="Actual Sums", color="blue")
    plt.title("Monte Carlo: Actual vs Simulated Sum Distribution")
    plt.xlabel("Sum")
    plt.legend()
    savefig(PLOTS_DIR / "29_monte_carlo.png")
    print("Explanation: Compares actual outcomes to random-draw expectations.")


# --- 30. Comprehensive Summary Report ---
def plot_comprehensive_summary(df: pd.DataFrame):
    print("30) Comprehensive Summary Report:")
    plt.figure(figsize=(14, 10))
    plt.suptitle("Comprehensive Summary", fontsize=14, y=0.98)

    # 1: Main frequency top 10
    ax1 = plt.subplot(2, 2, 1)
    freq = pd.Series(df[MAIN_COLS].values.ravel("K")).value_counts().sort_values(ascending=False)
    top10 = freq.head(10)
    ax1.bar(top10.index.astype(str), top10.values, color="#fb9a99")
    ax1.set_title("Top 10 Main Numbers")

    # 2: Range distribution
    ax2 = plt.subplot(2, 2, 2)
    ranges = df[MAIN_COLS].max(axis=1) - df[MAIN_COLS].min(axis=1)
    ax2.hist(ranges, bins=20, color="#a6cee3", edgecolor="black")
    ax2.set_title("Range Distribution")

    # 3: Sum over time
    ax3 = plt.subplot(2, 2, 3)
    sums = df[MAIN_COLS].sum(axis=1)
    ax3.plot(df[DATE_COL], sums, color="#1f78b4")
    ax3.set_title("Main Sum Over Time")

    # 4: Even vs Odd bar
    ax4 = plt.subplot(2, 2, 4)
    main_vals = pd.Series(df[MAIN_COLS].values.ravel("K"))
    counts = pd.Series({"Even": (main_vals % 2 == 0).sum(), "Odd": (main_vals % 2 == 1).sum()})
    ax4.bar(counts.index, counts.values, color=["#66c2a5", "#fc8d62"])
    ax4.set_title("Even vs Odd (Main)")

    savefig(PLOTS_DIR / "30_comprehensive_summary.png")
    print("Explanation: Multi-panel figure summarizing frequency, spread, trend, and parity.")


def engineer_features_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Focused feature engineering to support train.py:
      - Lags for each number position (lag1..lag3)
      - Recency features (days since last seen for each number in each position)
      - Global recency (days since number last appeared in any position)
      - Rolling frequency features (last k draws frequency for each position)
      - Draw-level features (sum, range, parity counts, high/low counts)
      - Date features (year, month, day, dayofweek)
      - Jackpot scaled (if available)
      - Targets are the next draw's numbers for supervised learning alignment

    Produces a row per draw with engineered features and next-draw targets.
    """
    df = df.copy().reset_index(drop=True)

    # Ensure continuous date and basic draw-level features
    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["day"] = df[DATE_COL].dt.day
    df["dayofweek"] = df[DATE_COL].dt.dayofweek

    # Draw-level engineered features
    df["main_sum"] = df[MAIN_COLS].sum(axis=1)
    df["main_range"] = df[MAIN_COLS].max(axis=1) - df[MAIN_COLS].min(axis=1)
    df["main_even_count"] = df[MAIN_COLS].apply(lambda r: int(sum((r % 2) == 0)), axis=1)
    df["main_high_count"] = df[MAIN_COLS].apply(lambda r: int(sum(r >= 26)), axis=1)
    df["stars_sum"] = df[STAR_COLS].sum(axis=1)
    df["stars_even_count"] = df[STAR_COLS].apply(lambda r: int(sum((r % 2) == 0)), axis=1)

    # Jackpot (optional)
    jp_col = get_optional_column(df, JACKPOT_COLS)
    if jp_col:
        df["jackpot_val"] = pd.to_numeric(df[jp_col], errors="coerce").fillna(method="ffill").fillna(0.0)
    else:
        df["jackpot_val"] = 0.0

    # Lags per position (lag1..lag3)
    for col in ALL_NUM_COLS:
        for L in (1, 2, 3):
            df[f"{col}_lag{L}"] = df[col].shift(L)

    # Rolling frequency per position (window of last k draws)
    # For efficiency, compute boolean occurrence then rolling sum
    for col in ALL_NUM_COLS:
        for k in (10, 25, 50):
            # Rolling frequency that current value equals prior values is not known until the current draw occurs.
            # Alternative: rolling distinct value count baseline for the column
            df[f"{col}_rolling_unique_{k}"] = (
                df[col].rolling(window=k, min_periods=1).apply(lambda x: len(pd.unique(x)), raw=False)
            )

    # Recency features: days since last seen for each number position and globally
    # Build last seen indices and dates for 1..50 main, 1..12 stars
    max_main, max_star = 50, 12
    last_seen_pos: Dict[str, Dict[int, int]] = {c: {} for c in ALL_NUM_COLS}
    last_seen_any: Dict[int, int] = {n: None for n in range(1, max_main + 1)}
    last_seen_star_any: Dict[int, int] = {n: None for n in range(1, max_star + 1)}

    # Initialize output arrays
    for col in ALL_NUM_COLS:
        df[f"{col}_days_since_seen"] = np.nan
    df["main_any_days_since_seen_avg"] = np.nan
    df["star_any_days_since_seen_avg"] = np.nan

    date_values = df[DATE_COL].values
    for i in range(len(df)):
        d_i = date_values[i]
        # per-position recency
        for col in ALL_NUM_COLS:
            val = int(df.at[i, col])
            last_idx = last_seen_pos[col].get(val, None)
            if last_idx is None:
                df.at[i, f"{col}_days_since_seen"] = np.nan
            else:
                delta = (d_i - date_values[last_idx]).astype("timedelta64[D]").astype(int)
                df.at[i, f"{col}_days_since_seen"] = int(delta)
            last_seen_pos[col][val] = i

        # global recency for main numbers across any position
        main_vals = [int(df.at[i, c]) for c in MAIN_COLS]
        deltas = []
        for v in main_vals:
            li = last_seen_any.get(v)
            if li is None:
                continue
            delta = (d_i - date_values[li]).astype("timedelta64[D]").astype(int)
            deltas.append(int(delta))
        df.at[i, "main_any_days_since_seen_avg"] = float(np.mean(deltas)) if deltas else np.nan
        for v in main_vals:
            last_seen_any[v] = i

        # star global
        star_vals = [int(df.at[i, c]) for c in STAR_COLS]
        deltas_s = []
        for v in star_vals:
            li = last_seen_star_any.get(v)
            if li is None:
                continue
            delta = (d_i - date_values[li]).astype("timedelta64[D]").astype(int)
            deltas_s.append(int(delta))
        df.at[i, "star_any_days_since_seen_avg"] = float(np.mean(deltas_s)) if deltas_s else np.nan
        for v in star_vals:
            last_seen_star_any[v] = i

    # Targets as next-draw ground truth for supervised setup
    for i, col in enumerate(ALL_NUM_COLS):
        df[f"target_{col}"] = df[col].shift(-1)

    # Drop initial rows with NaNs introduced by lags/recency
    df_fe = df.copy()
    df_fe = df_fe.dropna().reset_index(drop=True)
    return df_fe


def export_training_dataset(df_fe: pd.DataFrame, out_path: Path = Path("features_for_training.csv")) -> Path:
    """
    Exports a compact CSV with features aligned with next-draw targets.
    """
    # Select features aligned with train.py and useful engineered features
    base_features = [
        "year", "month", "day", "dayofweek",
        "main_sum", "main_range", "main_even_count", "main_high_count",
        "stars_sum", "stars_even_count",
        "jackpot_val",
        "main_any_days_since_seen_avg", "star_any_days_since_seen_avg",
    ]

    lag_features = [f"{c}_lag{L}" for c in ALL_NUM_COLS for L in (1, 2, 3)]
    recency_features = [f"{c}_days_since_seen" for c in ALL_NUM_COLS]
    rolling_features = [f"{c}_rolling_unique_{k}" for c in ALL_NUM_COLS for k in (10, 25, 50)]

    target_cols = [f"target_{c}" for c in ALL_NUM_COLS]

    cols = base_features + lag_features + recency_features + rolling_features + target_cols
    cols = [c for c in cols if c in df_fe.columns]
    out = df_fe[cols].copy()
    out.to_csv(out_path, index=False)
    print(f"[export] training features -> {out_path} ({len(out)} rows, {len(out.columns)} columns)")
    return out_path


def run_all_analyses(df: pd.DataFrame):
    print("1) Data Preparation:")
    print(f"Rows after cleaning: {len(df)}; Date range: {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")
    print("Columns available:", ", ".join(df.columns))
    print("Explanation: Data was parsed, cleaned (missing values dropped), and sorted by date.")

    # Focused model-improving analyses and feature export
    plot_summary_statistics(df)                 # sanity and distribution context
    plot_autocorrelation(df)                    # dependency/seasonality check for sums
    plot_number_recurrence(df)                  # recency signal
    plot_number_gaps(df)                        # long/short memory
    plot_hot_cold_numbers(df)                   # prior frequency signal
    plot_number_sum_analysis(df)                # scale of target sequences
    plot_number_range_analysis(df)              # dispersion indicator
    plot_moving_averages(df)                    # trend component
    plot_correlation_matrix(df)                 # relationship check (incl. jackpot if present)
    plot_number_pair_frequency(df)              # co-occurrence structure (useful for embeddings)
    plot_star_number_patterns(df)               # star-specific relationships

    # Engineer features for training and export a dataset aligned to next draw
    df_fe = engineer_features_for_training(df)
    export_training_dataset(df_fe)

    print("Focused analysis complete. Engineered features exported for model training.")


def main():
    data_file = "euromillions_core_data.txt"
    print(f"Running analyses on: {data_file}")
    df = load_data(data_file)
    run_all_analyses(df)
    print(f"All analyses completed. Plots saved in: {PLOTS_DIR.resolve()}")
    print("Features exported to features_for_training.csv (next-draw-aligned).")
    print("Note: Any jackpot/rollover analyses were conditionally executed based on column presence.")


if __name__ == "__main__":
    main()
