"""
Bloomberg Global Trading Challenge 2025 - IPCA Trading Strategy

This script implements the Instrumented Principal Component Analysis (IPCA) 
framework from Kelly, Pruitt, and Su (2019) for dynamic factor modeling.

The key insight is that characteristics predict returns because they reveal
time-varying risk exposures: "Characteristics are Covariances"
"""

import pandas as pd
import numpy as np
from ipca import InstrumentedPCA


# =============================================================================
# Configuration
# =============================================================================
DATA_FILE = "2Y_Strategy Model Data.xlsx"  # Bloomberg Terminal export
SHEET_NAME = "CN"                          # China/HK market data
N_FACTORS = 3                              # Number of latent factors
FORWARD_RETURN_DAYS = 5                    # Prediction horizon


# =============================================================================
# Helper Functions
# =============================================================================
def calc_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI measures momentum by comparing the magnitude of recent gains 
    to recent losses.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate technical indicator features for IPCA model.
    
    Features used (following Kelly et al. 2021):
    - mom_1m: 1-month momentum (proxy for time-varying beta)
    - rsi_9: 9-period RSI (overbought/oversold detection)
    - vol_90: 90-day volatility (risk measurement)
    - sma_30: 30-day moving average (trend smoothing)
    """
    df = df.copy()
    
    # Daily log returns
    df["ret_1d"] = np.log(df["close"] / df["close"].shift(1))
    
    # Momentum
    df["mom_1m"] = np.log(df["close"] / df["close"].shift(20))
    
    # RSI
    df["rsi_9"] = calc_rsi(df["close"], 9)
    
    # Volatility (90-day)
    df["vol_90"] = df["ret_1d"].rolling(90).std()
    
    # Moving average
    df["sma_30"] = df["close"].rolling(30).mean()
    
    return df


# =============================================================================
# Data Loading
# =============================================================================
def load_bloomberg_data(filepath: str, sheet_name: str) -> pd.DataFrame:
    """
    Load and parse Bloomberg Terminal data export.
    
    Handles multi-level headers and cleans column names.
    """
    print(f"Loading data from {filepath}...")
    
    # Read Excel with multi-level header
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=[8, 9])
    
    # Flatten multi-level column names
    clean_cols = []
    for a, b in df.columns:
        a = "" if (a is np.nan or not isinstance(a, str)) else a.strip()
        b = "" if (b is np.nan or not isinstance(b, str)) else b.strip()
        
        if b and "Unnamed" not in b:
            clean_cols.append(f"{a} {b}".strip())
        else:
            clean_cols.append(a)
    
    df.columns = clean_cols
    
    # Identify and rename date column
    date_col = [c for c in df.columns if "Updated" in c or "Date" in c][0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Drop unnecessary columns
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def create_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide-format Bloomberg data to long panel format.
    
    Output: DataFrame with columns [date, close, ticker]
    """
    # Extract tickers with Price Close data
    tickers = sorted({c.split()[0] for c in df.columns if "Price Close" in c})
    print(f"Found {len(tickers)} unique tickers")
    
    frames = []
    for t in tickers:
        sub = df[["date", f"{t} Price Close"]].copy()
        sub.columns = ["date", "close"]
        sub["ticker"] = t
        frames.append(sub)
    
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["close"])
    
    print(f"Panel created with {len(panel):,} observations")
    return panel


# =============================================================================
# Feature Engineering
# =============================================================================
def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to panel data.
    
    Steps:
    1. Calculate technical indicators per ticker
    2. Cross-sectional z-score normalization by date
    3. Compute forward returns as target variable
    """
    # Sort for proper time-series operations
    panel = panel.sort_values(["ticker", "date"]).copy()
    
    # Generate features per ticker
    print("Generating features...")
    features = panel.groupby("ticker", group_keys=False).apply(add_features)
    
    # Define feature columns
    original_cols = ["mom_1m", "rsi_9", "vol_90", "sma_30"]
    
    # Cross-sectional z-score normalization
    print("Applying cross-sectional normalization...")
    for col in original_cols:
        features[f"{col}_z"] = features.groupby("date")[col].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )
    
    # Forward returns as target
    features["ret_fwd_5d"] = (
        features.groupby("ticker")["close"]
        .transform(lambda x: np.log(x.shift(-FORWARD_RETURN_DAYS) / x))
    )
    
    # Clean panel
    feature_cols = [f"{c}_z" for c in original_cols]
    final_panel = features.dropna(subset=feature_cols).reset_index(drop=True)
    
    print(f"Final panel: {len(final_panel):,} observations")
    return final_panel, feature_cols


# =============================================================================
# IPCA Model
# =============================================================================
def fit_ipca(panel: pd.DataFrame, feature_cols: list, target_col: str = "ret_fwd_5d"):
    """
    Fit IPCA model to panel data.
    
    IPCA Model:
        r_{i,t+1} = β_{i,t}' f_{t+1} + ε_{i,t+1}
        β_{i,t} = Γ' Z_{i,t}
    
    Where:
        - Γ maps characteristics to factor loadings (constant)
        - Z_{i,t} are observable characteristics (time-varying)
        - f_t are latent factors (estimated)
    """
    # Prepare data
    input_panel = panel.dropna(subset=feature_cols + [target_col]).copy()
    input_panel = input_panel.set_index(["ticker", "date"]).sort_index()
    
    y = input_panel[target_col]
    X = input_panel[feature_cols]
    
    print(f"\nFitting IPCA with {N_FACTORS} factors...")
    print(f"Panel dimensions: N={len(X.index.get_level_values('ticker').unique())}, "
          f"T={len(X.index.get_level_values('date').unique())}, "
          f"L={len(feature_cols)}")
    
    # Fit model
    model = InstrumentedPCA(n_factors=N_FACTORS, intercept=False)
    model.fit(X=X, y=y)
    
    # Get factors and loadings
    Gamma, Factors = model.get_factors(label_ind=True)
    
    # Risk premia (time-series average of factors)
    lambda_hat = np.nanmean(Factors.values, axis=1)
    
    # Model performance
    r2 = model.score(X, y)
    print(f"Model R²: {r2:.4f}")
    
    return model, Gamma, Factors, lambda_hat, input_panel


def generate_signals(input_panel: pd.DataFrame, feature_cols: list, 
                     Gamma: pd.DataFrame, lambda_hat: np.ndarray) -> pd.DataFrame:
    """
    Generate trading signals for the most recent date.
    
    Expected return formula:
        E[r_{i,t+1}] = β_{i,t}' λ = Z_{i,t}' Γ λ
    
    High expected return → BUY signal
    Low expected return → SELL signal
    """
    # Get latest date
    last_date = input_panel.index.get_level_values("date").max()
    print(f"\nGenerating signals for: {last_date.date()}")
    
    # Extract characteristics at last date
    Z_last = input_panel.xs(last_date, level="date")[feature_cols].to_numpy()
    
    # Compute conditional betas: β = Γ' Z
    betas_last = Z_last @ Gamma.values
    
    # Expected returns: μ = β' λ
    mu_hat = betas_last @ lambda_hat
    
    # Create signal DataFrame
    tickers = input_panel.xs(last_date, level="date").index
    signals = pd.DataFrame({
        "ticker": tickers,
        "expected_return": mu_hat
    }).sort_values("expected_return", ascending=False)
    
    return signals


# =============================================================================
# Main Execution
# =============================================================================
def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("IPCA Trading Strategy - Bloomberg Global Trading Challenge")
    print("=" * 60)
    
    # Load data
    df = load_bloomberg_data(DATA_FILE, SHEET_NAME)
    
    # Create panel
    panel = create_panel(df)
    
    # Engineer features
    final_panel, feature_cols = engineer_features(panel)
    
    # Fit IPCA
    model, Gamma, Factors, lambda_hat, input_panel = fit_ipca(
        final_panel, feature_cols
    )
    
    # Generate signals
    signals = generate_signals(input_panel, feature_cols, Gamma, lambda_hat)
    
    # Display results
    print("\n" + "=" * 60)
    print("TOP 10 BUY SIGNALS (Highest Expected Return)")
    print("=" * 60)
    print(signals.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("TOP 10 SELL SIGNALS (Lowest Expected Return)")
    print("=" * 60)
    print(signals.tail(10).to_string(index=False))
    
    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Γ loading strength)")
    print("=" * 60)
    importance = (Gamma ** 2).sum(axis=1).sort_values(ascending=False)
    print(importance.to_frame("importance").to_string())
    
    # Factor risk premia
    print("\n" + "=" * 60)
    print("FACTOR RISK PREMIA (λ)")
    print("=" * 60)
    for i, lam in enumerate(lambda_hat):
        print(f"Factor {i+1}: {lam:.6f}")
    
    return signals, model, Gamma


if __name__ == "__main__":
    signals, model, Gamma = main()
