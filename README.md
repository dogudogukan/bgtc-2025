# Bloomberg Global Trading Challenge 2025

Quantitative trading strategy based on Instrumented Principal Component Analysis (IPCA) from Kelly, Pruitt, and Su (2019).

## Overview

This repository implements a dynamic factor model that captures time-varying risk exposures. The core idea from AQR's research is that momentum acts as a proxy for time-varying beta; when stocks trend upward, their systematic risk exposure increases proportionally.

### Model

The IPCA framework models returns as:

```
r_{i,t+1} = β_{i,t}' f_{t+1} + ε
β_{i,t} = Γ' Z_{i,t}
```

Where factor loadings (β) are linear functions of observable characteristics (Z). We use following characteristics by adapting our code:
Momentum candidates included 3-day, 5-day, 1-month, 1-month to 6-month, and 1-month to 12-month
Volatility candidates included 7-day, 10-day, 30-day, 60-day, and 90-day rolling windows
RSI candidates included 9-day, 14-day, and 30-day windows
SMA candidates included 7-day, 10-day, 20-day, 30-day, and 60-day moving averages

## Files

- `ipca_strategy.py` - Main script for model estimation and signal generation
- `requirements.txt` - Python dependencies

## Usage

```bash
pip install -r requirements.txt
python ipca_strategy.py
```

Requires terminal data export in Excel format (multi-level header structure from Bloomberg Terminal or Refinitiv).

## References

Kelly, B., Pruitt, S., Su, Y. (2019). Characteristics are covariances: A unified model of risk and return. Journal of Financial Economics.

Kelly, B., Moskowitz, T., Pruitt, S. (2021). Understanding momentum and reversals. Journal of Financial Economics.
