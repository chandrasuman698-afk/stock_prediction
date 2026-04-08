# Empirical Analysis of ML Models for Stock Price Movement Prediction

## Overview

This project investigates the practical limits of machine learning in financial 
market prediction. Using 6 years of daily stock data, I systematically tested 
multiple approaches — from naive price regression to direction classification 
to big-move detection — and documented what actually works (and what doesn't).

The final model uses an ensemble of Random Forest and Gradient Boosting 
classifiers with 20 engineered technical indicators to predict significant 
price movements (>2%) over 5-day windows.

## Key Findings

**1. Price regression is misleading**  
Models predicting exact next-day prices achieved R² > 0.95, but this is 
deceptive — they essentially learn to copy the previous day's price. 
A naive "tomorrow = today" baseline performs nearly as well.

**2. Daily direction prediction is intractable**  
After extensive feature engineering and model tuning, no model consistently 
beat the "always predict UP" baseline (~54.8%) for 1-day direction prediction. 
This aligns with the weak-form Efficient Market Hypothesis.

**3. Longer horizons are more predictable**  
5-day direction prediction showed marginal improvement over baselines, 
with ensemble models reaching ~58% accuracy. Confidence-filtered predictions 
(only trading on high-confidence signals) pushed this to ~61%.

**4. Big-move detection shows genuine signal**  
The most promising results came from predicting significant drops (>2% in 5 days). 
At a 0.5 confidence threshold, the model achieved 20% precision against a 17.4% 
baseline — a small but real edge that could have practical value.

**5. Feature importance reveals what matters**  
Contrary to popular trading wisdom, simple volatility measures (5-day rolling std) 
and mean-reversion signals (distance from 20-day MA) were far more predictive 
than traditional technical indicators like MA crossovers or volume spikes.

### Feature Importance Rankings

| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | 5-day volatility | 0.1242 | Volatility |
| 2 | Distance from MA20 | 0.1187 | Mean reversion |
| 3 | Volatility ratio | 0.0902 | Volatility |
| 4 | RSI (14-day) | 0.0854 | Momentum |
| 5 | RSI (7-day) | 0.0853 | Momentum |

## Methodology

### Data
- Source: Yahoo Finance (via yfinance API)
- Default: AAPL, Jan 2018 — Jan 2024 (~1500 trading days)
- App supports any publicly traded stock

### Feature Engineering (20 features)
- **Returns**: 1, 2, 3, 5, 10-day percentage changes
- **Trend**: Price position relative to 5/10/20/50-day moving averages
- **Crossovers**: MA5/MA20 and MA10/MA50 crossover signals
- **Volatility**: 5, 10, 20-day rolling standard deviation + ratio
- **Momentum**: RSI at 7 and 14-day periods
- **Volume**: Volume spike detection (>1.5x 10-day average)
- **Other**: Day of week, consecutive up-days, distance from MA20

### Models
- **Random Forest Classifier** (500 trees, max_depth=4, balanced classes)
- **Gradient Boosting Classifier** (300 trees, lr=0.005, max_depth=3)
- **Ensemble**: Equal-weight probability averaging

### Evaluation
- Time-series split: 70% train / 15% validation / 15% test
- No future data leakage — strict temporal ordering
- Primary metric: Precision (when model signals, how often is it right?)
- Compared against baselines: random guess, always-UP, event frequency

### Prediction Targets
| Target | Definition | Baseline Frequency |
|--------|-----------|-------------------|
| Big Up | >2% gain in 5 days | ~32% |
| Big Down | >2% drop in 5 days | ~17% |
| Crash | >5% drop in 5 days | ~4% |

## Interactive Web App

Built with Streamlit. Users can analyze any stock with adjustable parameters.

**Features:**
- Real-time data download for any ticker
- Interactive price chart with crash probability overlay
- RSI and moving average visualization
- Model performance metrics with precision/edge analysis
- Current signal detection based on latest data
- Adjustable confidence and movement thresholds

### Run Locally



## Project Structure


## Limitations and Future Work

### Current Limitations
- Uses only price/volume data (no fundamental or alternative data)
- Limited training data (~1500 samples)
- Single-stock analysis (no cross-asset signals)
- Crash prediction suffers from class imbalance (only ~4% positive cases)

### Potential Extensions
- Incorporate NLP sentiment analysis from financial news
- Add macroeconomic indicators (interest rates, VIX, yield curve)
- Implement LSTM/Transformer architectures for sequence modeling
- Cross-asset correlation features (sector ETFs, market indices)
- Options flow data as predictive signal
- Walk-forward optimization with expanding training windows
- Portfolio-level backtesting with transaction costs

## Tech Stack

- **Python 3.x**
- **scikit-learn** — Random Forest, Gradient Boosting, preprocessing
- **yfinance** — Market data API
- **Streamlit** — Web interface
- **Plotly** — Interactive visualizations
- **pandas/numpy** — Data manipulation

## References

- Fama, E. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work
- Malkiel, B. (2003). The Efficient Market Hypothesis and Its Critics
- Lopez de Prado, M. (2018). Advances in Financial Machine Learning
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System

## About

Built as an independent research project exploring the intersection of 
machine learning and quantitative finance. The goal was not to build a 
profitable trading system, but to rigorously evaluate what ML can and 
cannot do in financial prediction — and to document the findings honestly.

If you have questions or suggestions, feel free to reach out.
