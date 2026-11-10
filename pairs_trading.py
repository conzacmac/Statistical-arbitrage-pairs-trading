import yfinance as yf
import pandas as pd
import numpy as np

# Download 3 years of daily data for ING and ABN Amro (Euronext Amsterdam)
tickers = ['INGA.AS', 'ABN.AS']
data = yf.download(
    tickers, start='2023-01-01', end='2026-01-01', auto_adjust=False
)['Adj Close']
data = data.dropna()

import statsmodels.tsa.stattools as ts

# Run the cointegration test
score, p_value, _ = ts.coint(data['INGA.AS'], data['ABN.AS'])
print(f"Cointegration p-value: {p_value}")

# If p-value < 0.05, the stocks are cointegrated!

import statsmodels.api as sm

# Calculate the spread using OLS regression
X = sm.add_constant(data['ABN.AS'])
model = sm.OLS(data['INGA.AS'], X).fit()
spread = data['INGA.AS'] - (model.params['ABN.AS'] * data['ABN.AS'] + model.params['const'])

# Calculate the Z-score (Rolling window of 21 days to avoid look-ahead bias)
rolling_mean = spread.rolling(window=21).mean()
rolling_std = spread.rolling(window=21).std()
z_score = (spread - rolling_mean) / rolling_std

# --- PRINT OUTPUT & STRATEGY SIGNALS ---

# 1. Print the latest date and Z-score
latest_date = z_score.index[-1].strftime('%Y-%m-%d')
latest_z = z_score.iloc[-1]

print("\n" + "=" * 40)
print(f"  PAIRS TRADING SIGNAL ({latest_date})")
print("=" * 40)
print(f"Latest Rolling Z-Score: {latest_z:.4f}\n")

# 2. Evaluate Strategy Logic
if np.isnan(latest_z):
    print("Signal: INSUFFICIENT DATA (Z-score is NaN)")
elif latest_z < -1.5:
    print("Signal: BUY SPREAD (INGA.AS is undervalued relative to ABN.AS)")
elif latest_z > 1.5:
    print("Signal: SHORT SPREAD (INGA.AS is overvalued relative to ABN.AS)")
else:
    print("Signal: NEUTRAL / HOLD (Z-score within normal range)")

# 3. Print recent 5-day trend for context
print("\nRecent 5-Day Z-Score Trend:")
print(z_score.tail())
