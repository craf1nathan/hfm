import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Générer 500 barres de données réalistes
dates = [datetime(2025, 3, 1) + timedelta(hours=i) for i in range(500)]
np.random.seed(42)

# Marche aléatoire réaliste
returns = np.random.normal(0.0001, 0.002, 500)
prices = 3000 * np.exp(np.cumsum(returns))

ohlcv = pd.DataFrame({
    'time': dates,
    'open': prices + np.random.uniform(-1, 1, 500),
    'high': prices + np.abs(np.random.normal(0, 2, 500)),
    'low': prices - np.abs(np.random.normal(0, 2, 500)),
    'close': prices,
    'volume': np.random.randint(1000, 5000, 500),
})

ohlcv = ohlcv.round(2)
ohlcv.to_csv('data/XAUUSD_1H_sample.csv', index=False)
print(f"✓ Sample data: {len(ohlcv)} bars")
