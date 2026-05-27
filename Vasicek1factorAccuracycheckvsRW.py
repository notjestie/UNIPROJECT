#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from scipy import stats

# =====================
# BACKTEST: VASICEK VS RANDOM WALK
# =====================

MIN_OBS = 100       # minimum observations before first forecast
H_DAYS = 30         # 1 month forecast horizon
DAY_COUNT = 365.25

results = []

for i in range(MIN_OBS, len(ruonia) - H_DAYS):

    train = ruonia.iloc[:i+1].copy()

    r_train = train["r"].values
    x = r_train[:-1]
    y = r_train[1:]

    ols = stats.linregress(x, y)

    alpha = ols.intercept
    beta = ols.slope

    # skip bad Vasicek estimates
    if beta <= 0 or beta >= 1:
        continue

    dt = train["date"].diff().dt.days.median() / DAY_COUNT

    a = -np.log(beta) / dt
    b = alpha / (1 - beta)

    r_t = train["r"].iloc[-1]

    current_date = train["date"].iloc[-1]
    target_date = current_date + pd.Timedelta(days=H_DAYS)

    # realized future rate: first available observation after target date
    future = ruonia[ruonia["date"] >= target_date]

    if future.empty:
        continue

    r_realized = future["r"].iloc[0]
    realized_date = future["date"].iloc[0]

    H = (realized_date - current_date).days / DAY_COUNT

    # Vasicek forecast
    r_vasicek = b + (r_t - b) * np.exp(-a * H)

    # random walk forecast
    r_rw = r_t

    results.append({
        "date": current_date,
        "realized_date": realized_date,
        "r_t": r_t,
        "r_realized": r_realized,
        "r_vasicek": r_vasicek,
        "r_random_walk": r_rw,
        "error_vasicek": r_realized - r_vasicek,
        "error_random_walk": r_realized - r_rw,
        "a": a,
        "b": b,
        "beta": beta
    })

backtest = pd.DataFrame(results)

display(backtest.head())
print("Number of forecasts:", len(backtest))

# =====================
# FORECAST ACCURACY
# =====================

rmse_vasicek = np.sqrt(np.mean(backtest["error_vasicek"] ** 2))
rmse_rw = np.sqrt(np.mean(backtest["error_random_walk"] ** 2))

mae_vasicek = np.mean(np.abs(backtest["error_vasicek"]))
mae_rw = np.mean(np.abs(backtest["error_random_walk"]))

print("RMSE Vasicek:", rmse_vasicek * 100, "%")
print("RMSE Random Walk:", rmse_rw * 100, "%")
print()

print("MAE Vasicek:", mae_vasicek * 100, "%")
print("MAE Random Walk:", mae_rw * 100, "%")

