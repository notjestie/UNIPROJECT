#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =====================
# RANDOM WALK COMPARISON: 2F VASICEK VS RANDOM WALK
# =====================

import numpy as np
import pandas as pd
from scipy import stats

H = 1 / 12
MAX_REALIZED_GAP_DAYS = 10   # if exact +1m date is not trading day, use nearest later date within 10 days

# Make sure dates are datetime
ofz["DATE"] = pd.to_datetime(ofz["DATE"])
ofz["MATURITY_DATE"] = pd.to_datetime(ofz["MATURITY_DATE"])
factors["DATE"] = pd.to_datetime(factors["DATE"])

# Make sure YTM is decimal
if ofz["YTM"].abs().median() > 1:
    ofz["YTM"] = ofz["YTM"] / 100

# Merge factors into OFZ panel
panel = ofz.merge(
    factors[["DATE", "x", "y", "r"]],
    on="DATE",
    how="inner"
)

panel = panel.sort_values(["DATE", "SECID"]).reset_index(drop=True)

forecast_rows = []

dates = sorted(panel["DATE"].unique())

for t in dates:
    t = pd.Timestamp(t)
    target_date = t + pd.DateOffset(months=1)

    today_curve = panel[panel["DATE"] == t].copy()

    if today_curve.empty:
        continue

    # Current factors
    x_t = today_curve["x"].iloc[0]
    y_t = today_curve["y"].iloc[0]

    # Forecast factors under P
    x_forecast = b1_P + (x_t - b1_P) * np.exp(-a1_P * H)
    y_forecast = b2_P + (y_t - b2_P) * np.exp(-a2_P * H)

    for _, row in today_curve.iterrows():

        secid = row["SECID"]
        maturity = row["MATURITY_DATE"]
        ytm_today = row["YTM"]

        # Remaining maturity at forecast date
        tau_1m = (maturity - target_date).days / DAY_COUNT

        if tau_1m <= 0:
            continue

        # 2-factor Vasicek forecast
        ytm_2f_forecast = two_factor_yield(
            tau=np.array([tau_1m]),
            x=x_forecast,
            y=y_forecast,
            a1=a1_Q,
            a2=a2_Q,
            b1=b1_Q,
            b2=b2_Q,
            sigma1=sigma1_Q,
            sigma2=sigma2_Q,
            rho=rho_Q
        )[0]

        # Random walk forecast
        ytm_rw_forecast = ytm_today

        # Realized YTM of same SECID around target date
        future_rows = panel[
            (panel["SECID"] == secid) &
            (panel["DATE"] >= target_date)
        ].copy()

        if future_rows.empty:
            continue

        realized_row = future_rows.iloc[0]
        realized_date = realized_row["DATE"]

        gap_days = (realized_date - target_date).days

        if gap_days > MAX_REALIZED_GAP_DAYS:
            continue

        ytm_realized = realized_row["YTM"]

        forecast_rows.append({
            "origin_date": t,
            "target_date": target_date,
            "realized_date": realized_date,
            "SECID": secid,
            "MATURITY_DATE": maturity,
            "tau_1m": tau_1m,
            "YTM_today": ytm_today,
            "YTM_realized": ytm_realized,
            "YTM_2F_forecast": ytm_2f_forecast,
            "YTM_RW_forecast": ytm_rw_forecast,
            "error_2F": ytm_realized - ytm_2f_forecast,
            "error_RW": ytm_realized - ytm_rw_forecast
        })

backtest_2f = pd.DataFrame(forecast_rows)

print("Number of backtest observations:", len(backtest_2f))

display(backtest_2f.head())


# In[ ]:


# =====================
# RMSE / MAE COMPARISON
# =====================

rmse_2f = np.sqrt(np.mean(backtest_2f["error_2F"] ** 2))
rmse_rw = np.sqrt(np.mean(backtest_2f["error_RW"] ** 2))

mae_2f = np.mean(np.abs(backtest_2f["error_2F"]))
mae_rw = np.mean(np.abs(backtest_2f["error_RW"]))

print("RMSE Two-Factor Vasicek:", rmse_2f * 100, "%")
print("RMSE Random Walk:", rmse_rw * 100, "%")
print()

print("MAE Two-Factor Vasicek:", mae_2f * 100, "%")
print("MAE Random Walk:", mae_rw * 100, "%")


# In[ ]:





# In[ ]:





# In[ ]:




