#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy import stats

# =====================
# INPUT
# =====================

OFZ_FILE = "ofz_zcyc_ytmNEW_2024-01-01_to_2026-05-20.csv"

DAY_COUNT = 365.25
FORECAST_MONTHS = 1


# =====================
# 1. LOAD OFZ YTM PANEL
# =====================

ofz = pd.read_csv(OFZ_FILE, encoding="utf-8-sig", skipinitialspace=True)
ofz.columns = ofz.columns.str.strip()

ofz["DATE"] = pd.to_datetime(ofz["DATE"])
ofz["MATURITY_DATE"] = pd.to_datetime(ofz["MATURITY_DATE"])
ofz["YTM"] = pd.to_numeric(ofz["YTM"], errors="coerce")

# percent to decimal
if ofz["YTM"].dropna().abs().median() > 1:
    ofz["YTM"] = ofz["YTM"] / 100

ofz = ofz.dropna(subset=["DATE", "MATURITY_DATE", "YTM"]).copy()

ofz["tau"] = (ofz["MATURITY_DATE"] - ofz["DATE"]).dt.days / DAY_COUNT
ofz = ofz[ofz["tau"] > 0].copy()

ofz = ofz.sort_values(["DATE", "tau"]).reset_index(drop=True)

print("Dates:", ofz["DATE"].min().date(), "to", ofz["DATE"].max().date())
print("Number of curve dates:", ofz["DATE"].nunique())
print("Rows:", len(ofz))

display(ofz.head())


# In[2]:


# =====================
# 2. NELSON-SIEGEL LOADINGS
# =====================

def ns_loadings(tau, lam):
    tau = np.asarray(tau, dtype=float)

    L1 = (1 - np.exp(-lam * tau)) / (lam * tau)
    L2 = L1 - np.exp(-lam * tau)

    X = np.column_stack([
        np.ones_like(tau),  # beta0: level
        L1,                 # beta1: slope
        L2                  # beta2: curvature
    ])

    return X


def fit_ns_for_date(curve, lam):
    tau = curve["tau"].values
    y = curve["YTM"].values

    X = ns_loadings(tau, lam)

    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    fitted = X @ beta
    sse = np.sum((y - fitted) ** 2)

    return beta, fitted, sse


# In[3]:


# =====================
# 3. CHOOSE GLOBAL LAMBDA
# =====================
# lambda controls where the curvature loading peaks.
# We estimate one fixed lambda for the full sample.

def total_sse_for_lambda(lam):
    if lam <= 0:
        return 1e99

    total_sse = 0.0

    for date, curve in ofz.groupby("DATE"):
        if len(curve) < 3:
            continue

        beta, fitted, sse = fit_ns_for_date(curve, lam)
        total_sse += sse

    return total_sse


res_lam = minimize_scalar(
    total_sse_for_lambda,
    bounds=(0.01, 5.0),
    method="bounded"
)

lam = res_lam.x

print("Fitted Nelson-Siegel lambda:", lam)
print("Total SSE:", res_lam.fun)


# In[4]:


# =====================
# 4. FIT NS FACTORS FOR EACH DATE
# =====================

factor_rows = []
fitted_rows = []

for date, curve in ofz.groupby("DATE"):

    if len(curve) < 3:
        continue

    beta, fitted, sse = fit_ns_for_date(curve, lam)

    factor_rows.append({
        "DATE": date,
        "beta0_level": beta[0],
        "beta1_slope": beta[1],
        "beta2_curvature": beta[2],
        "sse": sse,
        "n_bonds": len(curve)
    })

    tmp = curve.copy()
    tmp["NS_fit"] = fitted
    fitted_rows.append(tmp)

factors = pd.DataFrame(factor_rows).sort_values("DATE").reset_index(drop=True)
ofz_fitted = pd.concat(fitted_rows, ignore_index=True)

print("Estimated DNS factors:")
display(factors.head())
display(factors.tail())


# In[5]:


# =====================
# 5. FIT DIRECT 1-MONTH AR(1) FOR EACH FACTOR
# =====================
# For every date t, target is the first available factor date >= t + 1 month.
# This estimates directly:
# beta_{t+1m} = c + phi * beta_t + error

def make_1m_factor_pairs(factors, col):
    df = factors[["DATE", col]].dropna().sort_values("DATE").reset_index(drop=True)

    rows = []

    for i in range(len(df)):
        date_t = df.loc[i, "DATE"]
        target_date = date_t + pd.DateOffset(months=FORECAST_MONTHS)

        future = df[df["DATE"] >= target_date]

        if future.empty:
            continue

        j = future.index[0]

        rows.append({
            "DATE_t": date_t,
            "DATE_target": df.loc[j, "DATE"],
            "x": df.loc[i, col],
            "y": df.loc[j, col]
        })

    return pd.DataFrame(rows)


def fit_ar1_direct_1m(factors, col):
    pairs = make_1m_factor_pairs(factors, col)

    ols = stats.linregress(pairs["x"], pairs["y"])

    c = ols.intercept
    phi = ols.slope

    return {
        "col": col,
        "c": c,
        "phi": phi,
        "r2": ols.rvalue ** 2,
        "pairs": pairs
    }


ar0 = fit_ar1_direct_1m(factors, "beta0_level")
ar1 = fit_ar1_direct_1m(factors, "beta1_slope")
ar2 = fit_ar1_direct_1m(factors, "beta2_curvature")

for ar in [ar0, ar1, ar2]:
    print(ar["col"])
    print("c   =", ar["c"])
    print("phi =", ar["phi"])
    print("R2  =", ar["r2"])
    print()


# In[6]:


# =====================
# 6. FORECAST DNS FACTORS 1 MONTH AHEAD
# =====================

current_date = factors["DATE"].max()
forecast_date = current_date + pd.DateOffset(months=FORECAST_MONTHS)

last = factors[factors["DATE"] == current_date].iloc[0]

beta0_t = last["beta0_level"]
beta1_t = last["beta1_slope"]
beta2_t = last["beta2_curvature"]

beta0_forecast = ar0["c"] + ar0["phi"] * beta0_t
beta1_forecast = ar1["c"] + ar1["phi"] * beta1_t
beta2_forecast = ar2["c"] + ar2["phi"] * beta2_t

print("Current date:", current_date.date())
print("Forecast date:", forecast_date.date())
print()

print("Current factors:")
print("beta0 level     =", beta0_t * 100, "%")
print("beta1 slope     =", beta1_t * 100, "%")
print("beta2 curvature =", beta2_t * 100, "%")
print()

print("Forecasted factors:")
print("beta0 level     =", beta0_forecast * 100, "%")
print("beta1 slope     =", beta1_forecast * 100, "%")
print("beta2 curvature =", beta2_forecast * 100, "%")


# In[7]:


# =====================
# 7. BUILD CURRENT FIT AND FORECASTED CURVE
# =====================

current_curve = ofz_fitted[ofz_fitted["DATE"] == current_date].copy()
current_curve = current_curve.sort_values("tau").reset_index(drop=True)

current_curve["tau_1m"] = (
    current_curve["MATURITY_DATE"] - forecast_date
).dt.days / DAY_COUNT

current_curve = current_curve[current_curve["tau_1m"] > 0].copy()

# Current DNS fitted curve
X_today = ns_loadings(current_curve["tau"].values, lam)
beta_today = np.array([beta0_t, beta1_t, beta2_t])
current_curve["DNS_fit_today"] = X_today @ beta_today

# Forecasted DNS curve
X_forecast = ns_loadings(current_curve["tau_1m"].values, lam)
beta_forecast = np.array([beta0_forecast, beta1_forecast, beta2_forecast])
current_curve["DNS_forecast_1m"] = X_forecast @ beta_forecast

output = current_curve[[
    "SECID",
    "MATURITY_DATE",
    "tau",
    "tau_1m",
    "YTM",
    "DNS_fit_today",
    "DNS_forecast_1m"
]].copy()

output["YTM"] *= 100
output["DNS_fit_today"] *= 100
output["DNS_forecast_1m"] *= 100

display(output)


# In[8]:


# =====================
# 8. GRAPH CURRENT MARKET, DNS FIT, DNS FORECAST
# =====================

plt.figure(figsize=(10, 6))

plt.plot(
    output["tau"],
    output["YTM"],
    marker="o",
    label="Current market OFZ YTM"
)

plt.plot(
    output["tau"],
    output["DNS_fit_today"],
    marker="o",
    label="Nelson-Siegel fit today"
)

plt.plot(
    output["tau_1m"],
    output["DNS_forecast_1m"],
    marker="o",
    label="Dynamic Nelson-Siegel forecast in 1 month"
)

plt.xlabel("Years to maturity")
plt.ylabel("Yield, %")
plt.title("Dynamic Nelson-Siegel Yield Curve Forecast")
plt.legend()
plt.grid(True)
plt.show()


# In[9]:


# =====================
# 9. PLOT DNS FACTORS THROUGH TIME
# =====================

plt.figure(figsize=(10, 5))

plt.plot(factors["DATE"], factors["beta0_level"] * 100, label="Level beta0")
plt.plot(factors["DATE"], factors["beta1_slope"] * 100, label="Slope beta1")
plt.plot(factors["DATE"], factors["beta2_curvature"] * 100, label="Curvature beta2")

plt.xlabel("Date")
plt.ylabel("Factor value, %")
plt.title("Dynamic Nelson-Siegel Factors")
plt.legend()
plt.grid(True)
plt.show()


# In[10]:


# # =====================
# # DYNAMIC NELSON-SIEGEL VS RANDOM WALK BACKTEST
# # =====================

# import numpy as np
# import pandas as pd
# from scipy import stats
# import matplotlib.pyplot as plt

# FORECAST_MONTHS = 1
# DAY_COUNT = 365.25
# MIN_FACTOR_OBS = 100
# MAX_REALIZED_GAP_DAYS = 10

# # make sure dates are datetime
# ofz["DATE"] = pd.to_datetime(ofz["DATE"])
# ofz["MATURITY_DATE"] = pd.to_datetime(ofz["MATURITY_DATE"])
# factors["DATE"] = pd.to_datetime(factors["DATE"])

# # make sure YTM is decimal
# if ofz["YTM"].abs().median() > 1:
#     ofz["YTM"] = ofz["YTM"] / 100

# ofz = ofz.sort_values(["DATE", "SECID"]).reset_index(drop=True)
# factors = factors.sort_values("DATE").reset_index(drop=True)


# In[11]:


# # =====================
# # HELPER: DIRECT 1-MONTH AR(1) FIT USING ONLY PAST FACTORS
# # =====================

# def make_factor_pairs_until(factors_train, col):
#     rows = []

#     for i in range(len(factors_train)):
#         date_t = factors_train.loc[i, "DATE"]
#         target_date = date_t + pd.DateOffset(months=FORECAST_MONTHS)

#         future = factors_train[factors_train["DATE"] >= target_date]

#         if future.empty:
#             continue

#         j = future.index[0]

#         rows.append({
#             "x": factors_train.loc[i, col],
#             "y": factors_train.loc[j, col]
#         })

#     return pd.DataFrame(rows)


# def fit_ar1_forecast(factors_train, col, current_value):
#     pairs = make_factor_pairs_until(factors_train, col)

#     if len(pairs) < 20:
#         return np.nan

#     ols = stats.linregress(pairs["x"], pairs["y"])

#     c = ols.intercept
#     phi = ols.slope

#     forecast = c + phi * current_value

#     return forecast


# In[12]:


# # =====================
# # BACKTEST LOOP
# # =====================

# bt_rows = []

# all_dates = sorted(factors["DATE"].unique())

# for i in range(MIN_FACTOR_OBS, len(all_dates)):

#     origin_date = pd.Timestamp(all_dates[i])
#     target_date = origin_date + pd.DateOffset(months=FORECAST_MONTHS)

#     # train only on factor data available up to origin date
#     factors_train = factors[factors["DATE"] <= origin_date].copy().reset_index(drop=True)

#     current_factor = factors_train.iloc[-1]

#     beta0_t = current_factor["beta0_level"]
#     beta1_t = current_factor["beta1_slope"]
#     beta2_t = current_factor["beta2_curvature"]

#     # forecast DNS factors using rolling AR(1)
#     beta0_f = fit_ar1_forecast(factors_train, "beta0_level", beta0_t)
#     beta1_f = fit_ar1_forecast(factors_train, "beta1_slope", beta1_t)
#     beta2_f = fit_ar1_forecast(factors_train, "beta2_curvature", beta2_t)

#     if np.isnan(beta0_f) or np.isnan(beta1_f) or np.isnan(beta2_f):
#         continue

#     today_curve = ofz[ofz["DATE"] == origin_date].copy()

#     if today_curve.empty:
#         continue

#     for _, row in today_curve.iterrows():

#         secid = row["SECID"]
#         maturity = row["MATURITY_DATE"]
#         ytm_today = row["YTM"]

#         tau_1m = (maturity - target_date).days / DAY_COUNT

#         if tau_1m <= 0:
#             continue

#         # DNS forecasted YTM
#         X_forecast = ns_loadings(np.array([tau_1m]), lam)

#         beta_forecast = np.array([
#             beta0_f,
#             beta1_f,
#             beta2_f
#         ])

#         ytm_dns_forecast = float(X_forecast @ beta_forecast)

#         # Random walk forecast
#         ytm_rw_forecast = ytm_today

#         # realized YTM of same bond around target date
#         future_rows = ofz[
#             (ofz["SECID"] == secid) &
#             (ofz["DATE"] >= target_date)
#         ].copy()

#         if future_rows.empty:
#             continue

#         realized_row = future_rows.iloc[0]
#         realized_date = realized_row["DATE"]

#         gap_days = (realized_date - target_date).days

#         if gap_days > MAX_REALIZED_GAP_DAYS:
#             continue

#         ytm_realized = realized_row["YTM"]

#         bt_rows.append({
#             "origin_date": origin_date,
#             "target_date": target_date,
#             "realized_date": realized_date,
#             "SECID": secid,
#             "MATURITY_DATE": maturity,
#             "tau_1m": tau_1m,
#             "YTM_today": ytm_today,
#             "YTM_realized": ytm_realized,
#             "YTM_DNS_forecast": ytm_dns_forecast,
#             "YTM_RW_forecast": ytm_rw_forecast,
#             "error_DNS": ytm_realized - ytm_dns_forecast,
#             "error_RW": ytm_realized - ytm_rw_forecast
#         })

# backtest_dns = pd.DataFrame(bt_rows)

# print("Number of backtest observations:", len(backtest_dns))
# display(backtest_dns.head())


# In[14]:


# =====================
# SIMPLE DNS VS RANDOM WALK: RMSE / MAE ONLY
# =====================
#USE THIS ONE !  !  ! 
import numpy as np
import pandas as pd

FORECAST_MONTHS = 1
DAY_COUNT = 365.25
MAX_REALIZED_GAP_DAYS = 10

ofz["DATE"] = pd.to_datetime(ofz["DATE"])
ofz["MATURITY_DATE"] = pd.to_datetime(ofz["MATURITY_DATE"])
factors["DATE"] = pd.to_datetime(factors["DATE"])

# Make sure YTM is decimal
if ofz["YTM"].abs().median() > 1:
    ofz["YTM"] = ofz["YTM"] / 100

ofz = ofz.sort_values(["DATE", "SECID"]).reset_index(drop=True)
factors = factors.sort_values("DATE").reset_index(drop=True)

rows = []

for _, frow in factors.iterrows():

    origin_date = frow["DATE"]
    target_date = origin_date + pd.DateOffset(months=FORECAST_MONTHS)

    # Forecast DNS factors using fitted AR(1)
    beta0_f = ar0["c"] + ar0["phi"] * frow["beta0_level"]
    beta1_f = ar1["c"] + ar1["phi"] * frow["beta1_slope"]
    beta2_f = ar2["c"] + ar2["phi"] * frow["beta2_curvature"]

    beta_forecast = np.array([beta0_f, beta1_f, beta2_f])

    today_curve = ofz[ofz["DATE"] == origin_date].copy()

    if today_curve.empty:
        continue

    for _, row in today_curve.iterrows():

        secid = row["SECID"]
        maturity = row["MATURITY_DATE"]
        ytm_today = row["YTM"]

        tau_1m = (maturity - target_date).days / DAY_COUNT

        if tau_1m <= 0:
            continue

        # DNS forecast
        X_forecast = ns_loadings(np.array([tau_1m]), lam)
        ytm_dns_forecast = (X_forecast @ beta_forecast)[0]

        # Random walk forecast
        ytm_rw_forecast = ytm_today

        # Realized YTM of same bond after 1 month
        future_rows = ofz[
            (ofz["SECID"] == secid) &
            (ofz["DATE"] >= target_date)
        ].copy()

        if future_rows.empty:
            continue

        realized_row = future_rows.iloc[0]
        realized_date = realized_row["DATE"]

        if (realized_date - target_date).days > MAX_REALIZED_GAP_DAYS:
            continue

        ytm_realized = realized_row["YTM"]

        rows.append({
            "error_DNS": ytm_realized - ytm_dns_forecast,
            "error_RW": ytm_realized - ytm_rw_forecast
        })

errors = pd.DataFrame(rows)

rmse_dns = np.sqrt(np.mean(errors["error_DNS"] ** 2))
rmse_rw = np.sqrt(np.mean(errors["error_RW"] ** 2))

mae_dns = np.mean(np.abs(errors["error_DNS"]))
mae_rw = np.mean(np.abs(errors["error_RW"]))

print("Number of observations:", len(errors))
print()
print("RMSE Dynamic Nelson-Siegel:", rmse_dns * 100, "%")
print("RMSE Random Walk:", rmse_rw * 100, "%")
print()
print("MAE Dynamic Nelson-Siegel:", mae_dns * 100, "%")
print("MAE Random Walk:", mae_rw * 100, "%")


# In[ ]:




