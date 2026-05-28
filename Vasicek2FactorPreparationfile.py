#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import glob
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt

# =====================
# INPUTS
# =====================

RUONIA_FILE = "RC_F01_01_2024_T20_05_2026.xlsx"

# Can be one CSV with many DATE values, or many files:
# "ofz_zcyc_ytmNEW_*.csv"
OFZ_INPUT = "ofz_zcyc_ytmNEW_2024-01-01_to_2026-05-20.csv"

RUONIA_DATE_COL = "Дата"
RUONIA_INDEX_COL = "Индекс"

DAY_COUNT = 365.25
H = 1 / 12

# Initial Q speeds.
# Factor 1 = faster factor, short-end effect.
# Factor 2 = slower factor, long-end effect.
A1_Q_INIT = 2.0
A2_Q_INIT = 0.20

# Initial volatility guesses used before factor extraction.
SIGMA1_INIT = 0.02
SIGMA2_INIT = 0.01
RHO_INIT = 0.0

MIN_DATES_FOR_2F = 20


# In[ ]:


# =====================
# HELPERS
# =====================

def to_decimal_rate(s):
    x = pd.to_numeric(s, errors="coerce")
    if x.dropna().abs().median() > 1:
        x = x / 100
    return x


def B_vasicek(tau, a):
    return (1 - np.exp(-a * tau)) / a


def I_square(tau, a):
    B = B_vasicek(tau, a)
    return (tau - 2 * B + (1 - np.exp(-2 * a * tau)) / (2 * a)) / a**2


def I_cross(tau, a1, a2):
    B1 = B_vasicek(tau, a1)
    B2 = B_vasicek(tau, a2)

    return (
        tau
        - B1
        - B2
        + (1 - np.exp(-(a1 + a2) * tau)) / (a1 + a2)
    ) / (a1 * a2)


def C_two_factor(tau, a1, a2, b1, b2, sigma1, sigma2, rho):
    B1 = B_vasicek(tau, a1)
    B2 = B_vasicek(tau, a2)

    I1 = I_square(tau, a1)
    I2 = I_square(tau, a2)
    I12 = I_cross(tau, a1, a2)

    C = (
        b1 * (B1 - tau)
        + b2 * (B2 - tau)
        + 0.5 * sigma1**2 * I1
        + 0.5 * sigma2**2 * I2
        + rho * sigma1 * sigma2 * I12
    )

    return C


def two_factor_yield(tau, x, y, a1, a2, b1, b2, sigma1, sigma2, rho):
    tau = np.asarray(tau, dtype=float)

    B1 = B_vasicek(tau, a1)
    B2 = B_vasicek(tau, a2)

    C = C_two_factor(tau, a1, a2, b1, b2, sigma1, sigma2, rho)

    Y = -C / tau + (B1 / tau) * x + (B2 / tau) * y

    return Y


def estimate_ou_params(series, dates):
    df = pd.DataFrame({"date": dates, "z": series}).dropna().sort_values("date")

    z = df["z"].values
    x = z[:-1]
    y = z[1:]

    dt = df["date"].diff().dt.days.median() / DAY_COUNT

    ols = stats.linregress(x, y)
    alpha = ols.intercept
    beta = ols.slope

    if beta <= 0 or beta >= 1:
        print("Warning: beta outside stable region:", beta)
        print("Using clipped beta close to stable region.")
        beta = min(max(beta, 1e-8), 0.999999)
        alpha = np.mean(y - beta * x)

    a = -np.log(beta) / dt
    b = alpha / (1 - beta)

    residuals = y - (alpha + beta * x)
    sigma_eps = np.std(residuals, ddof=2)

    sigma = sigma_eps * np.sqrt(2 * a / (1 - beta**2))

    return {
        "alpha": alpha,
        "beta": beta,
        "a": a,
        "b": b,
        "sigma": sigma,
        "residuals": residuals,
        "dates": df["date"].iloc[1:].values
    }


# In[ ]:


# =====================
# 1. LOAD RUONIA AND CREATE SHORT RATE r_t
# =====================

ruonia_raw = pd.read_excel(RUONIA_FILE)

ruonia = ruonia_raw[[RUONIA_DATE_COL, RUONIA_INDEX_COL]].copy()
ruonia.columns = ["date", "index"]

ruonia["date"] = pd.to_datetime(ruonia["date"])
ruonia["index"] = pd.to_numeric(ruonia["index"], errors="coerce")

ruonia = (
    ruonia
    .dropna()
    .sort_values("date")
    .drop_duplicates("date")
    .reset_index(drop=True)
)

gap = ruonia["date"].diff().dt.days

ruonia["r"] = (
    ruonia["index"] / ruonia["index"].shift(1) - 1
) * 365.0 / gap

ruonia = ruonia.dropna(subset=["r"]).reset_index(drop=True)

display(ruonia.head())
display(ruonia.tail())


# In[ ]:


# =====================
# 2. LOAD HISTORICAL OFZ CURVES
# =====================

files = sorted(glob.glob(OFZ_INPUT))

if len(files) == 0:
    raise FileNotFoundError(f"No files found for OFZ_INPUT={OFZ_INPUT}")

ofz_list = []

for f in files:
    tmp = pd.read_csv(f, encoding="utf-8-sig", skipinitialspace=True)
    tmp.columns = tmp.columns.str.strip()
    ofz_list.append(tmp)

ofz = pd.concat(ofz_list, ignore_index=True)

needed = {"DATE", "MATURITY_DATE", "YTM"}
missing = needed - set(ofz.columns)

if missing:
    raise ValueError(f"Missing columns in OFZ data: {missing}")

ofz["DATE"] = pd.to_datetime(ofz["DATE"])
ofz["MATURITY_DATE"] = pd.to_datetime(ofz["MATURITY_DATE"])
ofz["YTM"] = to_decimal_rate(ofz["YTM"])

ofz = ofz.dropna(subset=["DATE", "MATURITY_DATE", "YTM"])

ofz["tau"] = (ofz["MATURITY_DATE"] - ofz["DATE"]).dt.days / DAY_COUNT
ofz = ofz[ofz["tau"] > 0].copy()

ofz = ofz.sort_values(["DATE", "tau"]).reset_index(drop=True)

n_dates = ofz["DATE"].nunique()

print("Number of OFZ curve dates:", n_dates)
print("Date range:", ofz["DATE"].min().date(), "to", ofz["DATE"].max().date())

if n_dates < MIN_DATES_FOR_2F:
    raise ValueError(
        "You do not have enough historical OFZ curve dates for two-factor calibration. "
        "Use many daily CSV files or one CSV with many DATE values."
    )

display(ofz.head())


# In[ ]:


# =====================
# 3. MATCH EACH CURVE DATE TO RUONIA SHORT RATE
# =====================

curve_dates = pd.DataFrame({"DATE": sorted(ofz["DATE"].unique())})

ruonia_for_merge = ruonia[["date", "r"]].copy()
ruonia_for_merge = ruonia_for_merge.rename(columns={"date": "DATE"})

curve_dates = pd.merge_asof(
    curve_dates.sort_values("DATE"),
    ruonia_for_merge.sort_values("DATE"),
    on="DATE",
    direction="backward"
)

curve_dates = curve_dates.dropna(subset=["r"]).reset_index(drop=True)

ofz = ofz.merge(curve_dates, on="DATE", how="inner")

dates_unique = sorted(ofz["DATE"].unique())
date_to_idx = {d: i for i, d in enumerate(dates_unique)}

ofz["date_idx"] = ofz["DATE"].map(date_to_idx)

r_by_date = (
    ofz[["DATE", "r"]]
    .drop_duplicates("DATE")
    .sort_values("DATE")
    .reset_index(drop=True)
)

print("Dates after matching RUONIA:", len(r_by_date))
display(r_by_date.head())

