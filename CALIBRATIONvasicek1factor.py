#!/usr/bin/env python
# coding: utf-8

# In[ ]:


rom scipy.optimize import minimize_scalar
import numpy as np
import pandas as pd
from scipy import stats

# =====================
# FILE INPUTS
# =====================

RUONIA_FILE = "RC_F01_01_2024_T20_05_2026.xlsx"
OFZ_FILE = "ofz_zcyc_ytmNEW_2026-05-20.csv"

# For your RUONIA file, columns are:
# Дата, Индекс, 1 месяц, 3 месяца, 6 месяцев
RUONIA_DATE_COL = "Дата"
RUONIA_RATE_COL = "Индекс"

DAY_COUNT = 365.25


# =====================
# 1. EXTRACT RUONIA RATES
# =====================

ruonia = pd.read_excel(RUONIA_FILE)

ruonia = ruonia[[RUONIA_DATE_COL, RUONIA_RATE_COL]].copy()
ruonia.columns = ["date", "index"]

ruonia["date"] = pd.to_datetime(ruonia["date"])
ruonia = ruonia.sort_values("date").drop_duplicates("date").reset_index(drop=True)

gap = ruonia["date"].diff().dt.days
ruonia["r"] = (ruonia["index"] / ruonia["index"].shift(1) - 1) * 365.0 / gap   # annualized overnight
ruonia = ruonia.dropna(subset=["r"]).reset_index(drop=True)

ruonia = (
    ruonia
    .dropna()
    .sort_values("date")
    .drop_duplicates("date")
    .reset_index(drop=True)
)

print("RUONIA data:")
display(ruonia.head())
display(ruonia.tail())


# =====================
# 2. EXTRACT OFZ MATURITIES AND YTMS
# =====================

ofz = pd.read_csv(OFZ_FILE, skipinitialspace=True)

ofz.columns = ofz.columns.str.strip()

ofz = ofz[["MATURITY_DATE", "YTM"]].copy()

ofz["MATURITY_DATE"] = pd.to_datetime(ofz["MATURITY_DATE"])
ofz["YTM"] = pd.to_numeric(ofz["YTM"], errors="coerce") / 100

ofz = ofz.dropna().reset_index(drop=True)

print("OFZ curve data:")
display(ofz.head())

r = ruonia["r"].values

x = r[:-1]
y = r[1:]

dt = ruonia["date"].diff().dt.days.median() / DAY_COUNT

# =====================
# CONSTRAINED OLS:
# force 0 < beta < 1
# =====================

def sse_for_beta(beta):
    alpha = np.mean(y - beta * x)
    residuals = y - alpha - beta * x
    return np.sum(residuals ** 2)

res = minimize_scalar(
    sse_for_beta,
    bounds=(1e-8, 0.999999),
    method="bounded"
)

beta = res.x
alpha = np.mean(y - beta * x)

a_P = -np.log(beta) / dt
b_P = alpha / (1 - beta)

residuals = y - alpha - beta * x
sigma_eps = np.std(residuals, ddof=2)

sigma = sigma_eps * np.sqrt(2 * a_P / (1 - beta**2))

print("Constrained Vasicek P-measure parameters:")
print("alpha =", alpha)
print("beta  =", beta)
print("a_P   =", a_P)
print("b_P   =", b_P)
print("sigma =", sigma)

print()
print("In percent:")
print("b_P   =", b_P * 100, "%")
print("sigma =", sigma * 100, "%")

