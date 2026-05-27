#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =====================
# 5. FORECAST SHORT RATE 1 MONTH AHEAD UNDER P
# =====================

H = 1 / 12

curve_date = ruonia["date"].iloc[-1]
r0 = ruonia["r"].iloc[-1]

r_1m = b_P + (r0 - b_P) * np.exp(-a_P * H)

print("Curve date:", curve_date.date())
print("Current short rate r0:", r0 * 100, "%")
print("Expected short rate in 1 month:", r_1m * 100, "%")
# =====================
# 6. VASICEK YIELD FORMULA USING ONLY P PARAMETERS
# =====================

def B_vasicek(tau, a):
    return (1 - np.exp(-a * tau)) / a


def vasicek_yield(tau, r, a, b, sigma):
    tau = np.asarray(tau, dtype=float)

    B = B_vasicek(tau, a)

    lnA = (
        (b - sigma**2 / (2 * a**2)) * (B - tau)
        - sigma**2 * B**2 / (4 * a)
    )

    lnP = lnA - B * r

    return -lnP / tau
# =====================
# 7. FORECAST YIELD CURVE IN 1 MONTH
# =====================

forecast_date = curve_date + pd.DateOffset(months=1)

ofz["tau_today"] = (
    ofz["MATURITY_DATE"] - curve_date
).dt.days / DAY_COUNT

ofz["tau_1m"] = (
    ofz["MATURITY_DATE"] - forecast_date
).dt.days / DAY_COUNT

ofz = ofz[ofz["tau_1m"] > 0].copy()

ofz["vasicek_yield_today"] = vasicek_yield(
    tau=ofz["tau_today"].values,
    r=r0,
    a=a_P,
    b=b_P,
    sigma=sigma
)

ofz["forecast_yield_1m"] = vasicek_yield(
    tau=ofz["tau_1m"].values,
    r=r_1m,
    a=a_P,
    b=b_P,
    sigma=sigma
)
# =====================
# 8. OUTPUT
# =====================

output = ofz[[
    "MATURITY_DATE",
    "YTM",
    "vasicek_yield_today",
    "forecast_yield_1m"
]].copy()

output["YTM"] *= 100
output["vasicek_yield_today"] *= 100
output["forecast_yield_1m"] *= 100

output = output.sort_values("MATURITY_DATE").reset_index(drop=True)

print("P-measure Vasicek parameters:")
print("a_P   =", a_P)
print("b_P   =", b_P * 100, "%")
print("sigma =", sigma * 100, "%")
print()

print("Short rate forecast:")
print("r0      =", r0 * 100, "%")
print("E[r_1m] =", r_1m * 100, "%")
print()

display(output)

output["r_forecast_1m"] = r_1m * 100   # percent

display(output)


# =====================
# 10. GRAPH
# =====================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(
    output["MATURITY_DATE"],
    output["YTM"],
    marker="o",
    label="Current market YTM"
)

plt.plot(
    output["MATURITY_DATE"],
    output["vasicek_yield_today"],
    marker="o",
    label="Vasicek curve today"
)

plt.plot(
    output["MATURITY_DATE"],
    output["forecast_yield_1m"],
    marker="o",
    label="Forecasted curve in 1 month"
)

plt.axhline(
    y=r_1m * 100,
    linestyle="--",
    label="Forecasted short rate r in 1 month"
)

plt.xlabel("Maturity date")
plt.ylabel("Yield / rate, %")
plt.title("Vasicek Yield Curve Forecast: 1 Month Ahead")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:




