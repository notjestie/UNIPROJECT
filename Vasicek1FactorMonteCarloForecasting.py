#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =====================
# 5. MONTE CARLO FORECAST SHORT RATE USING dW ~ N(0,h)
# =====================

N = 10000          # number of Monte Carlo simulations
M = 30             # number of small time steps inside 1 month
H = 1 / 12         # 1 month in years
h = H / M          # small time increment

curve_date = ruonia["date"].iloc[-1]
forecast_date = curve_date + pd.DateOffset(months=1)

r0 = ruonia["r"].iloc[-1]

# matrix of Brownian increments:
# dW = W_{t+h} - W_t ~ N(0,h)
dW = np.random.normal(
    loc=0,
    scale=np.sqrt(h),
    size=(N, M)
)

# simulate short-rate paths
r_paths = np.zeros((N, M + 1))
r_paths[:, 0] = r0

for k in range(M):
    r_paths[:, k + 1] = (
        r_paths[:, k]
        + a_P * (b_P - r_paths[:, k]) * h
        + sigma * dW[:, k]
    )

# final simulated short rates after 1 month
r_1m_sim = r_paths[:, -1]

# Monte Carlo average forecasted short rate
r_1m_mc = np.mean(r_1m_sim)

print("Current short rate r0:", r0 * 100, "%")
print("Monte Carlo forecasted short rate in 1 month:", r_1m_mc * 100, "%")

# =====================
# 6. VASICEK YIELD FORMULA
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

ofz["tau_today"] = (
    ofz["MATURITY_DATE"] - curve_date
).dt.days / DAY_COUNT

ofz["tau_1m"] = (
    ofz["MATURITY_DATE"] - forecast_date
).dt.days / DAY_COUNT

ofz = ofz[ofz["tau_1m"] > 0].copy()
ofz = ofz.sort_values("tau_1m").reset_index(drop=True)

ofz["forecast_yield_1m"] = vasicek_yield(
    tau=ofz["tau_1m"].values,
    r=r_1m_mc,
    a=a_P,
    b=b_P,
    sigma=sigma
)
# =====================
# 8. OUTPUT
# =====================

output = ofz[[
    "MATURITY_DATE",
    "tau_1m",
    "YTM",
    "forecast_yield_1m"
]].copy()

output["YTM"] *= 100
output["forecast_yield_1m"] *= 100
output["r_forecast_1m"] = r_1m_mc * 100

display(output)

# =====================
# 9. GRAPH
# =====================

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))

plt.plot(
    output["tau_1m"],
    output["YTM"],
    marker="o",
    label="Current OFZ YTM"
)

plt.plot(
    output["tau_1m"],
    output["forecast_yield_1m"],
    marker="o",
    label="Forecasted Vasicek curve in 1 month"
)

plt.axhline(
    y=r_1m_mc * 100,
    linestyle="--",
    label="Forecasted short rate"
)

plt.xlabel("Years to maturity")
plt.ylabel("Yield, %")
plt.title("OFZ Yield Curve Forecast: 1 Month Ahead")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




