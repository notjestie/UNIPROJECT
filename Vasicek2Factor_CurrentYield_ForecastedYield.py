#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =====================
# 7. CHECK CURRENT CURVE FIT
# =====================

current_date = ofz["DATE"].max()

current_curve = ofz[ofz["DATE"] == current_date].copy()
current_factor = factors[factors["DATE"] == current_date].iloc[0]

x0 = current_factor["x"]
y0 = current_factor["y"]
r0 = current_factor["r"]

current_curve["model_YTM"] = two_factor_yield(
    tau=current_curve["tau"].values,
    x=x0,
    y=y0,
    a1=a1_Q,
    a2=a2_Q,
    b1=b1_Q,
    b2=b2_Q,
    sigma1=sigma1_Q,
    sigma2=sigma2_Q,
    rho=rho_Q
)

current_curve["fit_error_bp"] = (
    current_curve["model_YTM"] - current_curve["YTM"]
) * 10000

fit_rmse_bp = np.sqrt(np.mean(current_curve["fit_error_bp"] ** 2))

print("Current date:", current_date.date())
print("x0:", x0 * 100, "%")
print("y0:", y0 * 100, "%")
print("r0 = x0 + y0:", (x0 + y0) * 100, "%")
print("Observed RUONIA r0:", r0 * 100, "%")
print("Current curve fit RMSE:", fit_rmse_bp, "bp")

display(current_curve[["MATURITY_DATE", "YTM", "model_YTM", "fit_error_bp"]].head())


# In[ ]:


# =====================
# 8. FORECAST FACTORS 1 MONTH AHEAD UNDER P
# =====================

x_1m = b1_P + (x0 - b1_P) * np.exp(-a1_P * H)
y_1m = b2_P + (y0 - b2_P) * np.exp(-a2_P * H)

r_1m = x_1m + y_1m

forecast_date = current_date + pd.DateOffset(months=1)

print("Forecast date:", forecast_date.date())
print("x forecast 1m:", x_1m * 100, "%")
print("y forecast 1m:", y_1m * 100, "%")
print("r forecast 1m:", r_1m * 100, "%")


# In[ ]:


# =====================
# 9. FORECAST YIELD CURVE 1 MONTH AHEAD
# =====================

forecast_curve = current_curve.copy()

forecast_curve["tau_1m"] = (
    forecast_curve["MATURITY_DATE"] - forecast_date
).dt.days / DAY_COUNT

forecast_curve = forecast_curve[forecast_curve["tau_1m"] > 0].copy()

forecast_curve["forecast_YTM_1m"] = two_factor_yield(
    tau=forecast_curve["tau_1m"].values,
    x=x_1m,
    y=y_1m,
    a1=a1_Q,
    a2=a2_Q,
    b1=b1_Q,
    b2=b2_Q,
    sigma1=sigma1_Q,
    sigma2=sigma2_Q,
    rho=rho_Q
)

output = forecast_curve[[
    "MATURITY_DATE",
    "tau",
    "tau_1m",
    "YTM",
    "model_YTM",
    "forecast_YTM_1m"
]].copy()

output["YTM"] *= 100
output["model_YTM"] *= 100
output["forecast_YTM_1m"] *= 100
output["r_forecast_1m"] = r_1m * 100
output["x_forecast_1m"] = x_1m * 100
output["y_forecast_1m"] = y_1m * 100

display(output)


# In[ ]:


# =====================
# 10. GRAPH
# =====================

plt.figure(figsize=(9, 5))

plt.plot(
    output["tau"],
    output["YTM"],
    marker="o",
    label="Current market OFZ YTM"
)

plt.plot(
    output["tau"],
    output["model_YTM"],
    marker="o",
    label="Two-factor Vasicek fit today"
)

plt.plot(
    output["tau_1m"],
    output["forecast_YTM_1m"],
    marker="o",
    label="Forecasted curve in 1 month"
)

plt.axhline(
    y=r_1m * 100,
    linestyle="--",
    label=f"Forecasted short rate = {r_1m * 100:.2f}%"
)

plt.xlabel("Years to maturity")
plt.ylabel("Yield, %")
plt.title("Two-Factor Vasicek Yield Curve Forecast")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




