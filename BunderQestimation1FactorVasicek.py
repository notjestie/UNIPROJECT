#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy.optimize import minimize_scalar

# =====================
# FIT b* UNDER Q
# =====================

curve_date = pd.to_datetime("2026-05-20")
r0 = ruonia["r"].iloc[-1]

ofz["tau"] = (ofz["MATURITY_DATE"] - curve_date).dt.days / DAY_COUNT
ofz = ofz[ofz["tau"] > 0].copy()

tau = ofz["tau"].values
ytm = ofz["YTM"].values


def B(tau, a):
    return (1 - np.exp(-a * tau)) / a


def vasicek_yield(tau, r, a, b, sigma):
    Btau = B(tau, a)

    lnA = (
        (b - sigma**2 / (2 * a**2)) * (Btau - tau)
        - sigma**2 * Btau**2 / (4 * a)
    )

    lnP = lnA - Btau * r

    return -lnP / tau


def loss_bq(bq):
    model = vasicek_yield(
        tau=tau,
        r=r0,
        a=a_P,
        b=bq,
        sigma=sigma
    )
    return np.mean((model - ytm) ** 2)


res = minimize_scalar(loss_bq, bounds=(-1, 1), method="bounded")

b_Q = res.x

print("b_Q / b* =", b_Q)
print("b_Q / b* in percent =", b_Q * 100, "%")

ofz["model_YTM"] = vasicek_yield(
    tau=ofz["tau"].values,
    r=r0,
    a=a_P,
    b=b_Q,
    sigma=sigma
)

ofz["fit_error_bp"] = (ofz["model_YTM"] - ofz["YTM"]) * 10000

display(ofz[["MATURITY_DATE", "YTM", "model_YTM", "fit_error_bp"]])

