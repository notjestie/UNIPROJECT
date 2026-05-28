#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =====================
# 4. FIT Q-MEASURE b1_Q, b2_Q AND LATENT x_t, y_t
# =====================

def fit_Q_and_factors(ofz, r_by_date, a1_Q, a2_Q, sigma1, sigma2, rho):
    dates = r_by_date["DATE"].values
    r_vec = r_by_date["r"].values
    n = len(dates)

    tau = ofz["tau"].values
    y_market = ofz["YTM"].values
    idx = ofz["date_idx"].values.astype(int)

    # Initial guesses
    b1_init = np.mean(r_vec) / 2
    b2_init = np.mean(r_vec) / 2
    x_init = r_vec / 2

    theta0 = np.r_[b1_init, b2_init, x_init]

    def objective(theta):
        b1_Q = theta[0]
        b2_Q = theta[1]
        x_vec = theta[2:]

        x_obs = x_vec[idx]
        y_obs = r_vec[idx] - x_obs

        y_model = two_factor_yield(
            tau=tau,
            x=x_obs,
            y=y_obs,
            a1=a1_Q,
            a2=a2_Q,
            b1=b1_Q,
            b2=b2_Q,
            sigma1=sigma1,
            sigma2=sigma2,
            rho=rho
        )

        return np.mean((y_model - y_market) ** 2)

    bounds = [(-1.0, 1.0), (-1.0, 1.0)] + [(-1.0, 1.0)] * n

    res = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000}
    )

    theta = res.x

    b1_Q = theta[0]
    b2_Q = theta[1]
    x_vec = theta[2:]
    y_vec = r_vec - x_vec

    factors = pd.DataFrame({
        "DATE": dates,
        "r": r_vec,
        "x": x_vec,
        "y": y_vec
    })

    return {
        "result": res,
        "b1_Q": b1_Q,
        "b2_Q": b2_Q,
        "factors": factors
    }


# First Q/factor fit using initial guesses
qfit_1 = fit_Q_and_factors(
    ofz=ofz,
    r_by_date=r_by_date,
    a1_Q=A1_Q_INIT,
    a2_Q=A2_Q_INIT,
    sigma1=SIGMA1_INIT,
    sigma2=SIGMA2_INIT,
    rho=RHO_INIT
)

factors = qfit_1["factors"]

print("Initial Q fit success:", qfit_1["result"].success)
print("Initial b1_Q:", qfit_1["b1_Q"] * 100, "%")
print("Initial b2_Q:", qfit_1["b2_Q"] * 100, "%")

display(factors.head())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




