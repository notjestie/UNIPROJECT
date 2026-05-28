#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =====================
# 6. REFIT Q b1_Q, b2_Q USING P SPEEDS AND VOLATILITIES
# =====================

a1_Q = a1_P
a2_Q = a2_P
sigma1_Q = sigma1_P
sigma2_Q = sigma2_P
rho_Q = rho_P

qfit_2 = fit_Q_and_factors(
    ofz=ofz,
    r_by_date=r_by_date,
    a1_Q=a1_Q,
    a2_Q=a2_Q,
    sigma1=sigma1_Q,
    sigma2=sigma2_Q,
    rho=rho_Q
)

b1_Q = qfit_2["b1_Q"]
b2_Q = qfit_2["b2_Q"]
factors = qfit_2["factors"]

print("Final Q fit success:", qfit_2["result"].success)
print("b1_Q =", b1_Q * 100, "%")
print("b2_Q =", b2_Q * 100, "%")
print("Long-run short rate under Q =", (b1_Q + b2_Q) * 100, "%")

display(factors.tail())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




