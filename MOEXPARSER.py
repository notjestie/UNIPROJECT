#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests, pandas as pd
from datetime import datetime
from time import sleep

ISS = "https://iss.moex.com/iss"
s = requests.Session()
s.trust_env = False

def get(path, block, params=None):
    for _ in range(5):
        try:
            r = s.get(ISS + path, params={**(params or {}), "iss.meta": "off"}, timeout=30)
            r.raise_for_status()
            j = r.json()
            return pd.DataFrame(j[block]["data"], columns=j[block]["columns"]).rename(str.upper, axis=1)
        except Exception as e:
            err = e
            sleep(1)
    raise err

def get_maturity(secid):
    d = get(f"/securities/{secid}.json", "description")
    d["NAME"] = d["NAME"].astype(str).str.upper()

    row = d[d["NAME"].isin(["MATDATE", "MATURITYDATE"])]
    if not row.empty:
        return row.iloc[0]["VALUE"]

    return None

date = input("Date dd-mm-yyyy: ")
date = datetime.strptime(date, "%d-%m-%Y").strftime("%Y-%m-%d")

# current MOEX zero-coupon curve bond base
zcyc = get("/engines/stock/zcyc.json", "securities", {"iss.only": "securities"})
secids = zcyc["SECID"].dropna().unique()

rows = []

for secid in secids:
    matdate = get_maturity(secid)

    h = get(
        f"/history/engines/stock/markets/bonds/boards/TQOB/securities/{secid}.json",
        "history",
        {"from": date, "till": date}
    )

    if h.empty:
        rows.append({
            "SECID": secid,
            "MATURITY_DATE": matdate,
            "DATE": date,
            "YTM": None
        })
    else:
        row = h.iloc[-1]
        ytm = None

        for col in ["YIELDATLAST", "YIELDCLOSE", "YIELDATWAP", "YIELDATPREVWAPRICE"]:
            if col in h.columns and pd.notna(row[col]):
                ytm = row[col]
                break

        rows.append({
            "SECID": secid,
            "MATURITY_DATE": matdate,
            "DATE": date,
            "YTM": ytm
        })

    sleep(0.2)

result = pd.DataFrame(rows)

print(result)
result.to_csv(f"ofz_zcyc_ytmNEW_{date}.csv", index=False)

