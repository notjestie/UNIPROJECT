#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
from datetime import datetime
from time import sleep

ISS = "https://iss.moex.com/iss"

s = requests.Session()
s.trust_env = False


def get(path, block, params=None):
    for _ in range(5):
        try:
            r = s.get(
                ISS + path,
                params={**(params or {}), "iss.meta": "off"},
                timeout=30
            )
            r.raise_for_status()
            j = r.json()

            return (
                pd.DataFrame(
                    j[block]["data"],
                    columns=j[block]["columns"]
                )
                .rename(str.upper, axis=1)
            )

        except Exception as e:
            err = e
            sleep(1)

    raise err


def get_all(path, block, params=None):
    """
    MOEX ISS often returns data in pages.
    This function collects all pages.
    """
    frames = []
    start = 0

    while True:
        chunk = get(
            path,
            block,
            {**(params or {}), "start": start}
        )

        if chunk.empty:
            break

        frames.append(chunk)

        if len(chunk) < 100:
            break

        start += len(chunk)
        sleep(0.1)

    if frames:
        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame()


def get_maturity(secid):
    d = get(f"/securities/{secid}.json", "description")
    d["NAME"] = d["NAME"].astype(str).str.upper()

    row = d[d["NAME"].isin(["MATDATE", "MATURITYDATE"])]

    if not row.empty:
        return row.iloc[0]["VALUE"]

    return None


start_date = input("Start date dd-mm-yyyy: ")
end_date = input("End date dd-mm-yyyy: ")

start_date = datetime.strptime(start_date, "%d-%m-%Y").strftime("%Y-%m-%d")
end_date = datetime.strptime(end_date, "%d-%m-%Y").strftime("%Y-%m-%d")

if end_date < start_date:
    raise ValueError("End date must be after or equal to start date")

# current MOEX zero-coupon curve bond base
zcyc = get(
    "/engines/stock/zcyc.json",
    "securities",
    {"iss.only": "securities"}
)

secids = zcyc["SECID"].dropna().unique()

rows = []

for secid in secids:
    matdate = get_maturity(secid)

    h = get_all(
        f"/history/engines/stock/markets/bonds/boards/TQOB/securities/{secid}.json",
        "history",
        {
            "from": start_date,
            "till": end_date
        }
    )

    if h.empty:
        continue

    yield_cols = [
        "YIELDATLAST",
        "YIELDCLOSE",
        "YIELDATWAP",
        "YIELDATPREVWAPRICE"
    ]

    available_yield_cols = [col for col in yield_cols if col in h.columns]

    if not available_yield_cols:
        continue

    h["YTM"] = h[available_yield_cols].bfill(axis=1).iloc[:, 0]

    # Remove rows where YTM is NaN
    h = h.dropna(subset=["YTM"])

    # If this bond has no valid YTM values in the whole range, skip it
    if h.empty:
        continue

    for _, row in h.iterrows():
        rows.append({
            "SECID": secid,
            "MATURITY_DATE": matdate,
            "DATE": row["TRADEDATE"],
            "YTM": row["YTM"]
        })

    sleep(0.2)

result = pd.DataFrame(rows)

# Keep same column order as original code
result = result[["SECID", "MATURITY_DATE", "DATE", "YTM"]]

print(result)

output_file = f"ofz_zcyc_ytmNEW_{start_date}_to_{end_date}.csv"
result.to_csv(output_file, index=False)

print(f"Saved to {output_file}")


# In[ ]:




