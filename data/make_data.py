"""Generate the five synthetic teaching datasets for 125.785.
All data are SIMULATED. Structures mimic Vietnamese listed firms, banks and markets
so that every session has a known answer. Seed is fixed for reproducibility."""
import numpy as np, pandas as pd, os, sys
rng = np.random.default_rng(125785)
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 1. corporate panel
inds = ["Real estate", "Construction", "Materials", "Consumer", "Industrials",
        "Energy", "Technology", "Agriculture"]
nf, years = 150, np.arange(2015, 2025)
rows = []
for i in range(nf):
    ind = rng.integers(len(inds))
    quality = rng.normal(0, 1)                 # unobserved firm quality (fixed effect)
    state = rng.random() < 0.25
    size0 = rng.normal(28.5, 1.3)              # log total assets (VND)
    tang = np.clip(rng.normal(0.35, 0.15), 0.02, 0.9)
    ind_lev_mean = [0.62, 0.58, 0.48, 0.40, 0.50, 0.55, 0.30, 0.45][ind]
    lev_i = ind_lev_mean - 0.06 * quality + rng.normal(0, 0.08)
    foreign = np.clip(rng.beta(2, 8), 0, 0.49)
    esg_i = 40 + 6 * (size0 - 28.5) + 8 * quality + rng.normal(0, 8)
    for t, y in enumerate(years):
        size = size0 + 0.05 * t + rng.normal(0, 0.08)
        lev = np.clip(lev_i + rng.normal(0, 0.05), 0.02, 0.95)
        growth = rng.normal(0.08, 0.18) - (0.15 if y == 2020 else 0) + 0.03 * quality
        covid = 1 if y in (2020, 2021) else 0
        roa = (0.06 + 0.025 * quality - 0.07 * (lev - 0.5) + 0.004 * (size - 28.5)
               + 0.05 * growth + 0.02 * (tang - 0.35) - 0.01 * state - 0.015 * covid
               + rng.standard_t(5) * 0.02)
        esg = np.clip(esg_i + 1.5 * t + rng.normal(0, 5), 0, 100)
        kd = np.clip(0.095 - 0.0004 * (esg - 50) - 0.006 * (size - 28.5)
                     + 0.03 * (lev - 0.5) - 0.004 * quality + (0.01 if y in (2022, 2023) else 0)
                     + rng.normal(0, 0.012), 0.02, 0.2)
        bm = np.clip(np.exp(rng.normal(0.1 - 0.3 * quality, 0.4)), 0.1, 5)
        z = -1.2 + 25 * roa - 1.5 * lev + 0.3 * (size - 28.5) + 0.8 * state + rng.logistic()
        div = int(z > 0)
        rows.append(dict(firm_id=f"F{i+1:03d}", year=y, industry=inds[ind], state_owned=int(state),
                         foreign_own=round(foreign, 4), size=round(size, 4), leverage=round(lev, 4),
                         roa=round(roa, 5), sales_growth=round(growth, 4), tangibility=round(tang, 4),
                         esg_score=round(esg, 1), cost_of_debt=round(kd, 5), book_to_market=round(bm, 4),
                         div_payer=div))
corp = pd.DataFrame(rows)
# industry median leverage of OTHER firms (instrument)
med = corp.groupby(["industry", "year"])["leverage"].transform(lambda s: (s.sum() - s) / (len(s) - 1))
corp["ind_leverage"] = med.round(4)
# realistic data issues: a few missing values and extreme values
idx = rng.choice(len(corp), 25, replace=False); corp.loc[idx[:15], "esg_score"] = np.nan
corp.loc[idx[15:], "sales_growth"] = np.nan
out = rng.choice(len(corp), 4, replace=False); corp.loc[out, "roa"] = [0.45, -0.38, 0.52, -0.41]
corp.to_csv(f"{OUT}/vn_corp_panel.csv", index=False)

# ---------------------------------------------------------------- 2. bank panel
nb, byears = 27, np.arange(2010, 2025)
rows = []
for b in range(nb):
    state = b < 4
    alpha = rng.normal(0, 0.006)               # bank-specific NPL level
    size0 = rng.normal(19.5, 1.0) + (1.0 if state else 0)
    cg_hist = []
    for t, y in enumerate(byears):
        cg = np.clip(0.15 + 0.08 * (y in (2010, 2016, 2017)) - 0.05 * (y in (2012, 2020, 2023))
                     + rng.normal(0, 0.07) - 0.02 * state, -0.1, 0.6)
        cg_hist.append(cg)
        lag2 = cg_hist[t - 2] if t >= 2 else 0.15
        car = np.clip(rng.normal(0.115, 0.015) - 0.01 * state, 0.08, 0.18)
        size = size0 + 0.12 * t + rng.normal(0, 0.03)
        npl = np.clip(0.022 + alpha + 0.035 * (lag2 - 0.15) - 0.05 * (car - 0.115)
                      + 0.012 * (y in (2012, 2013)) + 0.008 * (y in (2023, 2024)) + rng.normal(0, 0.004), 0.003, 0.09)
        roa = np.clip(0.011 - 0.25 * (npl - 0.022) + 0.002 * (size - 20) + rng.normal(0, 0.003), -0.01, 0.035)
        nim = np.clip(rng.normal(0.034, 0.008) + 0.004 * (not state), 0.01, 0.08)
        green = np.nan if y < 2017 else np.clip(0.01 + 0.004 * (y - 2017) + rng.normal(0, 0.01) + 0.01 * state, 0, 0.2)
        rows.append(dict(bank_id=f"B{b+1:02d}", year=y, state_owned=int(state), size=round(size, 4),
                         credit_growth=round(cg, 4), npl=round(npl, 5), roa=round(roa, 5), nim=round(nim, 4),
                         car=round(car, 4), green_credit_share=None if np.isnan(green) else round(green, 4)))
banks = pd.DataFrame(rows)
banks.to_csv(f"{OUT}/vn_bank_panel.csv", index=False)

# ---------------------------------------------------------------- 3. monthly stock returns
dates = pd.period_range("2015-01", "2024-12", freq="M")
T = len(dates)
rf = np.clip(0.004 + 0.0015 * np.sin(np.arange(T) / 18) + rng.normal(0, 0.0003, T), 0.001, 0.008)
mkt_rf = rng.normal(0.009, 0.055, T); mkt_rf[62:64] = [-0.24, 0.18]   # Mar–Apr 2020
smb = rng.normal(0.002, 0.03, T); hml = rng.normal(0.003, 0.028, T)
rows = []
ns = 80
for s in range(ns):
    beta = rng.uniform(0.5, 1.6); bs = rng.normal(0.3, 0.4); bh = rng.normal(0.2, 0.4)
    lsize = rng.normal(29, 1.5); lbm = rng.normal(0, 0.5)
    for t in range(T):
        size_t = lsize + rng.normal(0, 0.05)
        bm_t = np.exp(lbm + rng.normal(0, 0.05))
        # small and value stocks earn a premium beyond factor loadings (for Fama–MacBeth)
        alpha = -0.0015 * (size_t - 29) + 0.002 * np.log(bm_t)
        e = rng.standard_t(4) * 0.06
        r = rf[t] + alpha + beta * mkt_rf[t] + bs * smb[t] + bh * hml[t] + e
        rows.append(dict(date=str(dates[t]), ticker=f"S{s+1:02d}", ret=round(r, 5),
                         log_mcap=round(size_t, 4), book_to_market=round(bm_t, 4)))
stk = pd.DataFrame(rows)
fac = pd.DataFrame(dict(date=[str(d) for d in dates], rf=rf.round(5), mkt_rf=mkt_rf.round(5),
                        smb=smb.round(5), hml=hml.round(5)))
stk = stk.merge(fac, on="date")
stk.to_csv(f"{OUT}/vn_stock_returns_monthly.csv", index=False)

# ---------------------------------------------------------------- 4. quarterly macro (VAR DGP)
q = pd.period_range("2005Q1", "2024Q4", freq="Q"); n = len(q)
A1 = np.array([[0.45, -0.05, -0.10, 0.04], [0.10, 0.70, 0.05, 0.03], [0.05, 0.30, 0.75, 0.00], [0.20, -0.10, -0.15, 0.55]])
A2 = np.array([[0.15, 0.00, -0.05, 0.02], [0.00, 0.05, 0.00, 0.02], [0.00, 0.05, 0.10, 0.00], [0.05, 0.00, -0.05, 0.10]])
mu = np.array([6.3, 5.5, 5.5, 15.0])           # gdp growth, inflation, policy rate, credit growth (% y/y)
sig = np.array([0.6, 0.9, 0.4, 2.0])
Y = np.zeros((n + 20, 4)); Y[:2] = mu
c = (np.eye(4) - A1 - A2) @ mu
for t in range(2, n + 20):
    Y[t] = c + A1 @ Y[t - 1] + A2 @ Y[t - 2] + rng.normal(0, sig)
Y = Y[20:]
Y[60:62, 0] -= [3.5, 5.0]                      # 2020 Q1–Q2 shock
Y[13:16, 1] += [8, 12, 9]                      # 2008 inflation spike
vni = 1.2 + 0.8 * (Y[:, 0] - 6.3) - 0.9 * (Y[:, 2] - 5.5) + rng.normal(0, 9, n); vni[60] = -22
macro = pd.DataFrame(dict(quarter=[str(x) for x in q], gdp_growth=Y[:, 0].round(2), inflation=Y[:, 1].round(2),
                          policy_rate=Y[:, 2].round(2), credit_growth=Y[:, 3].round(2), vnindex_return=vni.round(2)))
macro["vnindex_level"] = (500 * np.cumprod(1 + macro.vnindex_return / 100)).round(1)
macro.to_csv(f"{OUT}/vn_macro_quarterly.csv", index=False)

# ---------------------------------------------------------------- 5. earnings events
ne = 400; days = np.arange(-150, 11)
rows = []
for e in range(ne):
    sue = rng.normal(0, 1.2); ff = np.clip(rng.beta(3, 4), 0.05, 0.95)
    size = rng.normal(29, 1.3); beta = rng.uniform(0.6, 1.4); alpha = rng.normal(0, 0.0003)
    tick = f"S{rng.integers(1, 81):02d}"; qtr = f"{rng.integers(2022, 2026)}Q{rng.integers(1, 5)}"
    jump = 0.012 * sue * (0.4 + 1.2 * ff)       # reaction stronger when free float is higher
    for d in days:
        m = rng.normal(0.0004, 0.013)
        ar = rng.standard_t(4) * 0.014
        if d == 0: ar += 0.6 * jump
        if d == 1: ar += 0.4 * jump
        if d > 1: ar += 0.0012 * sue            # small post-announcement drift
        r = np.clip(alpha + beta * m + ar, -0.07, 0.07)   # HOSE ±7% price limit
        rows.append(dict(event_id=e + 1, ticker=tick, quarter=qtr, day=int(d), ret=round(r, 5),
                         mkt_ret=round(m, 5), sue=round(sue, 4), free_float=round(ff, 4), log_mcap=round(size, 4)))
ev = pd.DataFrame(rows)
ev.to_csv(f"{OUT}/vn_earnings_events.csv", index=False)
for f in os.listdir(OUT):
    print(f, pd.read_csv(f"{OUT}/{f}").shape)
