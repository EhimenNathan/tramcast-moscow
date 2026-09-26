"""Adaptivity test: stream October day by day. After each observed day, forecast the next 7 days.
frozen  = fit once on data <= Sep 30
online  = same fit + particle-filter online update with each streamed day (O(1) per day, no refit)
refit   = full re-fit every day (upper-bound reference, ~20-40 s per fit)"""
import copy, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from engine import Engine
from train import load_hourly
hourly, _ = load_hourly(); hourly["date"] = pd.to_datetime(hourly["date"])
truth = Engine.from_files(hourly[hourly.date <= "2025-10-31"]); D = truth.Y.sum(2)
def fit(ref_end):
    E = Engine.from_files(hourly[hourly.date <= ref_end]); E.cfg = copy.deepcopy(E.cfg); E.cfg["reference_end"] = ref_end
    E.set_history(hourly[hourly.date <= ref_end]); return E.fit()
E0 = fit("2025-09-30"); F = E0.online_filters()
days = pd.date_range("2025-10-01", "2025-10-31"); acc = {"frozen": [0, 0], "online": [0, 0], "refit": [0, 0]}
mult = {(r, g): 1.0 for (r, g) in F}
t0 = time.time()
for k, d in enumerate(days[:-1]):
    # 1) observe day d
    for i, r in enumerate(E0.routes):
        if (r, "wd") in F or (r, "sat") in F or (r, "sun") in F:
            o = E0.online_observe(F, r, d, D[i, truth.dates.get_loc(d)])
            if o: mult[(r, o[0])] = o[1]
    # 2) forecast next 7 days
    fut = days[k + 1: k + 8]; ok = np.array([not truth.cal.is_irregular(x) for x in fut]); fut = fut[ok]
    if len(fut) == 0: continue
    Y = truth.Y[:, [truth.dates.get_loc(x) for x in fut]]
    P0 = E0.predict(fut); P1 = P0.copy()
    for i, r in enumerate(E0.routes):
        for j, x in enumerate(fut):
            g = "wd" if x.dayofweek < 5 else ("sat" if x.dayofweek == 5 else "sun")
            P1[i, j] *= mult.get((r, g), 1.0)
    for n, P in (("frozen", P0), ("online", P1)):
        acc[n][0] += np.abs(Y - P).sum(); acc[n][1] += Y.sum()
    if k % 3 == 0:     # full refit every 3rd day (expensive reference)
        Er = fit(str(d.date())); Pr = Er.predict(fut); acc["refit"][0] += np.abs(Y - Pr).sum(); acc["refit"][1] += Y.sum()
        acc.setdefault("frozen_sub", [0, 0]); acc["frozen_sub"][0] += np.abs(Y - P0).sum(); acc["frozen_sub"][1] += Y.sum()
        acc.setdefault("online_sub", [0, 0]); acc["online_sub"][0] += np.abs(Y - P1).sum(); acc["online_sub"][1] += Y.sum()
for n, (a, b) in acc.items(): print("%-11s next-7-day WAPE-score %.4f" % (n, 1 - a / b))
print("elapsed %.0fs" % (time.time() - t0))
