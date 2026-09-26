"""How much would perfect block levels be worth?  (out-of-time splits; hybrid-free direct engine)
Rescale forecasts so each block's total matches the truth, for several block granularities."""
import copy, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from engine import Engine
from train import load_hourly
hourly, _ = load_hourly(); hourly["date"] = pd.to_datetime(hourly["date"])
truth = Engine.from_files(hourly[hourly.date <= "2025-10-31"])
for ref_end, a, b in [("2025-01-31", "2025-02-01", "2025-03-30"), ("2025-08-31", "2025-09-01", "2025-10-31"), ("2025-09-30", "2025-10-01", "2025-10-31")]:
    E = Engine.from_files(hourly[hourly.date <= ref_end]); E.cfg = copy.deepcopy(E.cfg); E.cfg["reference_end"] = ref_end
    if ref_end < "2025-03-01": E.cfg["winter_shape_alpha_by_month"] = {}
    E.set_history(hourly[hourly.date <= ref_end]); E.fit()
    dates = pd.date_range(a, b); P = E.predict(dates); Y = truth.Y[:, [truth.dates.get_loc(d) for d in dates]]
    sc = lambda Q: 1 - np.abs(Y - Q).sum() / Y.sum()
    month = dates.month.values; wk = (dates.dayofweek >= 5).astype(int)
    def rescale(keyfun):
        Q = P.copy()
        for i in range(len(E.routes)):
            keys = keyfun(i)
            for k in set(keys):
                m = keys == k
                if P[i, m].sum() > 0: Q[i, m] *= Y[i, m].sum() / P[i, m].sum()
        return Q
    g = sc(P) * 1.0
    glob = P * Y.sum() / P.sum()
    print(f"{a}..{b}: model {g:.4f} | global scale {sc(glob):.4f} | route {sc(rescale(lambda i: np.zeros(len(dates),int))):.4f} "
          f"| route x month {sc(rescale(lambda i: month)):.4f} | route x month x weekend {sc(rescale(lambda i: month*2+wk)):.4f} "
          f"| route x week {sc(rescale(lambda i: (np.arange(len(dates))//7))):.4f}")
