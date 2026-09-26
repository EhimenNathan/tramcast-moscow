"""Validate hierarchical level pooling (lambda) on out-of-time splits (direct engine, hourly WAPE-score)."""
import copy, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from engine import Engine
from train import load_hourly
SPLITS = [("2025-01-31", "2025-02-01", "2025-03-30"), ("2025-02-28", "2025-03-01", "2025-03-30"), ("2025-09-30", "2025-10-01", "2025-10-31"),
          ("2025-10-12", "2025-10-13", "2025-10-31"), ("2025-05-18", "2025-05-19", "2025-06-08")]
hourly, _ = load_hourly(); hourly["date"] = pd.to_datetime(hourly["date"])
truth = Engine.from_files(hourly[hourly.date <= "2025-10-31"])
for lam in (0.0, 0.5, 1.0):
    row = []
    for ref_end, a, b in SPLITS:
        E = Engine.from_files(hourly[hourly.date <= ref_end]); E.cfg = copy.deepcopy(E.cfg); E.cfg["reference_end"] = ref_end
        E.cfg["level_pool_weight"] = lam
        if ref_end < "2025-04-01": E.cfg["winter_shape_alpha_by_month"] = {}
        E.set_history(hourly[hourly.date <= ref_end]); E.fit()
        dates = pd.date_range(a, b); P = E.predict(dates)
        ok = np.array([not truth.cal.is_irregular(d) for d in dates]); Y = truth.Y[:, [truth.dates.get_loc(d) for d in dates]]
        row.append(1 - np.abs(Y[:, ok] - P[:, ok]).sum() / Y[:, ok].sum())
    print("lambda=%.1f  " % lam + "  ".join("%.4f" % x for x in row) + "   mean %.4f" % np.mean(row))
