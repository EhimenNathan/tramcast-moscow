"""Strict out-of-time comparison: EWMA+Hampel level vs robust particle-filter level (same everything else)."""
import copy, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from engine import Engine
from train import load_hourly
SPLITS = [("2025-01-31", "2025-02-01", "2025-03-30"), ("2025-02-28", "2025-03-01", "2025-03-30"), ("2025-05-18", "2025-05-19", "2025-06-08"),
          ("2025-09-30", "2025-10-01", "2025-10-31"), ("2025-10-12", "2025-10-13", "2025-10-31")]
hourly, _ = load_hourly(); hourly["date"] = pd.to_datetime(hourly["date"])
truth = Engine.from_files(hourly[hourly.date <= "2025-10-31"])
for method in ("pf",):
    row = []; t0 = time.time(); info = []
    for ref_end, a, b in SPLITS:
        E = Engine.from_files(hourly[hourly.date <= ref_end]); E.cfg = copy.deepcopy(E.cfg); E.cfg["reference_end"] = ref_end
        E.cfg["level_method"] = method
        if ref_end < "2025-04-01": E.cfg["winter_shape_alpha_by_month"] = {}
        E.set_history(hourly[hourly.date <= ref_end]); E.fit(); info.append(getattr(E, "pf_info", None))
        dates = pd.date_range(a, b); P = E.predict(dates)
        ok = np.array([not truth.cal.is_irregular(d) for d in dates]); Y = truth.Y[:, [truth.dates.get_loc(d) for d in dates]]
        row.append(1 - np.abs(Y[:, ok] - P[:, ok]).sum() / Y[:, ok].sum())
    print("%-5s " % method + "  ".join("%.4f" % x for x in row) + "   mean %.4f   (%.0fs)" % (np.mean(row), time.time() - t0), flush=True)
    if method == "pf": print("   fitted hyper-parameters per split:", [{g: (v["q"], v["h"]) for g, v in i.items()} for i in info])
