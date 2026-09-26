"""Strict validation of generic holiday handling on 2025 May/June holidays (fit before, forecast holiday days only)."""
import copy, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import engine as EN
from engine import Engine
from train import load_hourly
hourly, _ = load_hourly(); hourly["date"] = pd.to_datetime(hourly["date"])
truth = Engine.from_files(hourly[hourly.date <= "2025-10-31"])
CASES = [("2025-04-29", ["2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04"]),
         ("2025-05-06", ["2025-05-08", "2025-05-09", "2025-05-10", "2025-05-11"]),
         ("2025-06-10", ["2025-06-12", "2025-06-13", "2025-06-14", "2025-06-15"])]
orig_rule = Engine.day_rule
def make_rule(kind_dow, factor):
    def rule(self, d):
        if d in self.cal.holidays: return "weekend", kind_dow(d), factor(d)
        return orig_rule(self, d)
    return rule
VARIANTS = {"Sunday x1.0": (lambda d: 6, lambda d: 1.0),
            "Sunday x0.95": (lambda d: 6, lambda d: 0.95),
            "Sunday x0.85": (lambda d: 6, lambda d: 0.85),
            "Saturday x1.0": (lambda d: 5, lambda d: 1.0),
            "Sunday x0.9": (lambda d: 6, lambda d: 0.9),
            "own weekend dow, weekday->Sunday": (lambda d: d.dayofweek if d.dayofweek >= 5 else 6, lambda d: 1.0),
            "weekend-in-holiday x0.85, weekday-holiday Sunday x1.0": (lambda d: d.dayofweek if d.dayofweek >= 5 else 6, lambda d: 0.85 if d.dayofweek >= 5 else 1.0)}
import itertools
for Q, (name, (kd, fac)) in itertools.product((0.0, 0.5), [(k, v) for k, v in VARIANTS.items() if k.startswith("Sunday")]):
    name = f"shape Sat-weight {Q} | {name}"
    Engine.day_rule = make_rule(kd, fac); num = den = 0; per = []
    for ref_end, days in CASES:
        E = Engine.from_files(hourly[hourly.date <= ref_end]); E.cfg = copy.deepcopy(E.cfg); E.cfg["reference_end"] = ref_end
        E.cfg["winter_shape_alpha_by_month"] = {}; E.cfg["holiday_shape_sat_weight"] = Q; E.set_history(hourly[hourly.date <= ref_end]); E.fit()
        dates = pd.DatetimeIndex(days); P = E.predict(dates); Y = truth.Y[:, [truth.dates.get_loc(d) for d in dates]]
        n = np.abs(Y - P).sum(); dd = Y.sum(); num += n; den += dd; per.append(round(1 - n / dd, 3))
    print("%-55s holiday-days score %.4f  per block %s" % (name, 1 - num / den, per))
Engine.day_rule = orig_rule
