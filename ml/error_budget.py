"""Error budget: where does the remaining error live?  (out-of-time splits, production engine)

A  model as is
B  + oracle LEVEL: rescale each route x day-type (workday/Sat/Sun) to its true mean over the horizon
C  + oracle DAILY totals: every route-day rescaled to its true total (only the hourly shape remains)
Gaps: A->B = level/drift error, B->C = day-to-day fluctuation, 1-C = hourly-shape + noise.
"""
import copy, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from engine import Engine
from train import load_hourly

SPLITS = [("2025-01-31", "2025-02-01", "2025-03-30"), ("2025-09-30", "2025-10-01", "2025-10-31"), ("2025-10-12", "2025-10-13", "2025-10-31")]


def score(Y, P): return 1 - np.abs(Y - P).sum() / Y.sum()


def main():
    hourly, _ = load_hourly(); hourly["date"] = pd.to_datetime(hourly["date"])
    truth = Engine.from_files(hourly[hourly.date <= "2025-10-31"])
    for ref_end, a, b in SPLITS:
        E = Engine.from_files(hourly[hourly.date <= ref_end]); E.cfg = copy.deepcopy(E.cfg); E.cfg["reference_end"] = ref_end
        if ref_end < "2025-03-01": E.cfg["winter_shape_alpha_by_month"] = {}
        E.set_history(hourly[hourly.date <= ref_end]); E.fit()
        dates = pd.date_range(a, b); P = E.predict(dates)
        ok = np.array([not truth.cal.is_irregular(d) for d in dates]); dates = dates[ok]; P = P[:, ok]
        Y = truth.Y[:, [truth.dates.get_loc(d) for d in dates]]
        dtype = np.array([0 if d.dayofweek < 5 else d.dayofweek - 4 for d in dates])
        PB = P.copy(); PC = P.copy()
        for i in range(len(E.routes)):
            for k in range(3):
                m = dtype == k
                if P[i, m].sum() > 0: PB[i, m] *= Y[i, m].sum() / P[i, m].sum()
            ds = P[i].sum(1); PC[i] = P[i] * (Y[i].sum(1) / np.maximum(ds, 1e-9))[:, None]
        A, B, C = score(Y, P), score(Y, PB), score(Y, PC)
        # D: oracle daily totals x oracle class shape measured IN the target period (floor for class-shape models)
        cls = np.array([0 if d.dayofweek < 4 else d.dayofweek - 3 for d in dates]); PD = np.zeros_like(P)
        for i in range(len(E.routes)):
            for k in range(4):
                m = cls == k
                if m.any() and Y[i, m].sum() > 0:
                    sh = Y[i, m].sum(0) / Y[i, m].sum(); PD[i, m] = Y[i, m].sum(1)[:, None] * sh
        print(f"   floor (oracle daily + oracle in-period class shape): {score(Y, PD):.4f}  -> shape-estimation gap {score(Y, PD) - C:.4f}")
        per = {int(r): (round(score(Y[i], P[i]), 3), round(score(Y[i], PB[i]), 3)) for i, r in enumerate(E.routes) if Y[i].sum() > 0}
        print(f"{a}..{b}: model {A:.4f} | +oracle level {B:.4f} (level gap {B-A:.4f}) | +oracle daily {C:.4f} (day-to-day gap {C-B:.4f}) | shape+noise {1-C:.4f}")
        print("   per route (model, oracle-level):", per)


if __name__ == "__main__":
    main()
