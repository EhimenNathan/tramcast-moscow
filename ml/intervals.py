"""Probabilistic forecasting: split-conformal multiplicative prediction intervals.

Residual r = log((y+1)/(p+1)) from out-of-time backtests (artifacts/bt_*.npz from horizon_eval.py),
grouped by (level: hourly cell | daily route total) x hour band (night 0-5 / day 6-23) x horizon bucket.
Quantiles q_a, q_(1-a) -> interval [ (p+1)e^{q_a} - 1 , (p+1)e^{q_(1-a)} - 1 ].
Coverage is verified leave-one-split-out (calibrate on 2 periods, test on the 3rd).
Output: artifacts/intervals.json (used by the API: fields p10 / p90).
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from engine import ROOT

H_BUCKETS = [(0, 14), (14, 35), (35, 400)]
LEVELS = (0.10, 0.90)


def residuals(npz):
    z = np.load(npz); Y, P, ok = z["Y"], z["P"], z["ok"]
    nz = Y.sum((1, 2)) > 0; Y, P = Y[nz][:, ok], P[nz][:, ok]
    h = np.arange(len(ok))[ok]
    out = {}
    for b, (a, c) in enumerate(H_BUCKETS):
        m = (h >= a) & (h < c)
        if m.sum() == 0: continue
        for band, hs in (("night", slice(0, 6)), ("day", slice(6, 24))):
            y, p = Y[:, m, hs].ravel(), P[:, m, hs].ravel()
            out.setdefault(("hour", band, b), []).append(np.log((y + 1) / (p + 1)))
        yd, pd_ = Y[:, m].sum(2).ravel(), P[:, m].sum(2).ravel()
        out.setdefault(("day", "all", b), []).append(np.log((yd + 1) / (pd_ + 1)))
    return out


def merge(ds):
    m = {}
    for d in ds:
        for k, v in d.items(): m.setdefault(k, []).extend(v)
    return {k: np.concatenate(v) for k, v in m.items()}


def main():
    files = sorted((ROOT / "artifacts").glob("bt_*.npz")); R = [residuals(f) for f in files]
    print("splits:", [f.stem for f in files])
    # leave-one-split-out coverage
    report = []
    for k in range(len(R)):
        cal = merge([R[j] for j in range(len(R)) if j != k]); test = R[k]
        for key, v in test.items():
            if key not in cal: continue
            lo, hi = np.quantile(cal[key], LEVELS); t = np.concatenate(v)
            report.append({"test_split": files[k].stem, "group": "|".join(map(str, key)), "n": int(len(t)),
                           "coverage_80": round(float(((t >= lo) & (t <= hi)).mean()), 3),
                           "width_mult": [round(float(np.exp(lo)), 3), round(float(np.exp(hi)), 3)]})
    for r in report: print(r)
    by_split = {}
    for r in report: by_split.setdefault(r["test_split"], []).append((r["coverage_80"], r["n"]))
    summary = {s: round(sum(c * n for c, n in v) / sum(n for _, n in v), 3) for s, v in by_split.items()}
    print("pooled coverage of nominal 80% by held-out split:", summary)
    full = merge(R)
    q = {"|".join(map(str, k)): [float(x) for x in np.quantile(v, LEVELS)] for k, v in full.items()}
    json.dump({"levels": LEVELS, "horizon_buckets_days": H_BUCKETS, "log_ratio_quantiles": q,
               "loso_coverage": summary, "detail": report}, open(ROOT / "artifacts/intervals.json", "w"), indent=1)


if __name__ == "__main__":
    main()
