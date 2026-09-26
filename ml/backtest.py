"""Stage 4 — out-of-time validation of the production engine (fit strictly before the origin).
python ml/backtest.py  -> artifacts/validation.json"""
import copy, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from engine import Engine, ROOT
from train import load_hourly

SPLITS = [("2025-09-30", "2025-10-01", "2025-10-31"), ("2025-10-12", "2025-10-13", "2025-10-31"), ("2025-02-28", "2025-03-01", "2025-03-30")]

def wape_score(y, p): return float(1 - np.abs(y - p).sum() / y.sum())

def main():
    hourly, src = load_hourly(); hourly["date"] = pd.to_datetime(hourly["date"])
    full = Engine.from_files(hourly[hourly.date <= "2025-10-31"])            # cube for truth
    res = []
    for ref_end, a, b in SPLITS:
        E = Engine.from_files(hourly[hourly.date <= ref_end]); E.cfg = copy.deepcopy(E.cfg); E.cfg["reference_end"] = ref_end
        E.cfg["weekday_window_days"] = min(E.cfg["weekday_window_days"], 30) if ref_end.startswith("2025-09") else E.cfg["weekday_window_days"]
        E.set_history(hourly[hourly.date <= ref_end]); E.fit()
        dates = pd.date_range(a, b); P = E.predict(dates)
        ti = [full.dates.get_loc(d) for d in dates]; Y = full.Y[:, ti]
        ok = np.array([not full.cal.is_irregular(d) for d in dates])
        by_route = {int(r): round(wape_score(Y[i, ok], P[i, ok]), 4) for i, r in enumerate(E.routes) if Y[i].sum() > 0}
        res.append({"fit_until": ref_end, "forecast": f"{a}..{b}", "wape_score": round(wape_score(Y[:, ok], P[:, ok]), 4),
                    "bias": round(float(P[:, ok].sum() / Y[:, ok].sum() - 1), 4), "by_route": by_route})
        print(res[-1])
    out = {"metric": "WAPE-score = 1 - sum|y-p|/sum y (hourly, route x date x hour)", "splits": res,
           "noise_ceiling": 0.92, "note": "ceiling = score of a perfect route x weekday x hour expectation (irreducible hourly noise)"}
    (ROOT / "artifacts/validation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
