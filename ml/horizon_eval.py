"""Error accumulation over a 61-day multi-step horizon: DIRECT engine vs RECURSIVE (autoregressive) baseline.

Direct   : every day d is predicted from information available at the origin (no feedback of own forecasts).
Recursive: LightGBM on lagged daily totals (lags 1,2,7,14 + weekday), iterated day by day, feeding its own
           predictions back as lags (the scheme described by the organisers); hourly = daily x same shape.
Output: WAPE-score by forecast week, per split -> artifacts/horizon_eval.json
"""
import copy, json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).parent))
from engine import Engine, ROOT, cls4
from train import load_hourly

SPLITS = [("2025-01-31", "2025-02-01", "2025-03-30", "stable winter (58 d)"),
          ("2025-08-31", "2025-09-01", "2025-10-31", "regime change: summer->autumn + route 7/50 weekend closure (61 d)"),
          ("2025-09-30", "2025-10-01", "2025-10-31", "stable autumn (31 d)")]


def recursive_forecast(E: Engine, dates):
    """Global LightGBM on daily log-totals, iterated with its own predictions as lags."""
    D = E.Y.sum(2); T = D.shape[1]; lags = (1, 2, 7, 14)
    X, y = [], []
    for i in range(len(E.routes)):
        if D[i].sum() == 0: continue
        for t in range(14, T):
            X.append([i, E.dates[t].dayofweek] + [np.log1p(D[i, t - l]) for l in lags]); y.append(np.log1p(D[i, t]))
    m = lgb.train(dict(objective="l1", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, verbose=-1),
                  lgb.Dataset(np.array(X), np.array(y), categorical_feature=[0]), 400)
    out = np.zeros((len(E.routes), len(dates), 24))
    for i, r in enumerate(E.routes):
        hist = list(D[i]); mdl = E.models[r]
        for j, d in enumerate(dates):
            if D[i].sum() == 0: break
            x = [i, d.dayofweek] + [np.log1p(hist[-l]) for l in lags]
            v = float(np.expm1(m.predict(np.array([x]))[0])); hist.append(v)       # feedback of own prediction
            out[i, j] = v * mdl.S[cls4(d.dayofweek)]
    return out


def main():
    hourly, _ = load_hourly(); hourly["date"] = pd.to_datetime(hourly["date"])
    truth = Engine.from_files(hourly[hourly.date <= "2025-10-31"])
    res = []
    for ref_end, a, b, label in SPLITS:
        E = Engine.from_files(hourly[hourly.date <= ref_end]); E.cfg = copy.deepcopy(E.cfg); E.cfg["reference_end"] = ref_end
        if ref_end < "2025-03-01": E.cfg["winter_shape_alpha_by_month"] = {}      # already winter
        E.set_history(hourly[hourly.date <= ref_end]); E.fit()
        dates = pd.date_range(a, b); Pd = E.predict(dates); Pr = recursive_forecast(E, dates)
        ti = [truth.dates.get_loc(d) for d in dates]; Y = truth.Y[:, ti]
        ok = np.array([not truth.cal.is_irregular(d) for d in dates])
        weeks = []
        for w in range(0, len(dates), 7):
            sl = np.zeros(len(dates), bool); sl[w:w + 7] = True; sl &= ok
            s = lambda P: float(1 - np.abs(Y[:, sl] - P[:, sl]).sum() / Y[:, sl].sum())
            bias = lambda P: float(P[:, sl].sum() / Y[:, sl].sum() - 1)
            weeks.append({"week": w // 7 + 1, "direct": round(s(Pd), 4), "recursive": round(s(Pr), 4),
                          "direct_bias": round(bias(Pd), 4), "recursive_bias": round(bias(Pr), 4)})
        tot = lambda P: round(float(1 - np.abs(Y[:, ok] - P[:, ok]).sum() / Y[:, ok].sum()), 4)
        for wb in (0.2, 0.35, 0.5):
            print("   hybrid w_recursive=%.2f -> %.4f" % (wb, tot((1 - wb) * Pd + wb * Pr)))
        from hybrid import hybridize
        for wb in (0.2, 0.35):
            print("   PRODUCTION level-ratio hybrid w=%.2f -> %.4f" % (wb, tot(hybridize(E, Pd, dates, wb)[0])))
        np.savez_compressed(ROOT / f"artifacts/bt_{ref_end}.npz", Y=Y, P=Pd, ok=ok, h=np.arange(len(dates)))
        res.append({"split": label, "fit_until": ref_end, "period": f"{a}..{b}", "direct": tot(Pd), "recursive": tot(Pr), "by_week": weeks})
        print(label, "| direct", tot(Pd), "| recursive", tot(Pr))
        for w in weeks: print("   week %d  direct %.4f (bias %+.3f)   recursive %.4f (bias %+.3f)" % (w["week"], w["direct"], w["direct_bias"], w["recursive"], w["recursive_bias"]))
    # what would the recursive model say for Nov-Dec?
    E = Engine.from_files(hourly[hourly.date <= "2025-10-31"]).fit(); dates = pd.date_range("2025-11-01", "2025-12-31")
    Pd, Pr = E.predict(dates), recursive_forecast(E, dates)
    wd = np.array([d.dayofweek < 5 for d in dates])
    print("Nov-Dec total: direct %.0f  recursive %.0f  (ratio %.3f; weekdays %.3f, weekends %.3f)" % (Pd.sum(), Pr.sum(), Pr.sum() / Pd.sum(), Pr[:, wd].sum() / Pd[:, wd].sum(), Pr[:, ~wd].sum() / Pd[:, ~wd].sum()))
    (ROOT / "artifacts/horizon_eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
