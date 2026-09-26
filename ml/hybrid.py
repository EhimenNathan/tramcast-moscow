"""Direct + recursive hybrid as a smoothed LEVEL correction (keeps calendar, shapes and scenarios of the engine).

For each route x month x day-type (workday/weekend), ratio = sum(recursive) / sum(direct) over regular days;
factor = 1 + w * (ratio - 1), applied to the direct forecast of that cell group. Skipped for route-days under an
explicit scenario (restored weekend service) and for zero routes. w = 0.2 validated on 3 out-of-time splits
(winter 0.9102->0.9135, regime change 0.8042->0.8107, autumn 0.8992->0.9001)."""
import numpy as np, pandas as pd, lightgbm as lgb
from engine import cls4

LAGS = (1, 2, 7, 14)


def recursive_daily(E, dates):
    """Global LightGBM on log daily totals, iterated with own predictions (autoregressive)."""
    D = E.Y.sum(2); T = D.shape[1]; X, y = [], []
    ok = np.array([not E.cal.is_irregular(d) for d in E.dates])
    for i in range(len(E.routes)):
        if D[i].sum() == 0: continue
        for t in range(14, T):
            if ok[t]: X.append([i, E.dates[t].dayofweek] + [np.log1p(D[i, t - l]) for l in LAGS]); y.append(np.log1p(D[i, t]))
    m = lgb.train(dict(objective="l1", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, verbose=-1, seed=0, deterministic=True),
                  lgb.Dataset(np.array(X), np.array(y), categorical_feature=[0]), 400)
    out = np.zeros((len(E.routes), len(dates)))
    for i in range(len(E.routes)):
        if D[i].sum() == 0: continue
        hist = list(D[i])
        for j, d in enumerate(dates):
            v = float(np.expm1(m.predict(np.array([[i, d.dayofweek] + [np.log1p(hist[-l]) for l in LAGS]]))[0])); hist.append(v); out[i, j] = v
    return out


def hybridize(E, P, dates, w=0.2):
    if w <= 0: return P, {}
    R = recursive_daily(E, dates); Dd = P.sum(2); P = P.copy(); factors = {}
    regular = np.array([not E.cal.is_irregular(d) and d not in E.cal.special for d in dates])
    for i, r in enumerate(E.routes):
        if Dd[i].sum() == 0: continue
        for m in sorted(set(dates.month)):
            for weekend in (False, True):
                sel = (dates.month == m) & ((dates.dayofweek >= 5) == weekend)
                if weekend and any(E._restored(r, d) for d in dates[sel]): continue      # scenario owns these days
                L = E.models[r].L
                if weekend and min(L[5], L[6]) < E.cfg["closed_service_ratio"] * L[0]: continue   # closed-service regime
                reg = sel & regular
                if reg.sum() == 0 or Dd[i, reg].sum() == 0: continue
                ratio = R[i, reg].sum() / Dd[i, reg].sum()
                f = float(np.clip(1 + w * (ratio - 1), 0.9, 1.1))      # hedge, never a regime override
                P[i, sel] *= f; factors[(int(r), int(m), "weekend" if weekend else "workday")] = round(f, 4)
    return P, factors
