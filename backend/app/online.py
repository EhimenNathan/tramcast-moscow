"""Online Bayesian adaptation in the API (numpy only).

For every COMPLETE streamed day (a date strictly before the latest streamed date) and route, the daily total is
de-weathered, school-adjusted and weekday-normalised, then fed as one step to the route x day-group particle
filter (Student-t observation, random walk + change-point jumps) restored from artifacts/pf_particles.npz.
The posterior level / fitted level = multiplier applied to all later forecast days of that group.
Deterministic (fixed seeds) -> every replica computes identical multipliers from the same shared stream."""
from __future__ import annotations
from datetime import date
from math import lgamma, log, pi
import numpy as np


def _t_logpdf(r, s, nu):
    c = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * pi)
    return c - np.log(s) - (nu + 1) / 2 * np.log1p((r / s) ** 2 / nu)


def multipliers(meta_online, particles, daily, precip_by_day, d0):
    """daily: {(route, 'YYYY-MM-DD'): total} for complete days. Returns ({(route, group): mult}, last_day)."""
    if not meta_online or not daily: return {}, None
    J, nu = meta_online["J"], meta_online["nu"]; skip = set(meta_online["skip_days"])
    breaks = [(date.fromisoformat(a), date.fromisoformat(b)) for a, b in meta_online["school_breaks"]]
    out = {}; last = None
    for key, f in meta_online["filters"].items():
        r, g = key.split("|"); r = int(r)
        obs = sorted((ds, v) for (rr, ds), v in daily.items() if rr == r)
        if not obs: continue
        mu = particles[key].astype(np.float64).copy(); rng = np.random.default_rng(abs(hash(key)) % 2**31); n = len(mu)
        for ds, v in obs:
            dd = date.fromisoformat(ds); dw = dd.weekday(); gg = "wd" if dw < 5 else ("sat" if dw == 5 else "sun")
            if gg != g or ds in skip: continue
            i = (dd - d0).days; p = precip_by_day[i] if 0 <= i < len(precip_by_day) else meta_online["precip_ref_mm"]
            sch = meta_online["school_break_factor"] if (dw < 5 and any(a <= dd <= b for a, b in breaks)) else 1.0
            x = np.log(max(v * np.exp(meta_online["precip_beta"] * p) / sch, 1.0) / f["norm"].get(str(dw), 1.0))
            jump = rng.random(n) < f["h"]
            mu = mu + np.where(jump, rng.normal(0, J, n), rng.normal(0, np.sqrt(f["q"]), n))
            lw = _t_logpdf(x - mu, f["sigma"], nu); w = np.exp(lw - lw.max()); w /= w.sum()
            u = (rng.random() + np.arange(n)) / n; mu = mu[np.minimum(np.searchsorted(np.cumsum(w), u), n - 1)]
            last = max(last, dd) if last else dd
        out[(r, g)] = float(np.exp(np.median(mu)) / f["level0"])
    return out, last
