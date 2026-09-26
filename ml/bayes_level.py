"""Robust Bayesian level tracking: particle filter with Student-t observations and change-point jumps.

State-space model per route x day-group g (workday / Saturday / Sunday), on de-weathered, school-adjusted,
weekday-normalised log daily totals x_t:
    x_t  = mu_t + eps_t,           eps_t ~ Student-t(nu, sigma)          (outlier days barely move the posterior)
    mu_t = mu_{t-1} + eta_t,       eta_t ~ N(0, q)        w.p. 1-h
                                   eta_t ~ N(0, J^2)      w.p. h         (Bayesian change-point / regime jump)
Bootstrap particle filter (systematic resampling). Hyper-parameters (sigma, q, h) chosen by the pooled
marginal likelihood p(x_1:T) over routes (hierarchical / empirical Bayes). The filter is O(N) per new day, so the
API can update the level posterior online as validations stream in (see engine.update_online).
"""
from __future__ import annotations
import numpy as np

NU, J, N_PART = 4.0, 0.35, 2000


def t_logpdf(r, s, nu=NU):
    from math import lgamma, log, pi
    c = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * pi)
    return c - np.log(s) - (nu + 1) / 2 * np.log1p((r / s) ** 2 / nu)


def particle_filter(x, sigma, q, h, seed=0, n=N_PART, return_particles=False):
    """x: 1-D array of log observations (NaN = missing). Returns (posterior median of mu_T, log-lik, particles)."""
    rng = np.random.default_rng(seed)
    x0 = np.nanmedian(x[: min(5, len(x))])
    mu = x0 + rng.normal(0, 0.1, n); ll = 0.0
    for xt in x:
        jump = rng.random(n) < h
        mu = mu + np.where(jump, rng.normal(0, J, n), rng.normal(0, np.sqrt(q), n))
        if np.isnan(xt): continue
        lw = t_logpdf(xt - mu, sigma); m = lw.max(); w = np.exp(lw - m); sw = w.sum()
        ll += m + np.log(sw / n); w /= sw
        u = (rng.random() + np.arange(n)) / n; idx = np.minimum(np.searchsorted(np.cumsum(w), u), n - 1); mu = mu[idx]
    return (float(np.median(mu)), ll, mu) if return_particles else (float(np.median(mu)), ll)


def robust_sigma(x):
    d = np.diff(x[~np.isnan(x)]); return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) if len(d) > 3 else 0.05


GRID_Q = (1e-5, 5e-5, 2e-4, 8e-4, 3e-3)
GRID_H = (0.005, 0.02, 0.05, 0.1, 0.2)


def fit_hyper(series_list):
    """Empirical-Bayes: pick (q, h) maximising the pooled PF log-likelihood; sigma per series from robust diffs."""
    best = None
    for q in GRID_Q:
        for h in GRID_H:
            ll = sum(particle_filter(x, max(robust_sigma(x), 0.02), q, h, n=500)[1] for x in series_list if np.isfinite(x).sum() >= 6)
            if best is None or ll > best[0]: best = (ll, q, h)
    return {"q": best[1], "h": best[2], "loglik": best[0]}


class OnlineLevel:
    """Streaming posterior of a level: holds particles, one PF step per new observation (O(N))."""
    def __init__(self, particles, sigma, q, h, seed=0):
        self.mu = np.array(particles, float); self.sigma, self.q, self.h = sigma, q, h; self.rng = np.random.default_rng(seed)
    def step(self, xt):
        n = len(self.mu); jump = self.rng.random(n) < self.h
        self.mu = self.mu + np.where(jump, self.rng.normal(0, J, n), self.rng.normal(0, np.sqrt(self.q), n))
        if xt is None or not np.isfinite(xt): return self.level
        w = np.exp(t_logpdf(xt - self.mu, self.sigma) - t_logpdf(xt - self.mu, self.sigma).max()); w /= w.sum()
        u = (self.rng.random() + np.arange(n)) / n; self.mu = self.mu[np.minimum(np.searchsorted(np.cumsum(w), u), n - 1)]
        return self.level
    @property
    def level(self): return float(np.exp(np.median(self.mu)))
