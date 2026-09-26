"""Stage 2/3 — features & forecasting engine.

Model:  y(route, day, hour) = Level(route, day-type) x Shape(route, day-class, hour | season)
                               x Calendar(day) x Weather(day) x Scenario(route, day)

* Level   — recency-weighted (exponential half-life) mean of de-weathered, school-break-adjusted,
            Hampel-cleaned daily totals; pooled weekday factors (Mon 0.97 ...).
            Closed-service day-types use the median (WAPE-optimal under absolute loss).
* Shape   — last N full weeks per day-class (Mon-Thu / Fri / Sat / Sun), blended with the
            winter reference shape by month ("seasonal shape interpolation", validated +0.002).
* Calendar— RF production calendar: holidays behave like Sundays x factor; working Saturday;
            New Year's Eve; pre-New-Year days (config/calendar_ru.json).
* Weather — exp(-beta * (precip_mm - ref)); beta estimated by robust regression (README).
* Scenario— route regime changes from external sources (config/regimes.json).
* Nowcast — any observed actuals inside the horizon override the forecast (streaming ingest).

Everything is plain numpy: fitting 10 routes takes < 1 s, so the service can refit on new data.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))


def cls4(dw: int) -> int:           # day-class for shapes: Mon-Thu, Fri, Sat, Sun
    return 0 if dw < 4 else dw - 3


@dataclass
class Calendar:
    holidays: set
    working_weekends: set
    pre_holiday: set
    school_breaks: list
    special: dict
    ranges: list

    @classmethod
    def load(cls, path=ROOT / "config/calendar_ru.json"):
        c = load_json(path)
        return cls(set(pd.to_datetime(c["holidays"])), set(pd.to_datetime(c["working_weekends"])),
                   set(pd.to_datetime(c["pre_holiday"])),
                   [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in c["school_breaks"]],
                   {pd.Timestamp(k): v for k, v in c["special_factors"].items()}, c["range_factors"])

    def is_irregular(self, d):      # excluded from level/shape estimation
        return d in self.holidays or d in self.working_weekends or d in self.pre_holiday

    def in_school_break(self, d):
        return any(a <= d <= b for a, b in self.school_breaks)


@dataclass
class RouteModel:
    L: np.ndarray                    # level by day-of-week (7)
    S: np.ndarray                    # shape by class (4 x 24)
    SW: np.ndarray                   # winter reference shape (4 x 24)
    Lr: tuple | None = None          # restored weekend levels (Sat, Sun)
    Sr: tuple | None = None          # restored weekend shapes


@dataclass
class Engine:
    cfg: dict
    cal: Calendar
    regimes: dict
    weather: pd.Series               # daily daytime precipitation (mm), indexed by date
    routes: list = field(default_factory=list)
    models: dict = field(default_factory=dict)
    dates: pd.DatetimeIndex | None = None
    Y: np.ndarray | None = None      # observed cube R x T x 24

    # ------------------------------------------------------------------ data
    @classmethod
    def from_files(cls, hourly: pd.DataFrame, weather_csv=ROOT / "data/weather_daily.csv"):
        w = pd.read_csv(weather_csv, parse_dates=["date"]).set_index("date")["p"]
        e = cls(load_json(ROOT / "config/model.json"), Calendar.load(), load_json(ROOT / "config/regimes.json"), w)
        e.set_history(hourly); return e

    def set_history(self, hourly: pd.DataFrame):
        h = hourly.copy(); h["date"] = pd.to_datetime(h["date"])
        self.routes = sorted(set(h.route.unique()) | set(self.regimes.get("always_zero_routes", [])))
        self.dates = pd.date_range(h.date.min(), pd.Timestamp(self.cfg["reference_end"]))
        idx = pd.MultiIndex.from_product([self.routes, self.dates, range(24)])
        cube = h.set_index(["route", "date", "hour"]).boardings.reindex(idx, fill_value=0)
        self.Y = cube.values.reshape(len(self.routes), len(self.dates), 24).astype(np.float64)

    # --------------------------------------------------------------- fitting
    def _t(self, d):   # date -> index, clipped to the available history (robust to short histories)
        d = min(max(pd.Timestamp(d), self.dates[0]), self.dates[-1]); return self.dates.get_loc(d)

    def _clean_days(self, i, days):
        """Hampel filter on daily totals within each day-of-week group: drop disruption outliers."""
        D = self.Y[i, days].sum(1); keep = np.ones(len(days), bool); k = self.cfg["hampel_k"]
        dows = self.dates[days].dayofweek.values
        for dw in range(7):
            m = dows == dw
            if m.sum() >= 4:
                x = D[m]; med = np.median(x); mad = 1.4826 * np.median(np.abs(x - med)) + 1e-9
                z = (x - med) / mad; flag = np.abs(z) > k
                # run-aware: consecutive same-sign deviations are a level shift, not an outlier
                for j in range(len(x)):
                    nb = [q for q in (j - 1, j + 1) if 0 <= q < len(x)]
                    if flag[j] and any(np.sign(z[q]) == np.sign(z[j]) and abs(z[q]) > k / 2 for q in nb):
                        flag[j] = False
                keep[np.where(m)[0][flag]] = False
        return days[keep]

    def fit(self):
        c = self.cfg; end = self._t(c["reference_end"]); self.models = {}
        valid = np.array([not self.cal.is_irregular(d) for d in self.dates])
        wf = np.array(c["weekday_factors_mon_thu"]); wf = wf / wf.mean()
        for i, r in enumerate(self.routes):
            if r in self.regimes.get("always_zero_routes", []):
                self.models[r] = RouteModel(np.zeros(7), np.full((4, 24), 1 / 24), np.full((4, 24), 1 / 24)); continue
            D = self.Y[i].sum(1)
            adj = np.exp(c["precip_beta"] * self.weather.reindex(self.dates).fillna(c["precip_ref_mm"]).values)
            school = np.array([self.cal.in_school_break(d) and d.dayofweek < 5 for d in self.dates])
            Dn = D * adj / np.where(school, c["school_break_factor"], 1.0)
            self._Dn = getattr(self, "_Dn", {}); self._Dn[r] = Dn

            def days(n, pred):
                a = max(0, end - n + 1)
                return np.array([t for t in range(a, end + 1) if valid[t] and pred(self.dates[t].dayofweek)])

            def level(n, hl, pred, median=False):
                u = self._clean_days(i, days(n, pred))
                if median: return float(np.median(D[u]))
                w = 0.5 ** ((end - u) / hl); return float(np.sum(Dn[u] * w) / np.sum(w))

            L = np.zeros(7)
            l_mt = level(c["weekday_window_days"], c["weekday_half_life"], lambda d: d < 4)
            l_mt_long = level(c["weekday_window_days"], 1e9, lambda d: d < 4)
            fri_own = level(61, 1e9, lambda d: d == 4) / level(61, 1e9, lambda d: d < 4)
            L[:4] = l_mt * wf
            L[4] = l_mt * ((1 - c["friday_shrink"]) * fri_own + c["friday_shrink"] * c["friday_pooled"] / np.mean(c["weekday_factors_mon_thu"]))
            for dw in (5, 6):
                L[dw] = level(c["weekend_window_days"], c["weekend_half_life"], lambda d, dw=dw: d == dw)
                if L[dw] < c["closed_service_ratio"] * l_mt:        # closed service: WAPE-optimal median
                    L[dw] = level(c["weekend_window_days"], 1e9, lambda d, dw=dw: d == dw, median=True)
            # shapes: last N full weeks, optionally skipping school-break weeks
            S = np.zeros((4, 24)); SW = np.zeros((4, 24))
            wa, wb = map(self._t, c["winter_shape_window"])
            for k in range(4):
                u = [t for t in days(7 * c["shape_weeks"] + 7, lambda d, k=k: cls4(d) == k)
                     if not (c["shape_exclude_school_break"] and self.cal.in_school_break(self.dates[t]))]
                u = u[-(c["shape_weeks"] * (4 if k == 0 else 1)):]
                s = self.Y[i, u].sum(0); S[k] = s / max(s.sum(), 1)
                uw = [t for t in range(wa, wb + 1) if valid[t] and cls4(self.dates[t].dayofweek) == k]
                s = self.Y[i, uw].sum(0); SW[k] = s / max(s.sum(), 1)
            rm = RouteModel(L, S, SW)
            rm.lv = {"wd": (l_mt, l_mt_long)}
            for dw in (5, 6):
                if L[dw] >= c["closed_service_ratio"] * l_mt:
                    rm.lv[dw] = (L[dw], level(c["weekend_window_days"], 1e9, lambda d, dw=dw: d == dw))
            # restored-weekend reference for routes with a temporary weekend regime
            for reg in self.regimes.get("weekend_service_changes", []):
                if r in reg["routes"]:
                    ra, rb = map(self._t, c["restored_weekend_reference"])
                    ref = [t for t in range(ra, rb + 1) if valid[t]]
                    wd = [t for t in ref if self.dates[t].dayofweek < 5]; k_ = l_mt / D[wd].mean()
                    sat = [t for t in ref if self.dates[t].dayofweek == 5]; sun = [t for t in ref if self.dates[t].dayofweek == 6]
                    rm.Lr = (D[sat].mean() * k_, D[sun].mean() * k_)
                    rm.Sr = tuple(self.Y[i, u].sum(0) / self.Y[i, u].sum() for u in (sat, sun))
            self.models[r] = rm
        self._pool_levels()
        if self.cfg.get("level_method", "ewma") == "pf": self._pf_levels()
        return self

    def _pf_series(self, r, dws, norm):
        c = self.cfg; end = self._t(c["reference_end"]); start = self._t(c.get("pf_history_start", "2025-01-13"))
        valid = np.array([not self.cal.is_irregular(d) for d in self.dates]); Dn = self._Dn[r]; x = []
        for t in range(start, end + 1):
            dw = self.dates[t].dayofweek
            if dw in dws: x.append(np.log(max(Dn[t], 1.0) / norm[dw]) if valid[t] and Dn[t] > 0 else np.nan)
        return np.array(x)

    def _pf_levels(self):
        """Replace EWMA levels by the robust particle-filter posterior (Student-t + change-point jumps)."""
        from bayes_level import particle_filter, robust_sigma, fit_hyper
        groups = {"wd": [0, 1, 2, 3, 4], "sat": [5], "sun": [6]}; self.pf_info = {}
        for g, dws in groups.items():
            ser = {}
            for r, m in self.models.items():
                if m.L.sum() == 0: continue
                if g != "wd" and m.L[dws[0]] < self.cfg["closed_service_ratio"] * m.L[0]: continue   # closed regime: keep median
                norm = {dw: m.L[dw] / m.L[0] for dw in range(7)} if g == "wd" else {dws[0]: 1.0}
                ser[r] = (self._pf_series(r, dws, norm), norm)
            if not ser: continue
            hp = self.cfg.get("pf_hyper", {}).get(g) or fit_hyper([x for x, _ in ser.values()])
            self.pf_info[g] = hp
            for r, (x, norm) in ser.items():
                mu, _, parts = particle_filter(x, max(robust_sigma(x), 0.02), hp["q"], hp["h"], seed=int(r), return_particles=True)
                m = self.models[r]; lvl = float(np.exp(mu))
                self.pf_state = getattr(self, "pf_state", {})
                self.pf_state[(r, g)] = {"particles": parts, "sigma": max(robust_sigma(x), 0.02), "q": hp["q"], "h": hp["h"], "norm": norm,
                                         "level0": lvl}
                if g == "wd": m.L[:5] = m.L[:5] * (lvl / m.L[0])
                else: m.L[dws[0]] = lvl

    def _pool_levels(self):
        """Hierarchical level pooling: route level = own long-window level x NETWORK recent trend (shrinks noisy
        per-route trends of small routes toward the common trend estimated from ~200k boardings/day)."""
        lam = self.cfg.get("level_pool_weight", 0.0)
        if lam <= 0: return
        for key, idx in (("wd", slice(0, 5)), (5, slice(5, 6)), (6, slice(6, 7))):
            pairs = [(r, m.lv[key]) for r, m in self.models.items() if hasattr(m, "lv") and key in m.lv]
            if len(pairs) < 3: continue
            net = sum(p[0] for _, p in pairs) / sum(p[1] for _, p in pairs)
            self.pool_trend = getattr(self, "pool_trend", {}); self.pool_trend[str(key)] = round(net, 4)
            for r, (short, long_) in pairs:
                f = (1 - lam) + lam * long_ * net / short
                self.models[r].L[idx] *= f

    # ------------------------------------------------------------ prediction
    def _restored(self, r, d):
        for reg in self.regimes.get("weekend_service_changes", []):
            if r in reg["routes"] and reg.get("restore") and d >= pd.Timestamp(reg["restore"]):
                return True
        return False

    def day_rule(self, d):
        """-> (kind, dow used for level, factor)"""
        c = self.cfg; cal = self.cal; m = str(d.month); dw = d.dayofweek
        mwd = c["month_factor_weekday"].get(m, 1.0); mwe = c["month_factor_weekend"].get(m, 1.0)
        sp = cal.special.get(d)
        if sp:
            k = sp["kind"]
            if k == "working_saturday": return "working_saturday", 4, sp["factor"] * mwd
            if k == "holiday": return "weekend", 6, sp["factor"] * mwe
            if k == "new_year_eve": return "new_year_eve", 5, sp["factor"] * mwe
            if k == "workday": return "workday", dw, sp["factor"] * mwd
        if d in cal.holidays: return "weekend", 6, mwe * c.get("generic_holiday_factor", 1.0)   # generic holiday = Sunday-like (validated 0.9-0.99)
        f = mwd if dw < 5 else mwe
        for rg in cal.ranges:
            if pd.Timestamp(rg["from"]) <= d <= pd.Timestamp(rg["to"]) and (dw < 5) == (rg["applies"] == "workday"):
                f *= rg["factor"]
        return ("workday" if dw < 5 else "weekend"), dw, f

    def predict(self, dates, corrections=None) -> np.ndarray:
        """R x len(dates) x 24 forecast. corrections: optional {'weather':x,'event':x,'season':x} multipliers."""
        c = self.cfg; out = np.zeros((len(self.routes), len(dates), 24))
        mult = 1.0
        if corrections: mult = float(np.prod([v for v in corrections.values()]))
        for i, r in enumerate(self.routes):
            m = self.models[r]
            for j, d in enumerate(dates):
                kind, dl, f = self.day_rule(d); restored = self._restored(r, d)
                alpha = c["winter_shape_alpha_by_month"].get(str(d.month), 0.0)
                if kind == "workday":
                    lev, sh = m.L[dl], (1 - alpha) * m.S[cls4(dl)] + alpha * m.SW[cls4(dl)]
                elif kind in ("weekend", "new_year_eve"):
                    hol = d in self.cal.holidays and kind == "weekend"
                    q = c.get("holiday_shape_sat_weight", 0.0) if hol else 0.0     # holiday profile = Sat/Sun mix
                    if restored and m.Lr is not None:
                        lev = m.Lr[dl - 5]; sh = (1 - q) * m.Sr[dl - 5] + q * m.Sr[0]
                    else:
                        lev = m.L[dl]; sh = (1 - q) * m.S[cls4(dl)] + q * m.S[2]
                        if lev >= c["closed_service_ratio"] * m.L[0]:
                            sh = (1 - alpha) * sh + alpha * ((1 - q) * m.SW[cls4(dl)] + q * m.SW[2])
                    if kind == "new_year_eve": sh = sh.copy(); sh[21:] *= c["new_year_eve_late_evening_factor"]
                else:  # working Saturday: Friday level, Friday/Saturday shape mix
                    q = c["working_saturday_shape_mix"]
                    base = q * m.S[1] + (1 - q) * m.S[2]; win = q * m.SW[1] + (1 - q) * m.SW[2]
                    lev, sh = m.L[4], (1 - alpha) * base + alpha * win
                sh = sh / max(sh.sum(), 1e-12)
                p = self.weather.get(d, c["precip_ref_mm"])
                wfac = np.exp(-c["precip_beta"] * ((p if np.isfinite(p) else c["precip_ref_mm"]) - c["precip_ref_mm"]))
                out[i, j] = lev * sh * f * wfac * mult
        return np.clip(out, 0, None)

    def online_filters(self):
        from bayes_level import OnlineLevel
        return {k: OnlineLevel(v["particles"], v["sigma"], v["q"], v["h"], seed=hash(k) % 2**31) for k, v in getattr(self, "pf_state", {}).items()}

    def online_observe(self, filters, r, d, daily_total, precip=None):
        """Feed one observed day (route r, date d, total boardings) -> returns (group, level multiplier vs fit)."""
        c = self.cfg; dw = d.dayofweek; g = "wd" if dw < 5 else ("sat" if dw == 5 else "sun")
        if (r, g) not in filters or d in self.cal.holidays or d in self.cal.special: return None
        st = self.pf_state[(r, g)]; p = precip if precip is not None else self.weather.get(d, c["precip_ref_mm"])
        adj = np.exp(c["precip_beta"] * (p if np.isfinite(p) else c["precip_ref_mm"]))
        sch = c["school_break_factor"] if (self.cal.in_school_break(d) and dw < 5) else 1.0
        x = np.log(max(daily_total * adj / sch, 1.0) / st["norm"].get(dw, 1.0))
        lvl = filters[(r, g)].step(x); return g, lvl / st["level0"]

    def apply_actuals(self, P, dates, actuals: pd.DataFrame | None):
        """Nowcast: replace forecast cells by observed values (e.g. streamed validations)."""
        if actuals is None or actuals.empty: return P
        pos = {d: j for j, d in enumerate(dates)}; ri = {r: i for i, r in enumerate(self.routes)}
        for r, d, h, v in actuals[["route", "date", "hour", "boardings"]].itertuples(index=False):
            d = pd.Timestamp(d)
            if d in pos and r in ri: P[ri[r], pos[d], int(h)] = v
        return P
