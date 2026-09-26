"""In-memory forecast store: a dense float32 cube [route, day, hour] built from immutable artifacts.

Queries are pure numpy slicing/reduction -> O(days*24) per request, microseconds.
Instances are stateless (read-only artifacts + append-only stream directory on a shared volume),
so any number of replicas can run behind a load balancer (horizontal scaling).
"""
from __future__ import annotations
import json, math, os, threading, time
from datetime import date, datetime, timedelta
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ART = Path(os.getenv("ARTIFACTS_DIR", Path(__file__).resolve().parents[2] / "artifacts"))
STREAM = Path(os.getenv("STREAM_DIR", ART.parent / "data" / "stream"))


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, hint: str | None = None):
        self.status, self.code, self.message, self.hint = status, code, message, hint


def parse_date(s: str | None, name: str) -> date | None:
    if s is None or s == "": return None
    try: return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise ApiError(422, "bad_date", f"Параметр '{name}' должен быть в формате YYYY-MM-DD, получено '{s}'",
                       "Пример: 2025-11-15")


class Store:
    def __init__(self):
        self.lock = threading.Lock(); self.load()

    # ---------------------------------------------------------------- load
    def load(self):
        t = pq.read_table(ART / "cube.parquet")
        routes = t.column("route").to_numpy(); days = t.column("day").to_numpy(); hours = t.column("hour").to_numpy()
        meta = json.loads((ART / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.routes = [int(r) for r in meta["routes"]]
        self.r_idx = {r: i for i, r in enumerate(self.routes)}
        self.d0 = date.fromisoformat(meta["first_day"]); self.n_days = int(meta["n_days"])
        self.forecast_start = date.fromisoformat(meta["forecast_start"])
        self.cube = np.zeros((len(self.routes), self.n_days, 24), np.float32)
        ri = np.array([self.r_idx[int(r)] for r in routes])
        self.cube[ri, days, hours] = t.column("value").to_numpy().astype(np.float32)
        self.is_actual = np.zeros(self.n_days, bool)
        self.is_actual[: (self.forecast_start - self.d0).days] = True
        # weather used by the model per day (for user precipitation override)
        self.precip = np.array(meta["precip_by_day"], np.float32)
        self.beta = float(meta["precip_beta"]); self.p_ref = float(meta["precip_ref_mm"])
        self.stops = json.loads((ART / "stops.json").read_text(encoding="utf-8"))
        iv = ART / "intervals.json"
        self.iv = json.loads(iv.read_text()) if iv.exists() else None       # split-conformal p10/p90 (log-ratio quantiles)
        self.supply = {int(k): v for k, v in meta.get("supply", {}).items()}
        self.day_iso = [(self.d0 + timedelta(days=i)).isoformat() for i in range(self.n_days)]
        self.month_key = np.array([int(x[:4]) * 12 + int(x[5:7]) - 1 for x in self.day_iso])
        pp = ART / "pf_particles.npz"
        self.particles = dict(np.load(pp)) if pp.exists() else {}
        self.online_mult, self.online_last = {}, None
        self.stream_mtime = 0.0; self.stream_cells = {}; self.version = 0
        self.refresh_stream(force=True)

    # --------------------------------------------------------- streaming data
    def refresh_stream(self, force=False):
        """Merge streamed actuals (append-only parquet batches on a shared volume). Cheap mtime check."""
        if not STREAM.exists(): return
        m = STREAM.stat().st_mtime
        if not force and m <= self.stream_mtime: return
        with self.lock:
            cells = {}
            for f in sorted(STREAM.glob("*.parquet")):
                t = pq.read_table(f)
                for r, d, h, v in zip(t.column("route").to_pylist(), t.column("date").to_pylist(),
                                      t.column("hour").to_pylist(), t.column("boardings").to_pylist()):
                    k = (int(r), str(d)[:10], int(h)); cells[k] = cells.get(k, 0) + int(v)
            self.stream_cells = cells; self.stream_mtime = m; self.version += 1
            # online Bayesian adaptation from COMPLETE streamed days (strictly before the latest streamed date)
            from .online import multipliers
            days = {}
            for (r, ds, h), v in cells.items(): days[(r, ds)] = days.get((r, ds), 0) + v
            latest = max((ds for _, ds in days), default=None)
            complete = {k: v for k, v in days.items() if latest and k[1] < latest}
            self.online_mult, self.online_last = multipliers(self.meta.get("online"), self.particles, complete, self.precip, self.d0)

    # ---------------------------------------------------------------- query
    def day_index(self, d: date) -> int:
        return (d - self.d0).days

    def resolve_range(self, horizon: str, d_from: date | None, d_to: date | None):
        if d_from is None:
            d_from = self.forecast_start
        if d_to is None:
            if horizon == "day": d_to = d_from
            elif horizon == "month":
                nxt = (d_from.replace(day=28) + timedelta(days=4)).replace(day=1); d_to = nxt - timedelta(days=1)
            elif horizon == "year": d_to = d_from + timedelta(days=364)
            else: raise ApiError(422, "bad_horizon", f"Неизвестный горизонт '{horizon}'", "Допустимо: day, month, year")
        last = self.d0 + timedelta(days=self.n_days - 1)
        if d_from > d_to: raise ApiError(422, "bad_range", "date_from позже date_to")
        if d_from < self.d0 or d_to > last:
            raise ApiError(422, "out_of_domain", f"Период вне области определения модели: {self.d0} … {last}",
                           "История: 2025-01-01…2025-10-31; прогноз: 2025-11-01…2026-12-31 (2026 — сценарный годовой горизонт)")
        return d_from, d_to

    def series(self, route_ids, d_from, d_to, h_from, h_to, agg, corrections, precip_mm=None, stop_share=None):
        a, b = self.day_index(d_from), self.day_index(d_to) + 1
        if not (0 <= h_from <= h_to <= 23): raise ApiError(422, "bad_hours", "Часы должны удовлетворять 0 ≤ hour_from ≤ hour_to ≤ 23")
        n_days = b - a
        if agg == "hour" and n_days > 93:
            raise ApiError(422, "too_large", "Почасовая детализация доступна для периода до 93 дней",
                           "Используйте agg=day или agg=month для длинных периодов")
        ri = [self.r_idx[r] for r in route_ids]
        X = self.cube[ri, a:b, h_from:h_to + 1].astype(np.float64)           # R x D x H
        # correction coefficients apply to forecast days only (history stays factual)
        fc = ~self.is_actual[a:b]
        k = corrections.get("weather", 1.0) * corrections.get("event", 1.0) * corrections.get("season", 1.0)
        mult = np.where(fc, k, 1.0)
        if precip_mm is not None:
            mult = mult * np.where(fc, np.exp(-self.beta * (precip_mm - self.precip[a:b])), 1.0)
        sv = np.ones(24) if stop_share is None else np.asarray(stop_share, float)   # per-hour spatial share
        X = X * mult[None, :, None] * sv[None, None, h_from:h_to + 1]
        if self.online_mult and self.online_last is not None:          # online level adaptation for later days
            j0 = max((self.online_last - d_from).days + 1, 0)
            if j0 < n_days:
                dws = np.array([(d_from + timedelta(days=j)).weekday() for j in range(n_days)])
                for k, r in enumerate(route_ids):
                    for g, sel in (("wd", dws < 5), ("sat", dws == 5), ("sun", dws == 6)):
                        f = self.online_mult.get((r, g))
                        if f is None: continue
                        m_ = sel.copy(); m_[:j0] = False; m_ &= fc
                        X[k, m_] *= f
        # overlay streamed actuals (nowcast)
        act = self.is_actual[a:b].copy()
        if self.stream_cells:
            for (r, ds, h), v in self.stream_cells.items():
                if r in route_ids and h_from <= h <= h_to:
                    j = (date.fromisoformat(ds) - d_from).days
                    if 0 <= j < n_days: X[route_ids.index(r), j, h - h_from] = v * sv[h]; act[j] = True
        tot = X.sum(0)                                                          # D x H
        iso = self.day_iso[a:b]
        kinds = np.where(act, "actual", "forecast").tolist()
        if agg == "hour":
            H = tot.shape[1]; vals = np.round(tot, 1).tolist()
            bd = self._bounds(tot, a, n_days, "hour", (np.arange(h_from, h_to + 1) >= 6).astype(int))
            if bd is None:
                return [{"date": iso[i], "hour": h_from + h, "value": vals[i][h], "kind": kinds[i]} for i in range(n_days) for h in range(H)]
            lo, hi = np.round(bd[0], 1).tolist(), np.round(bd[1], 1).tolist()
            return [{"date": iso[i], "hour": h_from + h, "value": vals[i][h], "kind": kinds[i],
                     **({"p10": lo[i][h], "p90": hi[i][h]} if kinds[i] == "forecast" else {})} for i in range(n_days) for h in range(H)]
        day_tot = tot.sum(1)
        if agg == "day":
            vals = np.round(day_tot, 1).tolist()
            bd = self._bounds(day_tot[:, None] / max(len(route_ids), 1), a, n_days, "day")
            if bd is None:
                return [{"date": iso[i], "value": vals[i], "kind": kinds[i]} for i in range(n_days)]
            k = max(len(route_ids), 1)   # per-route calibration, comonotone sum over routes (conservative)
            lo, hi = np.round(bd[0][:, 0] * k, 1).tolist(), np.round(bd[1][:, 0] * k, 1).tolist()
            return [{"date": iso[i], "value": vals[i], "kind": kinds[i], **({"p10": lo[i], "p90": hi[i]} if kinds[i] == "forecast" else {})}
                    for i in range(n_days)]
        if agg == "month":
            mk = self.month_key[a:b]; keys, start = np.unique(mk, return_index=True)
            sums = np.add.reduceat(day_tot, start); cnt = np.diff(np.r_[start, n_days])
            anyfc = np.logical_or.reduceat(~act, start)
            return [{"month": f"{k // 12}-{k % 12 + 1:02d}", "value": round(float(v), 1), "days": int(c),
                     "kind": "forecast" if f else "actual"} for k, v, c, f in zip(keys, sums, cnt, anyfc)]
        if agg == "hour_profile":   # mean profile over the period
            prof = np.round(tot.mean(0), 1).tolist()
            return [{"hour": h_from + h, "value": v} for h, v in enumerate(prof)]
        if agg == "total":
            return [{"date_from": d_from.isoformat(), "date_to": d_to.isoformat(), "value": round(float(tot.sum()), 1)}]
        raise ApiError(422, "bad_agg", f"Неизвестная агрегация '{agg}'", "Допустимо: hour, day, month, hour_profile, total")

    def _bounds(self, v, a, n, level, band_of_col=None):
        """p10/p90 for values v (n_days x k) of forecast days; returns (lo, hi) arrays or None."""
        if self.iv is None: return None
        fs = (self.forecast_start - self.d0).days
        h = np.maximum(np.arange(a, a + n) - fs, 0)
        buckets = self.iv["horizon_buckets_days"]; qd = self.iv["log_ratio_quantiles"]
        b = np.select([h < bb[1] for bb in buckets], list(range(len(buckets))), len(buckets) - 1)
        lo = np.empty_like(v); hi = np.empty_like(v)
        for bi in range(len(buckets)):
            rows = b == bi
            if not rows.any(): continue
            if level == "day":
                q = qd.get(f"day|all|{bi}") or qd.get(f"day|all|{len(buckets)-2}")
                lo[rows] = (v[rows] + 1) * np.exp(q[0]) - 1; hi[rows] = (v[rows] + 1) * np.exp(q[1]) - 1
            else:
                for band, cols in (("night", band_of_col == 0), ("day", band_of_col == 1)):
                    q = qd.get(f"hour|{band}|{bi}") or qd.get(f"hour|{band}|{len(buckets)-2}")
                    sub = np.ix_(rows, cols)
                    lo[sub] = (v[sub] + 1) * np.exp(q[0]) - 1; hi[sub] = (v[sub] + 1) * np.exp(q[1]) - 1
        return np.maximum(lo, 0), hi

    def _route_stops(self, route):
        rs = self.stops.get(str(route))
        if not rs: raise ApiError(404, "no_stops", f"Для маршрута {route} нет справочника остановок",
                                  "Справочник покрывает маршруты: " + ", ".join(sorted(self.stops, key=int)))
        return rs

    def stop_share(self, route: int, stop_id: str | None):
        """-> (share vector over 24 hours, stop info). Share = prior spatial weight of the stop in the route's boardings."""
        if stop_id is None: return None, None
        for s in self._route_stops(route)["stops"]:
            if str(s["stop_id"]) == str(stop_id):
                return np.array(s.get("share_by_hour", [s["share"]] * 24)), {k: s[k] for k in ("stop_id", "name", "lat", "lon", "seq")}
        raise ApiError(404, "no_stop", f"Остановка {stop_id} не найдена на маршруте {route}", f"Список: /api/v1/routes/{route}/stops")

    def section_share(self, route: int, stop_from: str, stop_to: str):
        """Section (участок) = contiguous run of stops between two stops (inclusive, either order) -> summed shares."""
        st = self._route_stops(route)["stops"]; ids = [str(s["stop_id"]) for s in st]
        for x in (stop_from, stop_to):
            if str(x) not in ids: raise ApiError(404, "no_stop", f"Остановка {x} не найдена на маршруте {route}", f"Список: /api/v1/routes/{route}/stops")
        i, j = sorted((ids.index(str(stop_from)), ids.index(str(stop_to))))
        seg = st[i:j + 1]; vec = np.sum([s.get("share_by_hour", [s["share"]] * 24) for s in seg], axis=0)
        return vec, {"from": {k: seg[0][k] for k in ("stop_id", "name", "seq")}, "to": {k: seg[-1][k] for k in ("stop_id", "name", "seq")}, "n_stops": len(seg)}
