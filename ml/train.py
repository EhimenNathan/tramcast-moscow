"""Stage 3 — fit the model and publish immutable artifacts for the API + the competition submission.

python ml/train.py                      # uses data/hourly_boardings.parquet (from ingest.py) or official labels
Outputs: artifacts/cube.parquet, artifacts/meta.json, artifacts/stops.json, submissions/submission.csv
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from engine import Engine, ROOT, cls4
from hybrid import hybridize

ART = ROOT / "artifacts"; ART.mkdir(exist_ok=True)
FC_START, FC_END, YEAR_END = pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), pd.Timestamp("2026-12-31")


def load_hourly():
    p = ROOT / "data/hourly_boardings.parquet"
    if p.exists(): return pd.read_parquet(p), "ingest"
    lab = ROOT / "data/labels"
    return pd.concat([pd.read_csv(lab / f"labels_day_{s}.csv", sep=";") for s in ("train", "test")]), "labels"


def month_indices(E: Engine):
    """Observed 2025 seasonality (month level / October level): per-route ratio, median across stable routes
    (robust to a single route's disruption, e.g. route 17 weekends in April)."""
    D = E.Y.sum(2); dts = E.dates; ok = np.array([not E.cal.is_irregular(d) for d in dts])
    out = []
    for weekend in (False, True):
        sel = ok & ((dts.dayofweek >= 5) == weekend); ratios = []
        for i, r in enumerate(E.routes):
            if r in (5, 7, 50): continue
            m = pd.Series(D[i, sel], dts[sel]).groupby(lambda d: d.month).mean(); ratios.append(m / m[10])
        med = pd.concat(ratios, axis=1).median(axis=1)
        out.append({str(m): round(float(med[m]), 4) for m in range(1, 11)})
    return out[0], out[1]


STOP_PRIOR = {"interchange_bonus": 1.5, "terminal_bonus": 1.0, "peak_asymmetry": 0.6, "center": (55.7558, 37.6173), "radius_km": 15.0,
              "morning_hours": [6, 7, 8, 9], "evening_hours": [16, 17, 18, 19]}
INTERCHANGE = ("метро", "мцк", "мцд", "вокзал", "платформа", "станция", "ж/д")


def _stop_shares(stops):
    """Prior spatial model of boardings along a route (no per-stop data exists): base weight (interchange/terminal)
    x hour modulation (outer stops -> morning peak, central stops -> evening peak). Normalised per hour."""
    P = STOP_PRIOR; n = len(stops); lat0, lon0 = P["center"]
    base = np.ones(n); cen = np.zeros(n)
    for k, st in enumerate(stops):
        nm = (st["name"] or "").lower()
        base[k] += P["interchange_bonus"] * any(w in nm for w in INTERCHANGE) + P["terminal_bonus"] * (k in (0, n - 1))
        dkm = np.hypot((st["lat"] - lat0) * 111.2, (st["lon"] - lon0) * 111.2 * np.cos(np.radians(lat0)))
        cen[k] = 1 - min(dkm / P["radius_km"], 1.0)
    W = np.zeros((24, n))
    for h in range(24):
        mod = 1 + P["peak_asymmetry"] * ((1 - cen) * (h in P["morning_hours"]) + cen * (h in P["evening_hours"]))
        w = base * mod; W[h] = w / w.sum()
    for k, st in enumerate(stops):
        st["share_by_hour"] = [round(float(x), 6) for x in W[:, k]]; st["share"] = round(float(W[:, k].mean()), 6)
        st["centrality"] = round(float(cen[k]), 3); st["weight_base"] = float(base[k])
    return stops


def stops_reference(routes):
    """Geo-binding: organiser GTFS reference where available, OpenStreetMap route relations otherwise."""
    x = ROOT / "data/reference/routes_stops.xlsx"; out = {}
    st = pd.read_excel(x, sheet_name="Порядок_с_координатами")
    names = pd.read_excel(x, sheet_name="Маршруты GTFS_ROUTES", header=1)[["route_short_name", "route_long_name"]]
    for r, g in st.groupby("route_short_name"):
        g = g[g.direction_id == 0]; trip = g.groupby("trip_id").size().idxmax(); g = g[g.trip_id == trip].sort_values("stop_sequence")
        nm = names[names.route_short_name.astype(str) == str(r)].route_long_name
        out[str(int(r))] = {"route": int(r), "name": (nm.iloc[0] if len(nm) else None), "source": "GTFS (справочник организаторов)",
                            "stops": [{"stop_id": str(s.stop_id), "name": s.stop_name, "lat": float(s.stop_lat), "lon": float(s.stop_lon),
                                       "seq": int(s.stop_sequence)} for s in g.itertuples()]}
    osm = ROOT / "data/osm_tram_routes.json"
    if osm.exists():
        d = json.loads(osm.read_text(encoding="utf-8")); nodes = {e["id"]: e for e in d["elements"] if e["type"] == "node"}
        for rel in sorted((e for e in d["elements"] if e["type"] == "relation"), key=lambda e: -len(e["members"])):
            ref = rel.get("tags", {}).get("ref")
            if ref is None or ref in out or int(ref) not in routes: continue
            seq = [nodes[m["ref"]] for m in rel["members"] if m["type"] == "node" and m["role"].startswith("stop") and m["ref"] in nodes]
            if len(seq) < 5: continue
            out[ref] = {"route": int(ref), "name": rel["tags"].get("name"), "source": "OpenStreetMap relation %d (ODbL)" % rel["id"],
                        "stops": [{"stop_id": "osm%d" % n["id"], "name": n.get("tags", {}).get("name"), "lat": n["lat"], "lon": n["lon"], "seq": k + 1}
                                  for k, n in enumerate(seq)]}
    for v in out.values():
        v["stops"] = _stop_shares(v["stops"])
        v["share_method"] = "априорная пространственная модель (пересадки/конечные x асимметрия пиков по удалённости от центра); калибруется по АСКП"
    return out


def main():
    hourly, src = load_hourly(); hourly["date"] = pd.to_datetime(hourly["date"])
    hist = hourly[hourly.date <= pd.Timestamp("2025-10-31")]
    actual_tail = hourly[hourly.date >= FC_START]          # file-boundary tail = real Nov-1 boardings
    E = Engine.from_files(hist).fit()
    # ---- competition horizon (Nov-Dec 2025)
    fc_dates = pd.date_range(FC_START, FC_END)
    P, hyb = hybridize(E, E.predict(fc_dates), fc_dates, E.cfg.get("hybrid_recursive_weight", 0.0))   # direct+recursive level hedge
    P = E.apply_actuals(P, fc_dates, actual_tail)
    # ---- year horizon (2026): scenario using observed seasonal indices
    mwd, mwe = month_indices(E)
    E2 = copy.deepcopy(E); E2.cfg = copy.deepcopy(E.cfg)
    E2.cfg["month_factor_weekday"] = {**mwd, **E.cfg["month_factor_weekday"]}
    E2.cfg["month_factor_weekend"] = {**mwe, **E.cfg["month_factor_weekend"]}
    yr_dates = pd.date_range("2026-01-01", YEAR_END); P2 = E2.predict(yr_dates)
    # ---- cube: history + forecast
    all_dates = pd.date_range(E.dates[0], YEAR_END); n_hist = len(E.dates)
    cube = np.concatenate([E.Y, P, P2], axis=1)
    R, Dn, H = cube.shape
    rr, dd, hh = np.meshgrid(np.array(E.routes), np.arange(Dn), np.arange(H), indexing="ij")
    pq.write_table(pa.table({"route": rr.ravel().astype(np.int16), "day": dd.ravel().astype(np.int16),
                             "hour": hh.ravel().astype(np.int8), "value": cube.ravel().astype(np.float32)}),
                   ART / "cube.parquet", compression="zstd")
    precip = E.weather.reindex(all_dates).fillna(E.cfg["precip_ref_mm"]).round(2).tolist()
    sup = ROOT / "data/daily_supply.parquet"; supply = {}
    if sup.exists():
        s = pd.read_parquet(sup); s = s[pd.to_datetime(s.date).between("2025-10-01", "2025-10-31") & (pd.to_datetime(s.date).dt.dayofweek < 5)]
        supply = {int(r): {"vehicles_median_weekday_oct": float(g.vehicles.median()), "exits_median_weekday_oct": float(g.exits.median())}
                  for r, g in s.groupby("route")}
    hs = ROOT / "data/hourly_supply.parquet"
    if hs.exists():   # trams in service per hour (distinct garage_number with >=1 boarding), median over October
        h = pd.read_parquet(hs); h["date"] = pd.to_datetime(h.date); h = h[h.date.between("2025-10-01", "2025-10-31")]
        full = pd.MultiIndex.from_product([sorted(h.route.unique()), sorted(h.date.unique()), range(24)], names=["route", "date", "hour"])
        h = h.set_index(["route", "date", "hour"]).vehicles.reindex(full, fill_value=0).reset_index()
        for r, g in h.groupby("route"):
            wdm = g[g.date.dt.dayofweek < 5].groupby("hour").vehicles.median(); wem = g[g.date.dt.dayofweek >= 5].groupby("hour").vehicles.median()
            supply.setdefault(int(r), {}).update({"vehicles_by_hour_weekday_oct": [float(x) for x in wdm.reindex(range(24), fill_value=0)],
                                                  "vehicles_by_hour_weekend_oct": [float(x) for x in wem.reindex(range(24), fill_value=0)]})
    meta = {"model_version": "tramcast-2.0", "data_source": src, "routes": [int(r) for r in E.routes],
            "first_day": str(all_dates[0].date()), "n_days": int(Dn), "last_day": str(all_dates[-1].date()),
            "forecast_start": str(FC_START.date()), "precip_by_day": precip,
            "precip_beta": E.cfg["precip_beta"], "precip_ref_mm": E.cfg["precip_ref_mm"],
            "hybrid_factors": {f"{r}|{m}|{k}": v for (r, m, k), v in hyb.items()},
            "month_index_weekday": mwd, "month_index_weekend": mwe, "supply": supply,
            "validation": json.loads((ROOT / "artifacts/validation.json").read_text()) if (ROOT / "artifacts/validation.json").exists() else None,
            "domain": {"routes": "маршруты с ≥8 неделями истории в текущем режиме", "horizon_valid": "1–61 день (ноябрь–декабрь 2025)",
                       "horizon_scenario": "2026 — сценарный (сезонные индексы 2025 + календарь 2026)"},
            "external_sources": {"weather": "https://open-meteo.com/en/docs/historical-weather-api",
                                 "calendar": "https://www.consultant.ru/law/ref/calendar/proizvodstvennye/",
                                 "route_changes": "https://newsvostok.ru/dlya-tramvaev-7-i-50-izmeneniya-po-vyhodnym-budut-dejstvovat-do-kontsa-oseni/"}}
    # online-adaptation state: particle-filter posteriors per route x day-group (read by the API on streamed data)
    ps = getattr(E, "pf_state", {})
    if ps:
        np.savez_compressed(ART / "pf_particles.npz", **{f"{r}|{g}": v["particles"].astype(np.float32) for (r, g), v in ps.items()})
        meta["online"] = {"J": 0.35, "nu": 4.0, "precip_beta": E.cfg["precip_beta"], "precip_ref_mm": E.cfg["precip_ref_mm"],
                          "school_break_factor": E.cfg["school_break_factor"],
                          "school_breaks": [[str(a.date()), str(b.date())] for a, b in E.cal.school_breaks],
                          "skip_days": sorted({str(d.date()) for d in E.cal.holidays} | {str(d.date()) for d in E.cal.special}),
                          "filters": {f"{r}|{g}": {"sigma": v["sigma"], "q": v["q"], "h": v["h"], "level0": v["level0"],
                                                   "norm": {str(k): float(x) for k, x in v["norm"].items()}} for (r, g), v in ps.items()}}
    (ART / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (ART / "stops.json").write_text(json.dumps(stops_reference([int(r) for r in E.routes]), ensure_ascii=False), encoding="utf-8")
    # ---- submission (exact competition grid)
    grid = pd.MultiIndex.from_product([sorted(E.routes), fc_dates, range(24)], names=["route", "date", "hour"]).to_frame(index=False)
    ri = {r: i for i, r in enumerate(E.routes)}
    grid["prediction"] = [round(float(P[ri[r], (d - FC_START).days, h]), 2) for r, d, h in grid[["route", "date", "hour"]].itertuples(index=False)]
    grid["date"] = grid.date.dt.strftime("%Y-%m-%d")
    out = ROOT / "submissions"; out.mkdir(exist_ok=True); grid.to_csv(out / "submission.csv", sep=";", index=False)
    print(f"source={src} routes={E.routes} nov-dec total={P.sum():,.0f} year2026 total={P2.sum():,.0f} rows={len(grid)}")


if __name__ == "__main__":
    main()
