"""REST/HTTP API — прогноз пассажиропотока трамвайных маршрутов.

Stateless FastAPI service. All forecasts live in an in-memory numpy cube (see store.py).
"""
from __future__ import annotations
import csv, io, math, os, time, uuid
from datetime import datetime
from pathlib import Path
import orjson
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from .store import ApiError, Store, STREAM, parse_date

app = FastAPI(title="Tram Load Forecast API", version="1.0.0",
              description="Прогноз посадок (успешных валидаций) по маршрутам трамвая: час / день / месяц / год")
STORE = Store()
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# ------------------------------------------------------------------ metrics
BUCKETS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
METRICS = {"requests": 0, "errors": 0, "hist": [0] * (len(BUCKETS) + 1), "sum": 0.0}


class ORJSON(JSONResponse):
    media_type = "application/json"
    def render(self, content) -> bytes: return orjson.dumps(content)


class TimingMiddleware:
    """Pure-ASGI middleware (no BaseHTTPMiddleware overhead): latency histogram + headers + safe 500s."""
    def __init__(self, app): self.app = app
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http": return await self.app(scope, receive, send)
        t = time.perf_counter(); started = False
        async def send_wrapper(msg):
            nonlocal started
            if msg["type"] == "http.response.start":
                started = True
                msg.setdefault("headers", []).extend([(b"x-response-time-ms", f"{(time.perf_counter()-t)*1000:.2f}".encode()),
                                                      (b"x-instance", INSTANCE.encode())])
                if msg["status"] >= 400: METRICS["errors"] += 1
            await send(msg)
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            METRICS["errors"] += 1
            if not started:
                body = orjson.dumps({"error": {"code": "internal", "message": "Внутренняя ошибка сервиса. Повторите запрос позже."}})
                await send({"type": "http.response.start", "status": 500, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": body})
        finally:
            dt = time.perf_counter() - t; METRICS["requests"] += 1; METRICS["sum"] += dt
            METRICS["hist"][next((i for i, b in enumerate(BUCKETS) if dt <= b), len(BUCKETS))] += 1


INSTANCE = os.getenv("HOSTNAME", uuid.uuid4().hex[:8])
app.add_middleware(TimingMiddleware)

# immutable forecasts -> cache serialized responses; key includes stream version (ingest invalidates)
from collections import OrderedDict
_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict(); CACHE_MAX = int(os.getenv("CACHE_MAX", "4096"))
CACHE_STATS = {"hit": 0, "miss": 0}


@app.exception_handler(ApiError)
async def api_error(_, e: ApiError):
    body = {"error": {"code": e.code, "message": e.message}}
    if e.hint: body["error"]["hint"] = e.hint
    return ORJSON(body, e.status)


@app.exception_handler(RequestValidationError)
async def validation_error(_, e):
    errs = getattr(e, "errors", lambda: [])()
    msg = "; ".join(f"{'.'.join(map(str, x.get('loc', [])))}: {x.get('msg')}" for x in errs) or str(e)
    return ORJSON({"error": {"code": "validation", "message": f"Некорректные параметры запроса — {msg}"}}, 422)


# ------------------------------------------------------------------ helpers
def _routes(route: str) -> list[int]:
    if route in ("all", "*", ""): return [r for r in STORE.routes]
    out = []
    for x in route.split(","):
        try: r = int(x)
        except ValueError: raise ApiError(422, "bad_route", f"Маршрут '{x}' не является числом", "Пример: route=17 или route=1,7 или route=all")
        if r not in STORE.r_idx: raise ApiError(404, "unknown_route", f"Маршрут {r} отсутствует в модели", f"Доступные: {STORE.routes}")
        out.append(r)
    return out


def _coef(name, v):
    if not (0.0 <= v <= 5.0): raise ApiError(422, "bad_coef", f"Коэффициент {name}={v} вне диапазона [0, 5]")
    return v


def _query(route, date_from, date_to, hour_from, hour_to, agg, horizon, stop_id, k_weather, k_event, k_season, precip_mm,
           section_from=None, section_to=None):
    STORE.refresh_stream()
    routes = _routes(route)
    d_from, d_to = STORE.resolve_range(horizon, parse_date(date_from, "date_from"), parse_date(date_to, "date_to"))
    share, stop, section = None, None, None
    if stop_id is not None or section_from is not None or section_to is not None:
        if len(routes) != 1: raise ApiError(422, "stop_needs_route", "Для остановки/участка укажите ровно один маршрут")
        if stop_id is not None and (section_from or section_to):
            raise ApiError(422, "stop_or_section", "Укажите либо stop_id, либо участок section_from/section_to")
        if stop_id is not None: share, stop = STORE.stop_share(routes[0], stop_id)
        else:
            if not (section_from and section_to): raise ApiError(422, "section_needs_both", "Для участка нужны section_from и section_to")
            share, section = STORE.section_share(routes[0], section_from, section_to)
    corr = {"weather": _coef("k_weather", k_weather), "event": _coef("k_event", k_event), "season": _coef("k_season", k_season)}
    if precip_mm is not None and not (0 <= precip_mm <= 100): raise ApiError(422, "bad_precip", "precip_mm должен быть в [0, 100]")
    data = STORE.series(routes, d_from, d_to, hour_from, hour_to, agg, corr, precip_mm, share)
    return {"routes": routes, "stop": stop, "section": section, "date_from": d_from.isoformat(), "date_to": d_to.isoformat(),
            "hour_from": hour_from, "hour_to": hour_to, "agg": agg, "horizon": horizon,
            "corrections": {**corr, "precip_mm": precip_mm}, "unit": "посадки (успешные валидации)", "data": data}


COMMON = dict(
    date_from=Query(None, description="YYYY-MM-DD (по умолчанию — начало прогноза 2025-11-01)"),
    date_to=Query(None, description="YYYY-MM-DD (по умолчанию определяется горизонтом)"),
)


# ------------------------------------------------------------------ routes
@app.get("/", include_in_schema=False)
def index(): return FileResponse(STATIC / "index.html")


@app.get("/health", tags=["service"])
async def health(): return {"status": "ok", "instance": INSTANCE, "model_version": STORE.meta.get("model_version")}


@app.get("/api/v1/meta", tags=["reference"])
async def meta():
    m = STORE.meta
    return ORJSON({k: m[k] for k in ("model_version", "first_day", "forecast_start", "last_day", "routes", "validation", "domain", "external_sources") if k in m})


@app.get("/api/v1/adaptation", tags=["forecast"])
async def adaptation():
    """Online Bayesian adaptation state: level multipliers per route x day-group learned from streamed days."""
    STORE.refresh_stream()
    return ORJSON({"last_complete_streamed_day": STORE.online_last.isoformat() if STORE.online_last else None,
                   "multipliers": {f"{r}|{g}": round(v, 4) for (r, g), v in sorted(STORE.online_mult.items())}})


@app.get("/api/v1/routes", tags=["reference"])
async def routes():
    return ORJSON([{"route": r, "has_stops": str(r) in STORE.stops,
                    "name": STORE.stops.get(str(r), {}).get("name"), "supply": STORE.supply.get(r)} for r in STORE.routes])


@app.get("/api/v1/routes/{route}/stops", tags=["reference"])
async def stops(route: int):
    """Остановки маршрута (порядок, координаты, источник геопривязки, доля потока по часам)."""
    s = STORE.stops.get(str(route))
    if not s: raise ApiError(404, "no_stops", f"Для маршрута {route} нет справочника остановок", "Есть: " + ", ".join(sorted(STORE.stops)))
    return ORJSON(s)


@app.get("/api/v1/forecast", tags=["forecast"])
async def forecast(route: str = Query("all", description="номер маршрута, список через запятую или all"),
             date_from: str | None = COMMON["date_from"], date_to: str | None = COMMON["date_to"],
             hour_from: int = Query(0, ge=0, le=23), hour_to: int = Query(23, ge=0, le=23),
             agg: str = Query("hour", description="hour | day | month | hour_profile | total"),
             horizon: str = Query("day", description="day | month | year"),
             stop_id: str | None = Query(None, description="остановка (доля маршрутного потока, зависит от часа)"),
             section_from: str | None = Query(None, description="участок: первая остановка (stop_id)"),
             section_to: str | None = Query(None, description="участок: последняя остановка (stop_id)"),
             k_weather: float = 1.0, k_event: float = 1.0, k_season: float = 1.0,
             precip_mm: float | None = Query(None, description="сценарий осадков, мм/день (заменяет факт/прогноз погоды)")):
    STORE.refresh_stream()
    key = (route, date_from, date_to, hour_from, hour_to, agg, horizon, stop_id, section_from, section_to, k_weather, k_event, k_season, precip_mm, STORE.version)
    body = _CACHE.get(key)
    if body is None:
        CACHE_STATS["miss"] += 1
        body = orjson.dumps(_query(route, date_from, date_to, hour_from, hour_to, agg, horizon, stop_id, k_weather, k_event, k_season, precip_mm,
                                   section_from, section_to))
        _CACHE[key] = body
        if len(_CACHE) > CACHE_MAX: _CACHE.popitem(last=False)
    else:
        CACHE_STATS["hit"] += 1; _CACHE.move_to_end(key)
    return Response(body, media_type="application/json")


@app.get("/api/v1/export", tags=["forecast"])
def export(fmt: str = Query("csv", alias="format", description="csv | xlsx"), route: str = "all",
           date_from: str | None = None, date_to: str | None = None, hour_from: int = 0, hour_to: int = 23,
           agg: str = "hour", horizon: str = "month", stop_id: str | None = None,
           section_from: str | None = None, section_to: str | None = None,
           k_weather: float = 1.0, k_event: float = 1.0, k_season: float = 1.0, precip_mm: float | None = None):
    rows = []
    for r in _routes(route):
        q = _query(str(r), date_from, date_to, hour_from, hour_to, agg, horizon, stop_id, k_weather, k_event, k_season, precip_mm,
                   section_from, section_to)
        rows += [{"route": r, **x} for x in q["data"]]
    if not rows: raise ApiError(404, "empty", "Нет данных за выбранный период")
    cols = list(rows[0].keys()); stamp = datetime.now().strftime("%Y%m%d_%H%M")
    if fmt == "csv":
        buf = io.StringIO(); w = csv.DictWriter(buf, cols, delimiter=";"); w.writeheader(); w.writerows(rows)
        return Response(buf.getvalue().encode("utf-8-sig"), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="forecast_{stamp}.csv"'})
    if fmt == "xlsx":
        from openpyxl import Workbook
        wb = Workbook(write_only=True); ws = wb.create_sheet("forecast"); ws.append(cols)
        for x in rows: ws.append([x[c] for c in cols])
        b = io.BytesIO(); wb.save(b)
        return Response(b.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f'attachment; filename="forecast_{stamp}.xlsx"'})
    raise ApiError(422, "bad_format", f"Формат '{fmt}' не поддерживается", "Допустимо: csv, xlsx")


@app.post("/api/v1/ingest/validations", tags=["ingest"])
async def ingest(request: Request):
    """Приём потоковых валидаций (JSON-массив или CSV с разделителем ';' в формате train.csv).
    Нормализация: validation_result==1, маршрут из ngpt_route, час из tran_date_time.
    Агрегаты пишутся append-only в STREAM_DIR (общий том) — видны всем репликам."""
    import pyarrow as pa, pyarrow.parquet as pq, re
    ct = request.headers.get("content-type", ""); raw = await request.body()
    if len(raw) > 50 * 1024 * 1024: raise ApiError(413, "too_large", "Пакет больше 50 МБ — разбейте на части")
    try:
        if "json" in ct: recs = orjson.loads(raw)
        else: recs = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter=";"))
    except Exception:
        raise ApiError(400, "bad_payload", "Не удалось разобрать тело запроса", "Ожидается JSON-массив записей или CSV (;) с заголовком как в train.csv")
    if not isinstance(recs, list): raise ApiError(400, "bad_payload", "Ожидается массив записей")
    agg, rejected = {}, 0
    for x in recs:
        try:
            if str(x.get("validation_result", "1")).strip() != "1": continue
            ts = datetime.strptime(str(x["tran_date_time"])[:19], "%Y-%m-%d %H:%M:%S")
            r = int(re.search(r"\d+", str(x["ngpt_route"])).group())
            k = (r, ts.date().isoformat(), ts.hour); agg[k] = agg.get(k, 0) + 1
        except Exception:
            rejected += 1
    if agg:
        STREAM.mkdir(parents=True, exist_ok=True)
        ks = list(agg); t = pa.table({"route": [k[0] for k in ks], "date": [k[1] for k in ks], "hour": [k[2] for k in ks],
                                      "boardings": [agg[k] for k in ks]})
        pq.write_table(t, STREAM / f"batch_{datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex[:6]}.parquet")
        STORE.refresh_stream(force=True)
    return ORJSON({"received": len(recs), "accepted_boardings": sum(agg.values()), "rejected": rejected, "cells_updated": len(agg)})


@app.post("/api/v1/admin/reload", tags=["service"])
def reload():
    STORE.load(); _CACHE.clear()          # new artifacts -> drop cached responses
    return {"status": "reloaded", "model_version": STORE.meta.get("model_version")}


@app.get("/metrics", include_in_schema=False)
async def metrics():
    lines = ["# TYPE http_requests_total counter", f"http_requests_total {METRICS['requests']}",
             f"http_errors_total {METRICS['errors']}", "# TYPE http_request_duration_seconds histogram"]
    c = 0
    for b, n in zip(BUCKETS + [math.inf], METRICS["hist"]):
        c += n; lines.append(f'http_request_duration_seconds_bucket{{le="{b}"}} {c}')
    lines += [f"cache_hits_total {CACHE_STATS['hit']}", f"cache_misses_total {CACHE_STATS['miss']}"]
    lines += [f"http_request_duration_seconds_sum {METRICS['sum']:.6f}", f"http_request_duration_seconds_count {METRICS['requests']}"]
    return PlainTextResponse("\n".join(lines) + "\n")
