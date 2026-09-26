"""Load test: mixed realistic API traffic, fixed-concurrency closed loop.
python loadtest/loadtest.py --url http://127.0.0.1:8000 --concurrency 16 32 64 --duration 30 [--pids 123 456]
Reports throughput (RPS), latency p50/p95/p99, error rate, server CPU% (of pinned cores) and RSS.
"""
import argparse, asyncio, json, random, statistics, time
import aiohttp, psutil

ROUTES = [1, 7, 11, 12, 17, 25, 26, 28, 50]
def random_query(rng):
    k = rng.random(); r = rng.choice(ROUTES)
    d = f"2025-{rng.choice(['11','12'])}-{rng.randint(1,28):02d}"
    if k < 0.35: return f"/api/v1/forecast?route={r}&date_from={d}&agg=hour"                                   # day, hourly
    if k < 0.55: return f"/api/v1/forecast?route={r}&horizon=month&date_from=2025-11-01&agg=day"               # month, daily
    if k < 0.65: return f"/api/v1/forecast?route=all&date_from={d}&hour_from=7&hour_to=10&agg=total"           # network peak
    if k < 0.72: return f"/api/v1/forecast?route={r}&date_from=2026-01-01&date_to=2026-12-31&agg=month"         # year
    if k < 0.80: return f"/api/v1/forecast?route={r}&date_from={d}&agg=hour&k_weather=0.9&k_event=1.2&precip_mm=8"  # corrections
    if k < 0.86: return f"/api/v1/forecast?route=1&stop_id=2594&date_from={d}&agg=hour"                          # stop
    if k < 0.92: return f"/api/v1/forecast?route={r}&date_from=2025-11-01&date_to=2025-12-31&agg=hour_profile"
    if k < 0.96: return f"/api/v1/history?route={r}" if False else f"/api/v1/forecast?route={r},{rng.choice(ROUTES)}&date_from={d}&agg=day"
    if k < 0.98: return f"/api/v1/export?format=csv&route={r}&horizon=month&agg=day"
    return f"/api/v1/forecast?route=99"                                                                           # expected 404

async def worker(session, base, stop_at, lat, errs, rng):
    while time.perf_counter() < stop_at:
        q = random_query(rng); t = time.perf_counter()
        try:
            async with session.get(base + q) as r:
                await r.read()
                if r.status >= 500 or (r.status >= 400 and "route=99" not in q): errs.append(r.status)
        except Exception as e: errs.append(str(e))
        lat.append(time.perf_counter() - t)

async def run(base, conc, duration, pids):
    procs = [psutil.Process(p) for p in pids] if pids else []
    for p in procs: p.cpu_percent(None)
    lat, errs = [], []
    conn = aiohttp.TCPConnector(limit=conc, keepalive_timeout=30)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=10)) as s:
        for _ in range(50):  # warm-up
            async with s.get(base + "/health") as r: await r.read()
        t0 = time.perf_counter(); stop = t0 + duration
        await asyncio.gather(*[worker(s, base, stop, lat, errs, random.Random(i)) for i in range(conc)])
        el = time.perf_counter() - t0
    cpu = sum(p.cpu_percent(None) for p in procs) if procs else None
    rss = sum(p.memory_info().rss for p in procs) / 2**20 if procs else None
    q = statistics.quantiles(lat, n=100)
    return {"concurrency": conc, "requests": len(lat), "rps": round(len(lat) / el, 1), "p50_ms": round(q[49] * 1000, 1),
            "p95_ms": round(q[94] * 1000, 1), "p99_ms": round(q[98] * 1000, 1), "errors": len(errs),
            "server_cpu_pct_total": None if cpu is None else round(cpu, 1), "server_rss_mb": None if rss is None else round(rss, 1)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[8, 16, 32, 64]); ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--pids", type=int, nargs="*", default=[]); ap.add_argument("--out", default="loadtest/results.json")
    a = ap.parse_args(); res = []
    for c in a.concurrency:
        r = asyncio.run(run(a.url, c, a.duration, a.pids)); print(json.dumps(r), flush=True); res.append(r)
    json.dump(res, open(a.out, "w"), indent=1)
