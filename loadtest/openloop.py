"""Open-loop load test: Poisson arrivals at a FIXED rate (what an SLA means), plus streaming-ingest throughput.
python loadtest/openloop.py --url http://127.0.0.1:8001 --rates 200 300 400 --duration 30 --pids ...
"""
import argparse, asyncio, json, random, statistics, time
import aiohttp, psutil
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loadtest import random_query

POOL = 64

async def run_rate(base, rate, duration, pids):
    procs = [psutil.Process(p) for p in pids]; [p.cpu_percent(None) for p in procs]; psutil.cpu_percent(None, percpu=True)
    lat, errs, tasks = [], [], []; rng = random.Random(rate)
    conn = aiohttp.TCPConnector(limit=POOL, keepalive_timeout=60); to = aiohttp.ClientTimeout(total=10)   # like nginx upstream keepalive pool
    async with aiohttp.ClientSession(connector=conn, timeout=to) as s:
        async def one(q):
            t = time.perf_counter()
            try:
                async with s.get(base + q) as r:
                    await r.read()
                    if r.status >= 500 or (r.status >= 400 and "route=99" not in q): errs.append(r.status)
            except Exception as e: errs.append(type(e).__name__)
            lat.append(time.perf_counter() - t)
        t0 = time.perf_counter(); nxt = t0
        while nxt - t0 < duration:
            nxt += rng.expovariate(rate); d = nxt - time.perf_counter()
            if d > 0: await asyncio.sleep(d)
            tasks.append(asyncio.create_task(one(random_query(rng))))
        await asyncio.gather(*tasks); el = time.perf_counter() - t0
    cores = psutil.cpu_percent(None, percpu=True)[:2]   # system-level utilisation of the 2 pinned server cores
    cpu = sum(p.cpu_percent(None) for p in procs); rss = sum(p.memory_info().rss for p in procs) / 2**20
    q = statistics.quantiles(lat, n=100)
    return {"target_rps": rate, "achieved_rps": round(len(lat) / el, 1), "p50_ms": round(q[49] * 1e3, 1), "p95_ms": round(q[94] * 1e3, 1),
            "p99_ms": round(q[98] * 1e3, 1), "errors": len(errs), "cpu_pct_of_2_cores_process": round(cpu / 2, 1), "cpu_pct_cores_0_1_system": round(sum(cores) / 2, 1), "rss_mb": round(rss, 1)}

async def ingest_throughput(base, batches=20, size=5000):
    rng = random.Random(1); t0 = time.perf_counter(); n = 0
    async with aiohttp.ClientSession() as s:
        for b in range(batches):
            recs = [{"tran_date_time": f"2025-12-{rng.randint(20,28):02d} {rng.randint(5,23):02d}:{rng.randint(0,59):02d}:00",
                     "validation_result": 1, "ngpt_route": f"{rng.choice([1,7,11,12,17,25,26,28,50])} трамвай"} for _ in range(size)]
            async with s.post(base + "/api/v1/ingest/validations", json=recs) as r: await r.read(); n += size
    el = time.perf_counter() - t0
    return {"ingest_records": n, "seconds": round(el, 2), "records_per_s": round(n / el), "batch_size": size}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8001"); ap.add_argument("--rates", type=int, nargs="+", default=[200, 300, 400])
    ap.add_argument("--duration", type=float, default=30); ap.add_argument("--pids", type=int, nargs="*", default=[]); ap.add_argument("--out", default="loadtest/results_openloop.json")
    a = ap.parse_args(); res = []
    for r in a.rates: x = asyncio.run(run_rate(a.url, r, a.duration, a.pids)); print(json.dumps(x), flush=True); res.append(x)
    ing = asyncio.run(ingest_throughput(a.url)); print(json.dumps(ing)); res.append(ing)
    json.dump(res, open(a.out, "w"), indent=1)
