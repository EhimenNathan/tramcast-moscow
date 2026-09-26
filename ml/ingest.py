"""Stage 1 — ingestion & normalisation of raw validations (streamed from dataset.zip, never unpacked).

Output (data/):
  hourly_boardings.parquet : route, date, hour, boardings           (target, validation_result == 1)
  daily_supply.parquet     : route, date, vehicles, exits, cards    (unique trams / schedule exits / cards per day)
  hourly_supply.parquet    : route, date, hour, vehicles           (distinct trams with >=1 boarding in the hour)
  ingest_report.json       : row counts, rejects, validation vs official labels
Usage: python ml/ingest.py --zip ../dataset.zip [--files train.csv test.csv] [--chunk 1000000]
"""
import argparse, io, json, re, time, zipfile
import numpy as np, pandas as pd

ROUTE_RE = re.compile(r"(\d+)")
USE = ["tran_date_time", "crd_hashcode", "validation_result", "ngpt_route", "bus_exit_no", "garage_number"]

def normalise(ch: pd.DataFrame) -> pd.DataFrame:
    ch = ch[ch["validation_result"] == 1]                                   # boarding = successful validation
    ts = pd.to_datetime(ch["tran_date_time"], errors="coerce", format="%Y-%m-%d %H:%M:%S")
    route = pd.to_numeric(ch["ngpt_route"].astype(str).str.extract(ROUTE_RE, expand=False), errors="coerce")
    out = pd.DataFrame({"ts": ts, "route": route, "card": ch["crd_hashcode"],
                        "exit": ch["bus_exit_no"], "veh": ch["garage_number"]}).dropna(subset=["ts", "route"])
    out["route"] = out["route"].astype(np.int16)
    out["date"] = out["ts"].dt.normalize(); out["hour"] = out["ts"].dt.hour.astype(np.int8)
    return out

def extract_small(z, out_dir):
    """Organiser reference files needed by ml/train.py (not redistributed in the repository)."""
    import os
    os.makedirs(f"{out_dir}/labels", exist_ok=True); os.makedirs(f"{out_dir}/reference", exist_ok=True)
    for n in z.namelist():
        if n.startswith("labels/") and n.endswith(".csv"):
            open(f"{out_dir}/labels/" + n.split("/")[-1], "wb").write(z.read(n))
        if n.startswith("spravochniki/") and "10_" in n and n.endswith(".xlsx"):
            open(f"{out_dir}/reference/routes_stops.xlsx", "wb").write(z.read(n))


def run(zip_path, files, chunk, out_dir, cards=False):
    z = zipfile.ZipFile(zip_path); extract_small(z, out_dir); hourly = []; supply = {}; hsup = {}; rep = {"rows": 0, "boardings": 0, "files": files}
    t0 = time.time()
    for fn in files:
        with z.open(fn) as f:
            for ch in pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), sep=";", usecols=USE, chunksize=chunk,
                                  dtype={"crd_hashcode": "string", "ngpt_route": "string", "bus_exit_no": "string", "garage_number": "string"}):
                rep["rows"] += len(ch); n = normalise(ch); rep["boardings"] += len(n)
                hourly.append(n.groupby(["route", "date", "hour"]).size().rename("boardings").reset_index())
                for (r, d, h), g in n.groupby(["route", "date", "hour"]):
                    hsup.setdefault((r, d, h), set()).update(g["veh"].dropna().unique())
                for (r, d), g in n.groupby(["route", "date"]):            # exact distinct sets, merged across chunks
                    s = supply.setdefault((r, d), [set(), set(), set()])
                    s[0].update(g["veh"].dropna().unique()); s[1].update(g["exit"].dropna().unique()); (s[2].update(g["card"].dropna().unique()) if cards else None)
                print(f"{fn}: {rep['rows']:,} rows, {time.time()-t0:.0f}s", flush=True)
    h = pd.concat(hourly).groupby(["route", "date", "hour"], as_index=False).boardings.sum()
    h.to_parquet(f"{out_dir}/hourly_boardings.parquet", index=False)
    sp = pd.DataFrame([(r, d, len(a), len(b), len(c)) for (r, d), (a, b, c) in supply.items()],
                      columns=["route", "date", "vehicles", "exits", "cards"]).sort_values(["route", "date"])
    sp.to_parquet(f"{out_dir}/daily_supply.parquet", index=False)
    pd.DataFrame([(r, d, h, len(v)) for (r, d, h), v in hsup.items()], columns=["route", "date", "hour", "vehicles"]
                 ).sort_values(["route", "date", "hour"]).to_parquet(f"{out_dir}/hourly_supply.parquet", index=False)
    rep["seconds"] = round(time.time() - t0, 1); rep["hourly_rows"] = len(h)
    json.dump(rep, open(f"{out_dir}/ingest_report.json", "w"), indent=1, default=str); return h

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--zip", default="../dataset.zip")
    ap.add_argument("--files", nargs="+", default=["train.csv", "test.csv"]); ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--out", default="data"); ap.add_argument("--cards", action="store_true", help="exact unique cards (needs RAM)")
    a = ap.parse_args(); run(a.zip, a.files, a.chunk, a.out, a.cards)
