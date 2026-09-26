"""Start the API with N uvicorn workers pinned to K CPU cores (emulates a K-vCPU container on a dev box).
python loadtest/serve_pinned.py --cores 2 --workers 2 --port 8001   -> prints server PIDs"""
import argparse, subprocess, sys, time, psutil, os
ap = argparse.ArgumentParser(); ap.add_argument("--cores", type=int, default=2); ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--port", type=int, default=8001); a = ap.parse_args()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", str(a.port),
                      "--workers", str(a.workers), "--no-access-log", "--log-level", "warning"], cwd=root)
time.sleep(6)
cores = list(range(a.cores)); tree = [psutil.Process(p.pid)] + psutil.Process(p.pid).children(recursive=True)
for q in tree: q.cpu_affinity(cores)
print("PIDS", " ".join(str(q.pid) for q in tree), flush=True)
p.wait()
