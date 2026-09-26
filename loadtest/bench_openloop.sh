#!/usr/bin/env bash
# server: 2 uvicorn workers pinned to cores 0-1 (2-vCPU container emulation); client pinned to cores 2-3
export STREAM_DIR=$(mktemp -d); export CACHE_MAX=${CACHE_MAX:-4096}; OUT=${OUT:-loadtest/results_openloop.json}
python -u loadtest/serve_pinned.py --cores 2 --workers 2 --port 8001 > loadtest/server_openloop.log 2>&1 &
until grep -q PIDS loadtest/server_openloop.log; do sleep 1; done
PIDS=$(grep PIDS loadtest/server_openloop.log | cut -d' ' -f2-)
python -c "
import psutil,subprocess,sys
p=subprocess.Popen([sys.executable,'-u','loadtest/openloop.py','--rates','200','300','400','500','600','700','--duration','30','--pids',*'$PIDS'.split(),'--out','$OUT'],cwd='.')
psutil.Process(p.pid).cpu_affinity([2,3]); p.wait()"
taskkill //F //T //PID $(echo $PIDS | cut -d' ' -f1) > /dev/null 2>&1; rm -rf "$STREAM_DIR"
