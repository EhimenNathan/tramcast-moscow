#!/usr/bin/env bash
# usage: bash loadtest/bench.sh <label> <CACHE_MAX>   (server: 2 workers pinned to cores 0-1; client pinned to cores 2-3)
LABEL=$1; export CACHE_MAX=$2
python -u loadtest/serve_pinned.py --cores 2 --workers 2 --port 8001 > loadtest/server_$LABEL.log 2>&1 &
until grep -q PIDS loadtest/server_$LABEL.log; do sleep 1; done
PIDS=$(grep PIDS loadtest/server_$LABEL.log | cut -d' ' -f2-)
python -c "
import psutil,subprocess,sys
p=subprocess.Popen([sys.executable,'-u','loadtest/loadtest.py','--url','http://127.0.0.1:8001','--concurrency','16','32','64','128','--duration','20','--pids',*'$PIDS'.split(),'--out','loadtest/results_$LABEL.json'])
psutil.Process(p.pid).cpu_affinity([2,3]); p.wait()"
curl -s http://127.0.0.1:8001/metrics | grep -E "cache|requests_total"
taskkill //F //T //PID $(echo $PIDS | cut -d' ' -f1) > /dev/null 2>&1
