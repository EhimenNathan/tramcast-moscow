#!/usr/bin/env bash
# Open-loop test against the Docker container (limits: --cpus 2 --memory 2g), container CPU sampled via docker stats.
C=${1:-tram1}; URL=${2:-http://127.0.0.1:8090}; OUT=${3:-loadtest/results_docker.json}; echo "[" > $OUT
for R in ${RATES:-100 200 300 400 500}; do
  ( for k in $(seq 1 12); do docker stats --no-stream --format "{{.CPUPerc}} {{.MemUsage}}" $C; done ) > loadtest/tmp/dstats_$R.txt &
  SP=$!
  python loadtest/openloop.py --url $URL --rates $R --duration ${DUR:-30} --out loadtest/tmp/ol_$R.json > /dev/null 2>&1
  wait $SP
  CPU=$(awk '{gsub("%","",$1); s+=$1; n++} END{printf "%.0f", s/n/2}' loadtest/tmp/dstats_$R.txt)
  MEM=$(tail -1 loadtest/tmp/dstats_$R.txt | awk '{print $2}')
  python -c "
import json; r=json.load(open('loadtest/tmp/ol_$R.json'))[0]; r['container_cpu_pct_of_2_vcpu']=$CPU; r['container_mem']='$MEM'
for k in ('cpu_pct_of_2_cores_process','cpu_pct_cores_0_1_system','rss_mb'): r.pop(k,None)
print(json.dumps(r)); open('$OUT','a').write(json.dumps(r)+',\n')"
done
python loadtest/openloop.py --url $URL --rates 50 --duration 5 --out loadtest/tmp/ing.json > /dev/null 2>&1
python -c "import json;x=json.load(open('loadtest/tmp/ing.json'))[-1];print(json.dumps(x));open('$OUT','a').write(json.dumps(x)+']\n')"
