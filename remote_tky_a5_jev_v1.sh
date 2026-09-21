#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/STHGCN-git
PY=/root/miniconda3/bin/python
export TYPESAFE_API_KEY="$(cat /root/.typesafe_api_key)"

echo "[$(date -Is)] START tky_a5_jev_v1_test"
"$PY" -u run_a5_jev.py query-test \
  --dataset tky \
  --data data/tky/preprocessed \
  --candidates artifacts/tky_a4_candidates \
  --output artifacts/tky_a5_jev_v1

echo "[$(date -Is)] START tky_a5_jev_v1_report"
"$PY" -u run_a5_jev.py report \
  --dataset tky \
  --data data/tky/preprocessed \
  --candidates artifacts/tky_a4_candidates \
  --output artifacts/tky_a5_jev_v1

echo "[$(date -Is)] DONE tky_a5_jev_v1"
