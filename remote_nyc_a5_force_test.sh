#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/autodl-tmp/STHGCN-git
PY=/root/miniconda3/bin/python
OUT=artifacts/nyc_a5_jev_force_test
mkdir -p remote_runs
exec > >(tee -a remote_runs/nyc_a5_jev_force_test.log) 2>&1

echo "[$(date -Is)] START nyc_a5_jev_force_test"
if [[ -s "$OUT/test_report.json" ]]; then
  echo "[$(date -Is)] test_report already exists; refusing duplicate test"
  exit 0
fi

"$PY" -u force_a5_test_after_rejected_validation.py \
  --source-output artifacts/nyc_a5_jev_formal \
  --output "$OUT" \
  --reason "User explicitly requested NYC test after the formal validation gate rejected it" \
  --acknowledge-validation-rejection

export TYPESAFE_API_KEY="$(cat /root/.typesafe_api_key)"
"$PY" -u run_a5_jev.py query-test \
  --dataset nyc --data data/nyc/preprocessed \
  --candidates artifacts/nyc_a4_candidates --output "$OUT" --cap-usd 3.0
"$PY" -u run_a5_jev.py report \
  --dataset nyc --data data/nyc/preprocessed \
  --candidates artifacts/nyc_a4_candidates --output "$OUT" --cap-usd 3.0

echo "[$(date -Is)] DONE nyc_a5_jev_force_test"
