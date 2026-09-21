#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/STHGCN-git
PY=/root/miniconda3/bin/python
mkdir -p remote_runs artifacts
if [[ ! -s /root/.typesafe_api_key ]]; then
  echo 'Jev key file missing; waiting for operator setup.' >&2
  exit 21
fi
export TYPESAFE_API_KEY="$(cat /root/.typesafe_api_key)"

until [[ -s artifacts/ca_a4_candidates/candidate_manifest.json && -s artifacts/tky_a4_candidates/candidate_manifest.json ]]; do
  if [[ -s remote_runs/pipeline.log ]] && grep -q 'A4 complete; TYPESAFE_API_KEY is missing' remote_runs/pipeline.log; then
    echo 'A4 finished; attaching Jev continuation.'
    break
  fi
  sleep 30
done

run_stage() {
  local name="$1"; shift
  echo "[$(date -Is)] START $name"
  "$@" >"remote_runs/${name}.log" 2>&1
  echo "[$(date -Is)] DONE $name"
}

run_stage ca_jev_prepare "$PY" run_a5_jev.py prepare --dataset ca --data data/ca/preprocessed \
  --candidates artifacts/ca_a4_candidates --output artifacts/ca_a5_jev --cap-usd 1.5
run_stage ca_jev_validation "$PY" run_a5_jev.py query-validation --dataset ca --data data/ca/preprocessed \
  --candidates artifacts/ca_a4_candidates --output artifacts/ca_a5_jev
run_stage ca_jev_select "$PY" run_a5_jev.py select --dataset ca --data data/ca/preprocessed \
  --candidates artifacts/ca_a4_candidates --output artifacts/ca_a5_jev
run_stage ca_jev_test "$PY" run_a5_jev.py query-test --dataset ca --data data/ca/preprocessed \
  --candidates artifacts/ca_a4_candidates --output artifacts/ca_a5_jev
run_stage ca_jev_report "$PY" run_a5_jev.py report --dataset ca --data data/ca/preprocessed \
  --candidates artifacts/ca_a4_candidates --output artifacts/ca_a5_jev

run_stage tky_jev_prepare "$PY" run_a5_jev.py prepare --dataset tky --data data/tky/preprocessed \
  --candidates artifacts/tky_a4_candidates --output artifacts/tky_a5_jev --cap-usd 2.3
run_stage tky_jev_validation "$PY" run_a5_jev.py query-validation --dataset tky --data data/tky/preprocessed \
  --candidates artifacts/tky_a4_candidates --output artifacts/tky_a5_jev
run_stage tky_jev_select "$PY" run_a5_jev.py select --dataset tky --data data/tky/preprocessed \
  --candidates artifacts/tky_a4_candidates --output artifacts/tky_a5_jev
run_stage tky_jev_test "$PY" run_a5_jev.py query-test --dataset tky --data data/tky/preprocessed \
  --candidates artifacts/tky_a4_candidates --output artifacts/tky_a5_jev
run_stage tky_jev_report "$PY" run_a5_jev.py report --dataset tky --data data/tky/preprocessed \
  --candidates artifacts/tky_a4_candidates --output artifacts/tky_a5_jev
echo "[$(date -Is)] JEV_PIPELINE_COMPLETE"
