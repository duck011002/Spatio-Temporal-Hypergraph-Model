#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/STHGCN-git
PY=/root/miniconda3/bin/python
mkdir -p remote_runs artifacts

run_stage() {
  local name="$1"; shift
  echo "[$(date -Is)] START $name"
  "$@" >"remote_runs/${name}.log" 2>&1
  echo "[$(date -Is)] DONE $name"
}

latest_formal_checkpoint() {
  find tensorboard -path "*/$1/checkpoint.pt" -type f -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-
}

if ! grep -q 'Test evaluation result' remote_runs/ca_smoke.log; then
  echo 'CA smoke gate missing; refusing formal pipeline' >&2
  exit 20
fi

run_stage ca_a4 "$PY" -u run.py -f best_conf/ca_a4.yml
CA_CKPT=$(latest_formal_checkpoint ca)
test -n "$CA_CKPT"
run_stage ca_export "$PY" export_a5_candidates.py -f best_conf/ca_a4.yml \
  --checkpoint "$CA_CKPT" --output artifacts/ca_a4_candidates

run_stage tky_a4 "$PY" -u run.py -f best_conf/tky_a4.yml
TKY_CKPT=$(latest_formal_checkpoint tky)
test -n "$TKY_CKPT"
run_stage tky_export "$PY" export_a5_candidates.py -f best_conf/tky_a4.yml \
  --checkpoint "$TKY_CKPT" --output artifacts/tky_a4_candidates

if [[ -z "${TYPESAFE_API_KEY:-}" ]]; then
  echo 'A4 complete; TYPESAFE_API_KEY is missing, stopping before Jev.' >&2
  exit 21
fi

run_stage ca_jev_prepare "$PY" run_a5_jev.py prepare --dataset ca \
  --data data/ca/preprocessed --candidates artifacts/ca_a4_candidates \
  --output artifacts/ca_a5_jev --cap-usd 1.5
run_stage ca_jev_validation "$PY" run_a5_jev.py query-validation --dataset ca \
  --data data/ca/preprocessed --candidates artifacts/ca_a4_candidates \
  --output artifacts/ca_a5_jev
run_stage ca_jev_select "$PY" run_a5_jev.py select --dataset ca \
  --data data/ca/preprocessed --candidates artifacts/ca_a4_candidates \
  --output artifacts/ca_a5_jev
run_stage ca_jev_test "$PY" run_a5_jev.py query-test --dataset ca \
  --data data/ca/preprocessed --candidates artifacts/ca_a4_candidates \
  --output artifacts/ca_a5_jev
run_stage ca_jev_report "$PY" run_a5_jev.py report --dataset ca \
  --data data/ca/preprocessed --candidates artifacts/ca_a4_candidates \
  --output artifacts/ca_a5_jev

run_stage tky_jev_prepare "$PY" run_a5_jev.py prepare --dataset tky \
  --data data/tky/preprocessed --candidates artifacts/tky_a4_candidates \
  --output artifacts/tky_a5_jev --cap-usd 2.3
run_stage tky_jev_validation "$PY" run_a5_jev.py query-validation --dataset tky \
  --data data/tky/preprocessed --candidates artifacts/tky_a4_candidates \
  --output artifacts/tky_a5_jev
run_stage tky_jev_select "$PY" run_a5_jev.py select --dataset tky \
  --data data/tky/preprocessed --candidates artifacts/tky_a4_candidates \
  --output artifacts/tky_a5_jev
run_stage tky_jev_test "$PY" run_a5_jev.py query-test --dataset tky \
  --data data/tky/preprocessed --candidates artifacts/tky_a4_candidates \
  --output artifacts/tky_a5_jev
run_stage tky_jev_report "$PY" run_a5_jev.py report --dataset tky \
  --data data/tky/preprocessed --candidates artifacts/tky_a4_candidates \
  --output artifacts/tky_a5_jev

echo "[$(date -Is)] PIPELINE_COMPLETE"
