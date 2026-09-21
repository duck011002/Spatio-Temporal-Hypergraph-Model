#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/autodl-tmp/STHGCN-git
PY=/root/miniconda3/bin/python
LOG=remote_runs/nyc_a5_formal.log
mkdir -p remote_runs
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Is)] START nyc_a5_formal_sequence"

# The TKY test is the only preceding network stage.  Do not launch NYC
# until its runner and query process have both exited and its report exists.
while pgrep -f 'run_a5_jev.py query-test --dataset tky' >/dev/null || \
      pgrep -f 'remote_tky_a5_jev_v1.sh' >/dev/null; do
  echo "[$(date -Is)] WAIT tky_a5_jev_v1"
  sleep 30
done

if [[ ! -s artifacts/tky_a5_jev_v1/test_report.json ]]; then
  echo "[$(date -Is)] ERROR TKY test report is missing; refusing to start NYC"
  exit 2
fi
echo "[$(date -Is)] TKY complete; starting NYC candidate export"

NYC_CAND=artifacts/nyc_a4_candidates
NYC_OUT=artifacts/nyc_a5_jev_formal
if [[ -s "$NYC_CAND/candidate_manifest.json" ]]; then
  echo "[$(date -Is)] Reusing verified NYC candidate manifest"
elif [[ -s "$NYC_CAND/validation_candidates.npz" && -s "$NYC_CAND/test_candidates.npz" ]]; then
  echo "[$(date -Is)] Using verified legacy NYC A4 candidate cache"
  "$PY" -u make_a5_candidate_manifest.py \
    --dataset nyc --data data/nyc/preprocessed --candidates "$NYC_CAND" \
    --config conf/best_conf/nyc_r1_6_aux_free_dual_gate.yml --seed 80786525 \
    --source-note "Legacy A4 candidate cache from 20260723_155322; checkpoint metadata unavailable"
else
  NYC_CKPT=tensorboard/20260723_155322/nyc/checkpoint.pt
  if [[ ! -s "$NYC_CKPT" ]]; then
    echo "[$(date -Is)] ERROR documented NYC A4 checkpoint is missing: $NYC_CKPT"
    exit 3
  fi
  if [[ -d "$NYC_CAND" ]] && find "$NYC_CAND" -mindepth 1 -print -quit | grep -q .; then
    echo "[$(date -Is)] ERROR NYC candidate directory is non-empty without a manifest; refusing overwrite"
    exit 4
  fi
  "$PY" -u export_a5_candidates.py \
    -f best_conf/nyc_r1_6_aux_free_dual_gate.yml \
    --checkpoint "$NYC_CKPT" \
    --output "$NYC_CAND"
fi

if [[ -e "$NYC_OUT/prepared.json" ]]; then
  echo "[$(date -Is)] ERROR NYC output already contains prepared.json; refusing to mix runs"
  exit 5
fi

"$PY" -u run_a5_jev.py prepare \
  --dataset nyc --data data/nyc/preprocessed \
  --candidates "$NYC_CAND" --output "$NYC_OUT" --cap-usd 3.0

export TYPESAFE_API_KEY="$(cat /root/.typesafe_api_key)"
"$PY" -u run_a5_jev.py query-validation \
  --dataset nyc --data data/nyc/preprocessed \
  --candidates "$NYC_CAND" --output "$NYC_OUT" --cap-usd 3.0
"$PY" -u run_a5_jev.py select \
  --dataset nyc --data data/nyc/preprocessed \
  --candidates "$NYC_CAND" --output "$NYC_OUT" --cap-usd 3.0

if "$PY" - <<'PY'
import json
from pathlib import Path
p = Path('artifacts/nyc_a5_jev_formal/frozen.json')
print(json.loads(p.read_text())['eligible_for_test'])
raise SystemExit(0 if json.loads(p.read_text())['eligible_for_test'] else 1)
PY
then
  "$PY" -u run_a5_jev.py query-test \
    --dataset nyc --data data/nyc/preprocessed \
    --candidates "$NYC_CAND" --output "$NYC_OUT" --cap-usd 3.0
  "$PY" -u run_a5_jev.py report \
    --dataset nyc --data data/nyc/preprocessed \
    --candidates "$NYC_CAND" --output "$NYC_OUT" --cap-usd 3.0
else
  echo "[$(date -Is)] NYC validation rejected; test queries intentionally skipped"
fi

echo "[$(date -Is)] DONE nyc_a5_formal_sequence"
