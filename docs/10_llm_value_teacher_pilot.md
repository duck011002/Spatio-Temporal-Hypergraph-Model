# Value-add LLM teacher v3: validation-only pilot

## Objective

This pilot tests whether Qwen-Flash can add behavioural judgment beyond the
frozen NYC A4 rule teacher. It does not retrain A4, modify rule weights, send
test samples to an API, or train a student gate.

## Why the previous LLM protocol was insufficient

The archived `binary_v1` and `history_teacher_v2` protocols exposed neural
scores, rule scores, and rule weights to a binary selector. That permits a
small LLM to repeat the existing rule decision instead of contributing an
independent signal. The old full-validation label set also selected every
Top-1 rule override, which was too broad for a value-focused API budget.

## v3 frozen protocol

`run_llm_value_teacher.py` selects validation samples satisfying all of:

1. The frozen rule fusion changes the frozen A4 Top-1 candidate.
2. Its fused Top-1 margin is in the lowest 40% among such overrides.
3. The backbone Top-1 margin is at least the median among such overrides.

The teacher sees only two randomly ordered candidates, strict-past recent
category history, time, transition/category evidence, candidate distance, and
repeat-rate evidence. `model_score`, `rule_score`, and learned rule weights
are removed before request construction. It must return a pairwise preference,
confidence, abstention flag, controlled evidence codes, and a short reason.

Responses are keyed by dataset, sample, prompt version, and prompt hash. The
script supports dry-run, deterministic stratified sampling, a preflight token
estimate, a hard reserved CNY budget, retries, and resume from cache.

## Pilot preflight and actual usage

NYC has 407 validation rule overrides. The frozen selector reduced them to
80 low-confidence conflicts. A deterministic 12-request stratified pilot
contained 8 `distill_train`, 2 `calibration`, and 2 `audit` samples.

| Quantity | Value |
|---|---:|
| Estimated input tokens | 6,195 |
| Output-token cap | 1,440 |
| Estimated cost | 0.00308925 CNY |
| Hard budget | 0.015 CNY |
| Actual prompt tokens | 6,564 |
| Actual completion tokens | 632 |
| Actual cost | 0.0019326 CNY |

All 12 JSON responses passed schema validation. No stored reason mentioned
the hidden score or rule fields.

## Validation-only decision evidence

On the 12 sampled cases, the backbone was correct 0 times, the rule teacher 3
times, and Qwen 2 times. Thus Qwen was `-1` versus the frozen rule teacher.
It abstained 0 times despite a mean stated confidence of 0.796, so its
confidence is not calibrated enough to make an automatic student label safe.

For context only, a local oracle over all 80 eligible validation cases has 24
correct choices versus 16 for the rule teacher (an upper bound of +8). The
pilot did not demonstrate that Qwen can realize this remaining oracle space;
it even changed one rule-correct sampled case to incorrect. The calibration
and audit portions of the pilot are also too small to justify a student model.

## Stop decision

The experiment stops after the 12-request pilot. Do not expand API labels,
fit a distilled gate/reranker, or evaluate v3 on test. This preserves the
validation/test isolation and avoids spending more money on an unvalidated
teacher. A future retry needs genuinely new information (for example,
verified POI names or external venue semantics) and must begin with a new
validation-only preflight rather than reusing this pilot as a positive label
source.

## Reproduction

```powershell
python -m unittest tests/test_llm_distillation.py tests/test_llm_value_teacher.py
python run_llm_value_teacher.py --datasets nyc --dry-run --limit-per-dataset 12 --max-cost-cny 0.015
```

The saved pilot summary is
`docs/results/llm_value_teacher/value_pairwise_v3_summary.json`; decision
records point to immutable per-request cache files under
`server_artifacts/llm_value_teacher/`.
