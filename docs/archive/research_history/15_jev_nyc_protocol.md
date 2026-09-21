# NYC Jev probability arbitration: frozen protocol (2026-09-20)

This protocol is written before any NYC Jev outcome is examined. This is an
exploratory extension of an already-developed NYC model, not a new untouched
benchmark. Historical NYC test outcomes and validation have already informed
prior research. The new audit is user-disjoint within this run, but is not
independent of that historical research or the original rule-weight search.

## Scope and fixed comparisons

- Frozen NYC A4 candidate cache and archived rule weights; no GPU inference,
  backbone training, candidate re-export, or rule-weight tuning.
- Call Jev `jev-1.13.0` only where rule fusion changes the A4 Top-1.
- Two predefined prompts: `evidence` (normalized model/rule scores plus
  component evidence and strict-past history), and `context` (history,
  candidate category/distance and training preferences, without model/rule
  scores). They are not identical to the old Qwen prompts; Qwen's available
  aggregate results are historical context, not a model-only ablation.
- One Choice question: A, B, or neither. Candidate order is deterministic,
  randomized without labels. No user/POI IDs, target labels, target category,
  target coordinates, or future history are sent to the API.
- Query hour/weekday are supplied under the same known-query-time assumption
  as the existing rule teacher. History is capped at the previous observed
  check-in and strictly precedes the target timestamp. Training-only POI
  metadata and preference statistics follow the frozen rule baseline.
- If Jev selects the A4 candidate with probability at least the threshold,
  move that candidate to the front of the rule ranking, retaining relative
  order of all remaining candidates. Otherwise keep the rule ranking. A
  `neither` answer preserves rule ranking and does not remove candidates.
- Preserve Top-20 membership and Recall@20. Measure full-set R@1/5/10/20,
  MRR and NDCG@5, not accuracy restricted to easy/decisive cases.

## Selection, calibration, and audit

Partition by SHA256(`jev-nyc-v1:user:<UserId>`) modulo 10: 0..5 selection,
6..7 calibration, 8..9 audit. Users never cross these validation partitions.
Both prompts are run on every validation conflict before the fixed selection
procedure. Choose prompt on selection by R@1, then MRR, using threshold 0.5.
Choose threshold on calibration from [0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.01],
by R@1, then MRR, then larger threshold. 1.01 always keeps the rule ranking.
Audit does not change the choice. Deployment eligibility requires calibration
not worse than rules in R@1/MRR and audit strictly better in R@1 without an
MRR decrease.

Two local baselines use identical partitions: a fixed-C=1 standardized
multinomial logistic classifier, fitted to selection conflict outcomes
(A4 correct / rule correct / neither), and a score-margin gate with no fit.
Their thresholds use the same calibration grid. They are not used to choose
the Jev prompt. Report their different access to supervised development labels.

Regardless of audit eligibility, freeze selected Jev prompt/threshold before
one historical-test diagnostic; also report the chosen prompt at the
predeclared 0.5 threshold. If audit fails, this does not authorize deployment
or test-based tuning. No further prompt or threshold search follows test.

## Reliability, cost, and provenance

- 32 hash-selected validation conflicts per prompt get an option-order swap;
  report decision flip rate and probability change, without selecting cases
  by correctness or changing main predictions.
- Report 3-class Brier/NLL, top-label calibration and coverage/error bins.
  These concern A4/rule/neither outcomes, not calibration among only the two
  selected POIs. The main intervention metric is rescued minus lost Top-1.
- Paired 95% user-cluster bootstrap intervals and discordant-pair counts;
  descriptive only on the reused benchmark and post-selection partitions.
- API concurrency <=2, explicit HTTP timeout, at most 3 attempts per request,
  429/5xx exponential backoff honoring Retry-After. No opaque SDK retries.
- A persistent ledger reserves the cost of 64,000 input tokens before each
  attempt, reconciles successful usage, and conservatively retains that
  reservation after uncertain failures. Price: $0.042/M input tokens.
  Total run limit $1.00, substantially below the user's $5 balance.
- Responses and request hashes cached; API key supplied only through process
  environment, never included in requests saved to disk, code, or reports.
- Pin source hashes, data hashes, model response version, prompts, partition
  membership, validation selection, and freeze record before test.

## Existing evidence to reproduce first

Validation: 1,400 samples; A4 366 correct, rules 379; 407 conflicts.
Historical test: 1,347 samples; A4 352 correct, rules 371; 407 conflicts.
Old Qwen binary-v1 validation: 376 correct (379 - 3); only its aggregate is
currently available, so no Jev/Qwen paired confidence interval is possible.
