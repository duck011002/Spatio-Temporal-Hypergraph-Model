# NYC A4 + Jev direct Top-20 reranking, without a rule gate

Protocol fixed before this experiment's API calls (2026-09-20).

## Question and scope

Can a Jev next-POI distribution improve the frozen A4 backbone without rule
scores, rule weights, rule-based conflict selection, or the old arbitration
gate? Run on every validation query, including queries where the previous
rule teacher agreed with A4. No training or model-forward changes are needed:
the archived A4 Top-20 cache is the exact input to the proposed reranker.

This is exploratory reuse of NYC. Its validation and historical test have
already been observed in prior experiments; re-partitioning does not create
an untouched benchmark. Reuse the prior user-level selection/calibration/audit
split, explicitly acknowledging this limitation.

## Next-POI-specific state and decisions

- One Jev Choice with 20 shuffled candidate aliases plus `outside`.
- Model: `jev-1.13.0`; English instructions; no example labels in prompts.
- Query local hour/day, last 12 strictly observed visits with time gaps and
  category names. A historical visit refers to a candidate alias when that
  same POI is present, preserving repeat-visit identity without exposing IDs.
- Candidate features: training-only category/coordinates, distance from last
  observed visit, actual prior visit count, hours since last observed visit.
  Prior visit counts use this user's history up to the input cutoff, not the
  target or future events. Distances/counts are descriptive facts, not rule
  scores or manually weighted decisions.
- Candidate aliases and order are deterministic random permutations. No
  A4 ranks/logits, rule scores, aggregate rule priors, ground-truth target,
  target category/location, user ID, or global POI ID are sent.
- Previous observed event <= last_checkin_epoch_time and strictly before
  target timestamp. Query time is known as assumed by the original pipeline.

## Fusion and comparisons

Compare A4, pure Jev Top-20 ordering, and A4+Jev probability fusion.
The A4 distribution is softmax over its cached Top-20 logits (temperature 1).
The Jev candidate probabilities are normalized conditional on the next POI
being one of those 20. If Jev assigns zero total mass to the 20, use uniform
conditional mass and stable ties preserve A4. Keep `outside` probability as
an uncertainty diagnostic; it does not activate a gate or select samples.

`p_final = (1-alpha) * p_A4 + alpha * p_Jev_conditional`

Choose one global alpha on selection only from
[0, .01, .025, .05, .1, .2, .35, .5, .75, 1]. Maximize R@1, then MRR,
then prefer smaller alpha. Alpha 0 is exact A4; alpha 1 is pure Jev.
All candidates remain present, so R@20 is unchanged. Rankings outside the
Top-20 retain archived A4 ranks for full MRR.

Proceed to one historical-test diagnostic only if alpha>0, calibration R@1
and MRR are both >= A4, and audit R@1 is strictly above A4 without lower MRR.
Otherwise stop at validation and withdraw the proposed integration. If test
is reached, retain the proposal only if test R@1 and MRR improve, with a
positive lower bound for its user-cluster R@1 bootstrap interval; otherwise
withdraw. Retention is an engineering decision, not a claim of unbiased
generalization or a production deployment authorization.

## Reliability, costs, and rollback

- 32 hash-selected validation queries are repeated with a reversed candidate
  alias order. Report argmax changes, without modifying main predictions.
- Evaluate outside-event probability calibration, R@1/5/10/20, MRR, NDCG@5,
  rescued/lost cases and 2,000 user-cluster paired bootstrap replicates.
- Maximum concurrency 2, <=3 attempts, 429/5xx backoff, explicit timeouts,
  immutable request-hash cache, budget reservations per attempt.
- This experiment has a $1 cost cap, separate from the previous ~$0.052;
  combined known usage remains far below the user's $5 balance.
- All implementation and caches live under ignored
  `server_artifacts/jev_direct_nyc_20260920/`. Main model/configuration files
  are not modified. On failure, record `withdrawn` and leave no enabled
  integration; retain source/cache/results for audit and discussion.
- API key is process-environment only. Save provenance before calls and
  freeze the alpha and validation result before any test calls.
