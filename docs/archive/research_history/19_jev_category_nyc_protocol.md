# NYC Jev category-marginal fusion (exploratory)

Fixed before API calls, 2026-09-20. Hypothesis: semantic category prediction
is easier for Jev than identifying an individual next POI. No rule gate,
rule scores, sample filtering, model training, or main-model modifications.

## Input and output

All 1400 archived A4 validation queries; same user-level selection/calibration/
audit partition as previous experiments. This development set has already been
reused; neither these partitions nor the historical test are untouched data.

Jev jev-1.13.0 receives known query local hour/day, last 12 strictly observed
visits (category and elapsed hours), total observed history length, and the
distinct category names among A4 Top-20 candidates. Category metadata comes
from training data only. No target fields, global IDs, A4 scores, candidate
counts per category, POI distances, or handcrafted behavioral scores enter
the request. History is <= last observed event and strictly before query.

Predict one distribution over shuffled category aliases plus outside_category
(the next visit's category is absent from listed categories; this does NOT
mean that the next POI is absent). Use original dataset categories, not a
manually invented taxonomy. Unknown metadata is an explicit unknown category.

## Fusion

Let p_i be softmax of A4 Top-20 logits and P_c=sum(p_i for category c).
Normalize Jev category mass over the listed categories to q_c. If no listed
mass exists, use P_c (no change). Form r_i=q_c * p_i/P_c, then
p_final_i=(1-alpha)*p_i+alpha*r_i.

Thus only category mass changes; within-category A4 ordering is preserved.
Compare A4, full category replacement (alpha=1), selected Jev fusion, and a
uniform-category control (q_c=1/number_of_categories). Tune Jev and uniform
alpha independently on selection only, from [0,.01,.025,.05,.1,.2,.35,.5,.75,1],
maximizing R@1 then MRR then preferring smaller alpha. No post-result changes.

Same acceptance rule as the direct-POI experiment: Jev alpha>0, calibration
R@1 and MRR >= A4, audit R@1 > A4 and MRR >= A4. Only then run a single frozen
historical-test diagnostic. Retain a proposal only if test R@1 and MRR rise
and user-cluster bootstrap R@1 lower bound is >0. Report the uniform control
regardless; an improvement over A4 alone does not establish Jev-specific value.

## Checks and limits

- Freeze source/data/protocol hashes before calls, parameters before test.
- Target/future invariance, category mapping, alpha=0 identity, no-information
  identity, same-category order, candidate membership, probability sum checks.
- Reverse category alias order for 32 hash-selected validation queries;
  report mapped argmax flips, never alter original predictions.
- Category accuracy/NLL evaluated only where target category has training
  metadata. These labels are local evaluation only, never part of requests.
- R@1/5/20, MRR, paired rescued/lost, 2000 user-cluster bootstrap replicates.
- Concurrency 2, cached requests, retries/backoff, per-attempt reservations,
  hard conservative $1 cap for this experiment including any test calls.
- API key process environment only. Source/cache/results isolated under
  server_artifacts/jev_category_nyc_20260920. No enabled runtime integration.
  Withdraw failed proposal, retain audit artifacts; do not delete prior work.
