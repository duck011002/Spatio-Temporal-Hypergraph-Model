# LLM teacher follow-up: raw-context v4 and semantic-data gate

## v4: strict raw-context pilot

The first v3 pilot still included per-candidate rule components such as
spatial, temporal, transition, and preference values. Although it hid total
scores and weights, those values could let Qwen restate the rule teacher.

`run_llm_value_teacher.py` was therefore advanced to
`value_pairwise_v4_raw_context`. It removes every rule-derived candidate field:
model score, rule score, rule weights, spatial, temporal, category transition,
user preference, popularity, recent-rate features, and previous-category match.
The remaining prompt contains only strict-past category sequence, query time,
previous category, candidate category, and candidate distance. The LLM must
emit a pairwise preference, confidence, abstention flag, controlled evidence
codes, and a short reason.

The frozen validation selector is unchanged: rule Top-1 override, fused margin
in the bottom 40% of overrides, and backbone margin at least the median. It
selects 80 NYC A4 validation conflicts. The 24-request deterministic pilot had
15 distill-train, 5 calibration, and 4 audit samples.

| Measure | Value |
|---|---:|
| Qwen requests | 24 |
| Estimated cost | 0.0058536 CNY |
| Hard budget | 0.03 CNY |
| Actual cost | 0.00310725 CNY |
| Backbone-correct | 2 |
| Rule-teacher-correct | 5 |
| Qwen-correct | 1 |
| Qwen delta versus rule teacher | -4 |
| Pairwise oracle delta versus rule teacher | +2 |
| Audit Qwen-correct / rule-correct | 0 / 1 |
| Qwen abstentions | 1 / 24 |

The v4 teacher selected the backbone 11 times and the rule candidate 13 times.
Most explanations repeated generic recent-pattern language; they did not turn
the pairwise oracle space into correct decisions. These results do not support
labelling the remaining 56 v4 cases, fitting a local student, or evaluating
v4 on test.

## Why external POI semantics is the remaining credible direction

The raw NYC source preserves Foursquare venue IDs in `POI_id`, but the model
pipeline converts them to anonymous POI indices and retains only category,
coordinates, and time. The raw source has no venue-name field. Foursquare's
current open Places schema documents that its `fsq_place_id` can be joined to
venue name and address, but access now requires a Places Portal account and an
access token. Do not create an account, store a token, or call a paid endpoint
without user-provided authority.

This is materially different from another prompt rewrite: venue names and
addresses could add externally grounded semantics such as a station, campus,
office, or named restaurant. Without them, the LLM sees no information that a
local rule/student does not already have.

## Status and safe next action

No more Qwen calls should be made under the current category-only input. The
combined actual Qwen spend is 0.00503985 CNY, far below the user-authorized
5 CNY ceiling.

If a user supplies either of the following, the next experiment can be a new
validation-only 24-case semantic pilot with its own preflight and cache:

1. A Foursquare Places access token authorized for this research purpose; or
2. A local UTF-8 CSV/Parquet export with `fsq_place_id`, `name`, and optional
   `address`/`locality` fields, licensed for this use.

The semantic join must use only candidate and strict-past history POIs. Test
queries remain excluded until the semantic protocol, pilot, calibration, audit,
and any local student are frozen.
