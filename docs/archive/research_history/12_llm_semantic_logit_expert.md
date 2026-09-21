# LLM Semantic Logit Expert

## Frozen design

The deployable LLM component is an offline semantic logit expert attached to
the frozen A4 backbone. Qwen Flash is used only once to annotate category
semantics. No user trajectory, validation label, test label, or POI identifier
is sent to the API.

For trajectory state `h` and the frozen profile `p_j` of candidate POI `j`,
the expert computes:

```text
q = Wq LayerNorm(h)
k_j = Wk Standardize(p_j)
z_j = q^T k_j / sqrt(rank)
g = sigmoid(w_g^T h + b_g)
logit_j = A4_logit_j + max_scale * tanh(alpha) * g * z_j
```

`alpha` is initialized to exactly zero. Consequently, the attached model is
bitwise identical to A4 before semantic training. A4 parameters are frozen,
and its BatchNorm and Dropout modules remain in evaluation mode. Only the two
projections, the trajectory gate, and `alpha` are trainable.

## Semantic data isolation

- Profile width: 24 (10 intent, 6 time, 4 mobility role, and 4 scalar fields).
- NYC profile names: 207/207.
- New API calls required to complete coverage: 10; 197 were cache hits.
- Estimated cost before calling: CNY 0.0052584.
- Actual incremental cost: CNY 0.004104.
- Hard budget used for the call: CNY 0.02.
- The POI-to-category join uses the modal category from training events only,
  matching the frozen rule-teacher convention.
- Matrix shape: 4,982 x 24; 4,937 POIs have train-derived profiles, while the
  padding/unknown sentinel and STHGCN's legacy extra output row are all-zero
  and unavailable.

## Local artifacts

- `build_llm_semantic_matrix.py`: strict, auditable profile-to-POI join.
- `model/llm_semantic_expert.py`: zero-initialized semantic residual expert.
- `conf/best_conf/nyc_r1_6_aux_free_dual_gate_llm_semantic.yml`: validation-only
  10k-step semantic training configuration.
- `server_artifacts/llm_category_profiles/nyc_semantic_profile_matrix_v2.npz`:
  generated profile matrix.

## GPU gate

GPU training must initialize from the frozen A4 checkpoint. The runner rejects
a frozen semantic run when neither `--backbone-checkpoint` nor a full semantic
resume checkpoint is supplied. The experiment config also sets `do_test:
false`, so the first run cannot accidentally inspect test metrics.

Validation checkpoints are compared against A4 and the frozen rule teacher.
The final implementation uses profile-distillation warmup and a non-negative
semantic residual. Full results, ablations, and the single frozen test are
reported in `docs/13_llm_profile_distillation_results.md`.
