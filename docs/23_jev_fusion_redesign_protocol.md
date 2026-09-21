# Jev fusion redesign: validation-only offline comparison

Fixed before this experiment's evaluation. Existing NYC/CA/TKY validation
responses only; no API calls and no reading historical test predictions.
NYC uses A4; CA/TKY use R0. Existing prompts and category preprocessing differ
between NYC and CA/TKY, so this is not a controlled comparison of cities.

## Candidates

1. Arithmetic category fusion as before; alpha grid
   [0,.01,.025,.05,.1,.2,.35,.5,.75,1].
2. Smoothed, temperature-adjusted logarithmic category pooling. Let P be
   backbone category mass and Q Jev conditional category probability.
   Smooth Q by .02 uniform mass, then normalize Q^(1/T), T in [1,2,4].
   C = normalize(P^(1-lambda) * Q_T^lambda), lambda in [.05,.1,.2,.35,.5].
   Redistribute C using backbone within-category probabilities. Include
   lambda=0 as exact backbone. This dampens extreme arithmetic likelihood
   ratios and preserves within-category order. Temperature is selected
   using ranking utility; do not claim formal probability calibration.
3. Adaptive expected-gain model: standardized linear ridge regression
   predicts the per-query utility gain of each candidate logarithmic action.
   Utility = Top-1 correctness + .25 reciprocal rank. Backbone gain is zero.
   Select the action with highest predicted gain; ties prefer backbone.
   Inputs contain only score-distribution features, not labels or IDs.
   Ridge penalties [1,10,100,1000], selected inside training users only.

Inputs: POI margin/entropy, backbone and Jev category margin/entropy, category
agreement, Jensen-Shannon divergence, outside probability, category count,
Jev probability of backbone's top category, and backbone probability of
Jev's top category. Original outside probability is only an input feature,
not a trusted confidence or handcrafted threshold. All out-of-candidate
targets remain included, with zero possible ranking gain.

## Evaluation

Three predetermined user-fold repeats (seeds 20260921, 20260922, 20260923).
Five outer folds per repeat. Each global fusion parameter is selected using
outer-training users only. Adaptive penalty is chosen with three inner
user folds, with normalization fitted independently inside each fold.
Selection objective for ALL methods = R@1 + .25 MRR; ties prefer weaker
intervention. Every user is fully held out when making their outer-fold
prediction. Never train on an outer user's labels or use IDs as features.

Report out-of-fold R@1/5/MRR, gains vs backbone and vs arithmetic, per-repeat
results, first-repeat user-cluster bootstrap intervals and action frequencies.
Repeated folds reuse observations: do not treat repetitions as extra samples.
This is exploratory development on already inspected validation data, not
an untouched test or a guarantee of gains. No runtime integration enabled.
No additional variants will be added after seeing this run's results.
