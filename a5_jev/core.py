"""CPU-only fusion. No labels enter inference; lambda=0 is exact identity."""
import numpy as np

GRID = [(0., 1.)] + [(w, t) for w in (.05, .1, .2, .35, .5) for t in (1., 2., 4.)]


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite scores')
    p = np.exp(x - x.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def fuse(candidate_ids, logits, categories, predictions, weight, temperature=1., smoothing=.02):
    ids = np.asarray(candidate_ids)
    logits = np.asarray(logits, dtype=float)
    if ids.ndim != 2 or logits.shape != ids.shape or len(categories) != len(ids) or len(predictions) != len(ids):
        raise ValueError('Unaligned candidates, logits, categories or predictions')
    if not np.isfinite([weight, temperature, smoothing]).all() or not 0 <= weight <= 1 or temperature <= 0 or not 0 <= smoothing < 1:
        raise ValueError('Invalid fusion parameters')
    if any(len(set(row)) != len(row) for row in ids) or not np.isfinite(logits).all():
        raise ValueError('Duplicate candidates or nonfinite logits')
    if weight == 0:
        return ids.copy()
    out = ids.copy()
    for i, (names, prediction) in enumerate(zip(categories, predictions)):
        if len(names) != ids.shape[1]:
            raise ValueError('Category count differs from candidate count')
        unique, inverse = np.unique(names, return_inverse=True)
        supplied = dict(zip(prediction['categories'], prediction['probabilities']))
        if len(supplied) != len(prediction['categories']) or len(prediction['probabilities']) != len(supplied) or set(supplied) != set(unique):
            raise ValueError('Category mapping mismatch')
        q = np.array([supplied[c] for c in unique], dtype=float)
        if not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
            raise ValueError('Invalid Jev probability')
        if q.sum() == 0:
            continue  # No candidate-category evidence: exact backbone fallback.
        q /= q.sum()
        q = (1-smoothing)*q + smoothing/len(q)
        # Log-space implementation avoids underflow in tiny backbone category mass.
        z = logits[i] - logits[i].max()
        log_p = z - np.log(np.exp(z).sum())
        log_category = np.array([np.logaddexp.reduce(log_p[inverse == j]) for j in range(len(unique))])
        log_q = np.log(np.maximum(q, np.finfo(float).tiny))/temperature
        score = log_p + weight*(log_q[inverse]-log_category[inverse])
        out[i] = ids[i, np.argsort(-score, kind='stable')]
    return out


def select_parameters(batch, categories, predictions, mask):
    from rule_teacher import evaluate_ranking
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError('Empty selection split')
    trials = []
    for weight, temperature in GRID:
        ranked = fuse(batch.candidate_ids, batch.model_scores, categories, predictions, weight, temperature)
        metrics = evaluate_ranking(ranked[mask], batch.labels[mask], batch.base_label_ranks[mask])
        trials.append(dict(weight=weight, temperature=temperature, metrics=metrics,
                           objective=metrics['recall_at_1']+.25*metrics['mrr']))
    best = max(trials, key=lambda r: (r['objective'], -r['weight'], -r['temperature']))
    return best, trials
