"""Cached, budgeted, CPU-only Jev arbitration on frozen NYC A4 predictions.

See docs/15_jev_nyc_protocol.md. API payload construction never receives labels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import pandas as pd
import requests
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rule_teacher import (DEFAULT_RULE_NAMES, DatasetRuleStatistics, RankingBatch,
                          evaluate_ranking, normalize_rows, rerank_candidates)
from rule_teacher.core import _haversine_km

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'server_artifacts/jev_nyc_20260920'
CAND = ROOT / 'server_artifacts/nyc_rule_teacher_20260723/nyc_a4'
DATA = ROOT / 'data/nyc/preprocessed'
MODEL = 'jev-1.13.0'
PRICE = 0.042 / 1_000_000
RESERVE = 64_000 * PRICE
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
VARIANTS = ['evidence', 'context']


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def partition(user):
    bucket = int(hashlib.sha256(f'jev-nyc-v1:user:{int(user)}'.encode()).hexdigest(), 16) % 10
    return 'selection' if bucket < 6 else 'calibration' if bucket < 8 else 'audit'


def safe_float(value):
    value = float(value)
    return round(value, 6) if math.isfinite(value) else None


def rank_values(candidates, labels, fallback):
    matched = candidates == labels[:, None]
    return np.where(matched.any(axis=1), matched.argmax(axis=1) + 1, fallback)


def promote(rule, base, probabilities, threshold):
    result = rule.copy()
    switch = (probabilities.argmax(axis=1) == 0) & (probabilities[:, 0] >= threshold)
    switch &= base[:, 0] != rule[:, 0]
    for i in np.flatnonzero(switch):
        chosen = base[i, 0]
        result[i] = np.r_[chosen, rule[i][rule[i] != chosen]]
    if not np.array_equal(np.sort(result, axis=1), np.sort(rule, axis=1)):
        raise AssertionError('Candidate membership changed')
    return result, switch


def paired(ranks, reference, users):
    d1 = (ranks == 1).astype(float) - (reference == 1)
    dm = 1.0 / ranks - 1.0 / reference
    unique, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse)
    sums1 = np.bincount(inverse, weights=d1)
    sumsm = np.bincount(inverse, weights=dm)
    rng = np.random.default_rng(20260920)
    values1, valuesm = [], []
    for _ in range(2000):
        draws = rng.integers(0, len(unique), len(unique))
        denom = counts[draws].sum()
        values1.append(float(sums1[draws].sum() / denom))
        valuesm.append(float(sumsm[draws].sum() / denom))
    rescued, lost = int((d1 > 0).sum()), int((d1 < 0).sum())
    return dict(net_top1=rescued-lost, rescued=rescued, lost=lost,
                delta_r1=float(d1.mean()), delta_mrr=float(dm.mean()),
                user_cluster_ci_r1=np.quantile(values1, [.025, .975]).tolist(),
                user_cluster_ci_mrr=np.quantile(valuesm, [.025, .975]).tolist(),
                mcnemar_exact_p_descriptive=float(binomtest(rescued, rescued+lost).pvalue) if rescued+lost else 1.0)


def reliability(probabilities, outcomes):
    if not len(outcomes):
        return {}
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == outcomes
    bins, ece = [], 0.0
    for lo, hi in zip([0, .5, .6, .7, .8, .9], [.5, .6, .7, .8, .9, 1.00001]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.any():
            acc, conf = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += mask.mean()*abs(acc-conf)
            bins.append(dict(lower=lo, upper=min(hi, 1), n=int(mask.sum()), accuracy=acc, confidence=conf))
    return dict(n=len(outcomes), accuracy=float(correct.mean()),
                brier=float(((probabilities-np.eye(3)[outcomes])**2).sum(axis=1).mean()),
                nll=float(-np.log(np.clip(probabilities[np.arange(len(outcomes)), outcomes], 1e-12, 1)).mean()),
                ece=float(ece), bins=bins,
                outcome_counts=np.bincount(outcomes, minlength=3).tolist())


class BudgetClient:
    """Thread-safe reservations persisted before network I/O, including retries."""
    def __init__(self, output, cap=1.0):
        self.output, self.cap = Path(output), cap
        self.output.mkdir(parents=True, exist_ok=True)
        self.ledger = self.output / 'api_ledger.jsonl'
        self.lock = threading.Lock()
        self.thread = threading.local()
        self.halted = False
        self.charges = {}
        if self.ledger.exists():
            for line in self.ledger.read_text(encoding='utf-8').splitlines():
                item = json.loads(line)
                self.charges[item['attempt']] = item['charge_usd']

    def record(self, attempt, charge, **fields):
        item = dict(attempt=attempt, charge_usd=charge, utc=datetime.now(timezone.utc).isoformat(), **fields)
        with self.ledger.open('a', encoding='utf-8') as handle:
            handle.write(canonical(item)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        self.charges[attempt] = charge

    def total(self):
        with self.lock:
            return sum(self.charges.values())

    def query(self, payload, tag):
        request_hash = digest(payload)
        cache = self.output / 'cache' / f'{request_hash}.json'
        if cache.exists():
            saved = json.loads(cache.read_text(encoding='utf-8'))
            validate_answer(saved['response'])
            return saved
        if not hasattr(self.thread, 'session'):
            self.thread.session = requests.Session()
        key = os.environ.get('TYPESAFE_API_KEY')
        if not key:
            raise RuntimeError('TYPESAFE_API_KEY missing')
        last_error = 'No attempt'
        for retry in range(3):
            attempt = uuid.uuid4().hex
            with self.lock:
                if self.halted or sum(self.charges.values())+RESERVE > self.cap:
                    raise RuntimeError('Stopped by budget/authentication guard')
                self.record(attempt, RESERVE, status='reserved', request_hash=request_hash, tag=tag)
            started = time.perf_counter()
            try:
                response = self.thread.session.post('https://api.typesafe.ai/v1/systemone',
                    headers={'Authorization': f'Bearer {key}'}, json=payload, timeout=(10, 45))
            except requests.RequestException as exc:
                last_error = type(exc).__name__
                time.sleep(min(2**retry, 8))
                continue
            latency = time.perf_counter()-started
            if response.status_code != 200:
                last_error = f'HTTP {response.status_code}'
                if response.status_code in (401, 402, 403):
                    with self.lock:
                        self.halted = True
                    raise RuntimeError(last_error)
                if response.status_code == 429 or response.status_code >= 500:
                    try:
                        delay = float(response.headers.get('Retry-After', 2**retry))
                    except ValueError:
                        from email.utils import parsedate_to_datetime
                        delay = (parsedate_to_datetime(response.headers['Retry-After'])-datetime.now(timezone.utc)).total_seconds()
                    if delay > 60:
                        raise RuntimeError(f'{last_error}; long Retry-After: {delay}')
                    time.sleep(max(1, delay))
                    continue
                raise RuntimeError(last_error)
            data = response.json()
            tokens = int(data['usage']['input_tokens'])
            with self.lock:
                self.record(attempt, tokens*PRICE, status='success', input_tokens=tokens,
                    request_hash=request_hash, tag=tag, latency_s=latency,
                    request_id=response.headers.get('x-typesafe-request-id'))
            validate_answer(data)
            saved = dict(request_hash=request_hash, request=payload, response=data, latency_s=latency,
                         request_id=response.headers.get('x-typesafe-request-id'))
            save(cache, saved)
            return saved
        raise RuntimeError(f'Retries exhausted: {last_error}')


def validate_answer(response):
    if response['model'] != MODEL:
        raise ValueError('Unexpected Jev model version')
    answer = response['answers']['next_place']
    p = answer['probabilities']
    if set(p) != {'A', 'B', 'neither'} or answer['choice'] not in p:
        raise ValueError('Incorrect answer schema')
    values = np.array([p['A'], p['B'], p['neither']], float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or abs(values.sum()-1) > .02:
        raise ValueError('Invalid probabilities')
    return values / values.sum()


class NYC:
    def __init__(self):
        self.summary = json.loads((CAND/'rule_teacher_summary.json').read_text())
        self.stats = DatasetRuleStatistics.from_preprocessed_dir('nyc', str(DATA))
        self.names = self.stats.train_events.groupby('PoiCategoryId')['PoiCategoryName'].agg(lambda x: str(x.mode().iloc[0])).to_dict()
        self.histories = {int(u): g.sort_values('UTCTimeOffsetEpoch') for u,g in self.stats.history_events.groupby('UserId')}
        self.weights = np.array([self.summary['rule_search']['weights'][x] for x in DEFAULT_RULE_NAMES])
        self.strength = self.summary['rule_search']['rule_strength']

    def load(self, split):
        batch = RankingBatch.load(str(CAND/f'{split}_candidates.npz'))
        queries = pd.read_csv(DATA/('validate_sample.csv' if split == 'validation' else 'test_sample.csv'))
        if batch.sample_indices is not None:
            queries = queries.iloc[batch.sample_indices.astype(int)]
        queries = queries.reset_index(drop=True)
        if not np.array_equal(queries.PoiId.to_numpy(), batch.labels):
            raise AssertionError('Candidate/label alignment failed')
        features = self.stats.score_candidates(queries, batch.candidate_ids)
        rule, rule_scores = rerank_candidates(batch, features, DEFAULT_RULE_NAMES, self.weights, self.strength)
        expected_base = self.summary[f'base_{split}_metrics']
        expected_rule = self.summary['rule_search']['validation_metrics'] if split == 'validation' else self.summary['rule_test_metrics']
        for predictions, expected in [(batch.candidate_ids, expected_base), (rule, expected_rule)]:
            measured = evaluate_ranking(predictions, batch.labels, batch.base_label_ranks)
            for key in expected:
                if abs(measured[key]-expected[key]) > 1e-10:
                    raise AssertionError(f'Archived metric mismatch: {split} {key}')
        if split == 'test':
            archived = np.load(CAND/'test_reranked.npz')['candidate_ids']
            if not np.array_equal(archived, rule):
                raise AssertionError('Test rule ranking differs from archive')
        normalized = normalize_rows(batch.model_scores)
        fused = (1-self.strength)*normalized+self.strength*sum(self.weights[k]*features[name] for k,name in enumerate(DEFAULT_RULE_NAMES))
        conflicts = np.flatnonzero(rule[:,0] != batch.candidate_ids[:,0])
        return dict(batch=batch, queries=queries, features=features, rule=rule, fused=fused,
                    model_scores=normalized, rule_scores=rule_scores, conflicts=conflicts,
                    partitions=np.array([partition(u) for u in queries.UserId]))

    def make_record(self, split, d, index):
        """Allowlisted context only; never read label/target coordinates/category."""
        q = d['queries'].iloc[index]
        user, previous_time, query_time = int(q.UserId), int(q.last_checkin_epoch_time), int(q.UTCTimeOffsetEpoch)
        past = self.histories[user]
        past = past[(past.UTCTimeOffsetEpoch <= previous_time) & (past.UTCTimeOffsetEpoch < query_time)]
        previous = past.iloc[-1] if len(past) else None
        recent = past.tail(8)
        history = [dict(category=self.names.get(int(row.PoiCategoryId), 'unknown'),
                        hours_before_query=safe_float((query_time-int(row.UTCTimeOffsetEpoch))/3600))
                   for row in recent.itertuples()]
        stamp = pd.Timestamp(q.UTCTimeOffset)
        candidates, numeric = [], []
        for poi in [d['batch'].candidate_ids[index,0], d['rule'][index,0]]:
            poi = int(poi)
            col = int(np.flatnonzero(d['batch'].candidate_ids[index] == poi)[0])
            cat = self.stats.poi_category.get(poi, -1)
            distance = None
            if previous is not None and poi in self.stats.poi_latitude:
                distance = safe_float(_haversine_km(previous.Latitude, previous.Longitude,
                    self.stats.poi_latitude[poi], self.stats.poi_longitude[poi]))
            total = max(self.stats.user_totals.get(user, 0), 1)
            candidate = dict(category=self.names.get(cat, 'unknown'), distance_from_last_km=distance,
                training_user_category_fraction=safe_float(self.stats.user_category_counts[user].get(cat,0)/total),
                training_user_poi_fraction=safe_float(self.stats.user_poi_counts[user].get(poi,0)/total),
                visits_in_recent_8=int((recent.PoiId == poi).sum()),
                same_as_last_place=bool(previous is not None and int(previous.PoiId) == poi),
                model_score=safe_float(d['model_scores'][index,col]),
                fused_rule_score=safe_float(d['fused'][index,col]),
                rule_components={name:safe_float(d['features'][name][index,col]) for name in DEFAULT_RULE_NAMES})
            candidates.append(candidate)
            numeric.extend([distance if distance is not None else -1,
                candidate['training_user_category_fraction'], candidate['training_user_poi_fraction'],
                candidate['visits_in_recent_8'], float(candidate['same_as_last_place']),
                candidate['model_score'], candidate['fused_rule_score'],
                *candidate['rule_components'].values()])
        model_margin = float(d['model_scores'][index,0]-d['model_scores'][index,1])
        rule_margin = float(d['rule_scores'][index,0]-d['rule_scores'][index,1])
        numeric.extend([model_margin, rule_margin, math.sin(stamp.hour*math.pi/12), math.cos(stamp.hour*math.pi/12), float(stamp.weekday()>=5)])
        return dict(index=int(index), sample_id=int(q.check_ins_id), user=user,
            model_is_a=int(digest([split,int(q.check_ins_id),'order']),16)%2 == 0,
            context=dict(city='New York City', query_hour=int(stamp.hour), query_weekday=int(stamp.weekday()),
                recent_visits_oldest_to_newest=history,
                historical_typical_distance_km=safe_float(self.stats.user_distance_scale_km.get(user,self.stats.dataset_distance_scale_km))),
            candidates=candidates, numeric=numeric,
            model_margin=model_margin, rule_margin=rule_margin,
            history_max_epoch=int(past.UTCTimeOffsetEpoch.max()) if len(past) else None,
            query_epoch=query_time, previous_epoch=previous_time)


def payload_for(record, variant, swap=False):
    model_is_a = record['model_is_a'] != swap
    order = [0,1] if model_is_a else [1,0]
    candidates = {}
    for letter,j in zip(['A','B'], order):
        candidate = dict(record['candidates'][j])
        if variant == 'context':
            for key in ['model_score','fused_rule_score','rule_components']:
                candidate.pop(key)
        candidates[letter] = candidate
    state = dict(record['context'], candidates=candidates)
    if variant == 'evidence':
        state['score_legend'] = 'Scores and rule components are min-max normalized within the frozen Top-20, not probabilities. Higher means stronger support. All statistical priors use the training split.'
    instructions = ('Predict the next checked-in place at the supplied query time. Judge the two candidates using the observed history and evidence. '
        'A category is not a venue identity; two candidates with the same category can be different places. '
        'Nearby or popular does not automatically mean correct. Do not invent venue names or intentions. '
        'Choose neither when the next place is likely outside these two candidates. Account for ambiguity in the probabilities.')
    return dict(model=MODEL, state=state, questions={'next_place':dict(type='choice',instructions=instructions,
        criteria={'A':'Candidate A is the next checked-in place.', 'B':'Candidate B is the next checked-in place.',
                  'neither':'The next checked-in place is neither candidate A nor candidate B.'})})


def run_requests(client, records, variants, swap=False):
    tasks = [(r,v) for r in records for v in variants]
    results = {v:{} for v in variants}
    def work(record, variant):
        saved = client.query(payload_for(record,variant,swap), f'{variant}:{record["sample_id"]}:swap={swap}')
        p = validate_answer(saved['response'])
        if not (record['model_is_a'] != swap):
            p = p[[1,0,2]]
        return record['index'], variant, p.tolist(), saved['request_hash']
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(work,r,v) for r,v in tasks]
        for n,future in enumerate(as_completed(futures),1):
            i,v,p,h = future.result()
            results[v][str(i)] = dict(probabilities=p, request_hash=h)
            if n%50 == 0 or n == len(tasks):
                print(canonical(dict(completed=n,total=len(tasks),swap=swap,budget_used_usd=client.total())),flush=True)
    return results


def full_prob(d, result):
    p = np.tile([0.0,1.0,0.0], (len(d['queries']),1))
    for i,entry in result.items():
        p[int(i)] = entry['probabilities']
    return p


def metrics(d, candidates, mask):
    b = d['batch']
    return evaluate_ranking(candidates[mask], b.labels[mask], b.base_label_ranks[mask])


def choose_threshold(d, p):
    mask = d['partitions'] == 'calibration'
    trials = []
    for t in THRESHOLDS:
        candidate,_ = promote(d['rule'],d['batch'].candidate_ids,p,t)
        m = metrics(d,candidate,mask)
        trials.append(dict(threshold=t,metrics=m))
    chosen = max(trials,key=lambda x:(x['metrics']['recall_at_1'],x['metrics']['mrr'],x['threshold']))
    return chosen['threshold'], trials


def report_method(d, p, threshold, partitions=True):
    b = d['batch']
    predictions,switch = promote(d['rule'], b.candidate_ids,p,threshold)
    ranks = rank_values(predictions,b.labels,b.base_label_ranks)
    reference = rank_values(d['rule'],b.labels,b.base_label_ranks)
    outcomes = np.where(b.candidate_ids[:,0] == b.labels,0,np.where(d['rule'][:,0] == b.labels,1,2))
    output = {}
    for part in (['selection','calibration','audit','all'] if partitions else ['all']):
        mask = np.ones(len(ranks),bool) if part == 'all' else d['partitions'] == part
        conflict_mask = mask & (d['rule'][:,0] != b.candidate_ids[:,0])
        output[part] = dict(n=int(mask.sum()),n_users=int(d['queries'].UserId[mask].nunique()),
            n_conflicts=int(conflict_mask.sum()),switches=int(switch[mask].sum()),
            metrics=metrics(d,predictions,mask),base=metrics(d,b.candidate_ids,mask),rule=metrics(d,d['rule'],mask),
            paired_vs_rule=paired(ranks[mask],reference[mask],d['queries'].UserId.to_numpy()[mask]),
            probability_reliability=reliability(p[conflict_mask],outcomes[conflict_mask]))
    return output


def provenance(split):
    paths = [Path(__file__), ROOT/'docs/15_jev_nyc_protocol.md', ROOT/'rule_teacher/core.py',
             CAND/'rule_teacher_summary.json', CAND/f'{split}_candidates.npz',DATA/'sample.csv',
             DATA/('validate_sample.csv' if split == 'validation' else 'test_sample.csv')]
    return {str(p.relative_to(ROOT)):file_hash(p) for p in paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare','validation','test'])
    args = ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    nyc = NYC()
    split = 'test' if args.stage == 'test' else 'validation'
    d = nyc.load(split)
    records = [nyc.make_record(split,d,int(i)) for i in d['conflicts']]
    hashes = provenance(split)
    if args.stage == 'prepare':
        payloads = [payload_for(r,v) for r in records for v in VARIANTS]
        stats = dict(split=split,n=len(d['queries']),conflicts=len(records),
            partitions={k:int((d['partitions']==k).sum()) for k in ['selection','calibration','audit']},
            conflict_partitions={k:int(sum(partition(r['user'])==k for r in records)) for k in ['selection','calibration','audit']},
            requests=len(payloads)+64,
            prompt_chars_min=min(len(canonical(p)) for p in payloads),
            prompt_chars_max=max(len(canonical(p)) for p in payloads),
            prompt_chars_total=sum(len(canonical(p)) for p in payloads),
            approximate_cost_at_one_token_per_char_usd=sum(len(canonical(p)) for p in payloads)*PRICE,
            hard_budget_usd=1.0,max_concurrency=2,source_hashes=hashes)
        save(OUT/'preflight.json',stats)
        save(OUT/'example_request.json',payloads[0])
        print(json.dumps(stats,indent=2))
        return
    client = BudgetClient(OUT)
    if args.stage == 'validation':
        if (OUT/'frozen_selection.json').exists():
            raise RuntimeError('Selection already frozen; will not overwrite')
        preflight = json.loads((OUT/'preflight.json').read_text())
        if hashes != preflight['source_hashes']:
            raise RuntimeError('Source changed since preflight; rerun prepare before API work')
        results = run_requests(client,records,VARIANTS)
        save(OUT/'validation_predictions.json',results)
        probabilities = {v:full_prob(d,results[v]) for v in VARIANTS}
        selection = d['partitions']=='selection'
        def objective(v):
            predictions,_ = promote(d['rule'],d['batch'].candidate_ids,probabilities[v],.5)
            m = metrics(d,predictions,selection)
            return m['recall_at_1'],m['mrr']
        chosen = max(VARIANTS,key=objective)
        threshold,trials = choose_threshold(d,probabilities[chosen])
        raw_reports = {v:report_method(d,p,.5) for v,p in probabilities.items()}
        selected_report = report_method(d,probabilities[chosen],threshold)
        eligible = (selected_report['calibration']['paired_vs_rule']['net_top1']>=0
            and selected_report['calibration']['paired_vs_rule']['delta_mrr']>=0
            and selected_report['audit']['paired_vs_rule']['net_top1']>0
            and selected_report['audit']['paired_vs_rule']['delta_mrr']>=0)
        ordered = sorted(records,key=lambda r:digest(['permutation',r['sample_id']]))[:32]
        swapped = run_requests(client,ordered,VARIANTS,swap=True)
        save(OUT/'validation_swapped_predictions.json',swapped)
        permutation = {}
        for v in VARIANTS:
            original = np.array([results[v][str(r['index'])]['probabilities'] for r in ordered])
            alternative = np.array([swapped[v][str(r['index'])]['probabilities'] for r in ordered])
            permutation[v] = dict(n=len(ordered),argmax_flip_rate=float((original.argmax(1)!=alternative.argmax(1)).mean()),
                mean_absolute_probability_change=float(np.abs(original-alternative).mean()),
                max_absolute_probability_change=float(np.abs(original-alternative).max()))
        X = np.array([r['numeric'] for r in records])
        indices = np.array([r['index'] for r in records])
        b = d['batch']
        y = np.where(b.candidate_ids[indices,0]==b.labels[indices],0,np.where(d['rule'][indices,0]==b.labels[indices],1,2))
        fit_mask = d['partitions'][indices]=='selection'
        clf = make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=20260920))
        clf.fit(X[fit_mask],y[fit_mask])
        local_p = np.tile([0.,1.,0.],(len(d['queries']),1))
        for j,label in enumerate(clf.classes_):
            local_p[indices,int(label)] = clf.predict_proba(X)[:,j]
        margin_p = np.tile([0.,1.,0.],(len(d['queries']),1))
        for r in records:
            pm = 1/(1+math.exp(-5*(r['model_margin']-r['rule_margin'])))
            margin_p[r['index']] = [pm,1-pm,0]
        local_t,local_trials = choose_threshold(d,local_p)
        margin_t,margin_trials = choose_threshold(d,margin_p)
        local_report = report_method(d,local_p,local_t)
        margin_report = report_method(d,margin_p,margin_t)
        # Portable model parameters, no unsafe pickle or environment coupling.
        scaler,logistic = clf.steps[0][1],clf.steps[1][1]
        local_model = dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
            coef=logistic.coef_.tolist(),intercept=logistic.intercept_.tolist(),classes=logistic.classes_.tolist())
        freeze = dict(model=MODEL,prompt=chosen,threshold=threshold,eligible_for_deployment=bool(eligible),
            local_threshold=local_t,margin_threshold=margin_t,local_model=local_model,
            source_hashes=hashes,validation_predictions_sha256=file_hash(OUT/'validation_predictions.json'),
            created_utc=datetime.now(timezone.utc).isoformat(),
            test_policy='Single exploratory historical test; report frozen threshold and predeclared 0.5, no retuning.')
        save(OUT/'frozen_selection.json',freeze)
        report = dict(raw_fixed_threshold_05=raw_reports,selected=selected_report,selected_prompt=chosen,
            selected_threshold=threshold,threshold_trials=trials,eligible_for_deployment=bool(eligible),
            local_classifier=local_report,local_threshold=local_t,local_trials=local_trials,
            margin_gate=margin_report,margin_threshold=margin_t,margin_trials=margin_trials,
            permutation=permutation,cost_usd=client.total())
        save(OUT/'validation_report.json',report)
        print(canonical(dict(stage='validation_complete',prompt=chosen,threshold=threshold,
            eligible=bool(eligible),audit=selected_report['audit']['paired_vs_rule'],cost_usd=client.total())),flush=True)
    else:
        if (OUT/'test_report.json').exists():
            raise RuntimeError('Test report already exists; do not repeat test selection')
        freeze_path = OUT/'frozen_selection.json'
        freeze = json.loads(freeze_path.read_text())
        for name,sha in freeze['source_hashes'].items():
            if file_hash(ROOT/name)!=sha:
                raise RuntimeError(f'Frozen source changed: {name}')
        save(OUT/'test_preflight.json',dict(frozen_sha256=file_hash(freeze_path),source_hashes=hashes,
            n=len(d['queries']),conflicts=len(records),created_utc=datetime.now(timezone.utc).isoformat()))
        results = run_requests(client,records,[freeze['prompt']])
        save(OUT/'test_predictions.json',results)
        p = full_prob(d,results[freeze['prompt']])
        selected = report_method(d,p,freeze['threshold'],False)
        raw = report_method(d,p,.5,False)
        X = np.array([r['numeric'] for r in records])
        local = freeze['local_model']
        z = ((X-np.array(local['mean']))/np.array(local['scale']))@np.array(local['coef']).T+np.array(local['intercept'])
        z = np.exp(z-z.max(axis=1,keepdims=True)); z /= z.sum(axis=1,keepdims=True)
        lp = np.tile([0.,1.,0.],(len(d['queries']),1))
        mp = lp.copy()
        for j,r in enumerate(records):
            lp[r['index'],local['classes']] = z[j]
            pm = 1/(1+math.exp(-5*(r['model_margin']-r['rule_margin'])))
            mp[r['index']] = [pm,1-pm,0]
        report = dict(status='exploratory_reused_historical_test',selected=selected,raw_fixed_threshold_05=raw,
            local_classifier=report_method(d,lp,freeze['local_threshold'],False),
            margin_gate=report_method(d,mp,freeze['margin_threshold'],False),
            deployment_enabled=freeze['eligible_for_deployment'],cost_usd=client.total(),
            frozen_selection_sha256=file_hash(freeze_path))
        save(OUT/'test_report.json',report)
        print(canonical(dict(stage='test_complete',selected=selected['all']['paired_vs_rule'],
            raw=raw['all']['paired_vs_rule'],cost_usd=client.total())),flush=True)


if __name__ == '__main__':
    main()
