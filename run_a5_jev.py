"""A5-Jev staged runner: prepare -> query-validation -> select -> query-test -> report.

One process per output directory; interrupted queries resume immutable responses.
All prices are estimates, with unknown attempts conservatively reserved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import numpy as np
from a5_jev.core import fuse, select_parameters
from a5_jev.data import Dataset, digest, save, sha256
from a5_jev.client import Client, PRICE
from rule_teacher import evaluate_ranking

ROOT = Path(__file__).resolve().parent


def source_hashes():
    paths = [Path(__file__), *sorted((ROOT/'a5_jev').glob('*.py')), ROOT/'rule_teacher/core.py']
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metrics(data, ranked, mask):
    b = data.batch
    return evaluate_ranking(ranked[mask], b.labels[mask], b.base_label_ranks[mask])


def assess(data, predictions, weight, temperature):
    b = data.batch
    ranked = fuse(b.candidate_ids, b.model_scores, data.categories, predictions, weight, temperature)
    output = {}
    parts = ['selection','calibration','audit','all'] if data.split == 'validation' else ['all']
    for name in parts:
        mask = np.ones(len(ranked),bool) if name == 'all' else data.partitions == name
        if not mask.any():
            raise ValueError(f'Empty {name} partition')
        m, base = metrics(data, ranked, mask), metrics(data, b.candidate_ids, mask)
        assert m['recall_at_20'] == base['recall_at_20']
        before, after = b.candidate_ids[mask,0] == b.labels[mask], ranked[mask,0] == b.labels[mask]
        output[name] = dict(n=int(mask.sum()), metrics=m, backbone=base,
                            rescued=int((after & ~before).sum()), lost=int((before & ~after).sum()),
                            net_top1=int(after.sum()-before.sum()), delta_mrr=m['mrr']-base['mrr'])
    return output


def check_prepare(args):
    manifest = load_json(args.output/'prepared.json')
    if manifest['dataset'] != args.dataset or manifest['source_hashes'] != source_hashes():
        raise ValueError('Dataset/source changed since preparation; use a new run directory')
    if manifest['candidate_manifest_hash'] != sha256(args.candidates/'candidate_manifest.json'):
        raise ValueError('Candidate manifest changed')
    return manifest


def check_data(data, prepared):
    if data.provenance() != prepared['splits'][data.split]:
        raise ValueError('Input data/candidates changed since preparation')


def verify_predictions(data, path, prepared):
    value = load_json(path)
    if value['provenance'] != prepared['splits'][data.split] or len(value['predictions']) != len(data.queries):
        raise ValueError('Wrong prediction provenance or count')
    for i, item in enumerate(value['predictions']):
        payload, names = data.request(i)
        if item['request_hash'] != digest(payload) or item['categories'] != names:
            raise ValueError('Prediction/request alignment mismatch')
    return value['predictions']


def query(data, client):
    output = [None]*len(data.queries)
    def work(i):
        payload, names = data.request(i)
        p = client.query(payload)
        return i, dict(categories=names, probabilities=p[:-1], outside_probability=p[-1], request_hash=digest(payload))
    # Only two queued requests: a terminal error does not drain thousands of futures.
    with ThreadPoolExecutor(max_workers=2) as pool:
        indices = iter(range(len(output)))
        pending = {pool.submit(work,i) for i in [next(indices,None), next(indices,None)] if i is not None}
        count = 0
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    i, prediction = future.result(); output[i] = prediction; count += 1
                    if count % 100 == 0 or count == len(output):
                        print(json.dumps(dict(completed=count,total=len(output),split=data.split,reserved_cost_usd=client.total())),flush=True)
                    i = next(indices,None)
                    if i is not None:
                        pending.add(pool.submit(work,i))
        except BaseException:
            client.stop.set()
            for future in pending: future.cancel()
            raise
    return output


def execute(args):
    args.output.mkdir(parents=True,exist_ok=True)
    prepared_path = args.output/'prepared.json'
    if args.stage == 'prepare':
        if prepared_path.exists():
            raise FileExistsError('Preparation is immutable; reuse stages or choose a new directory')
        provenance = load_json(args.candidates/'candidate_manifest.json')
        if provenance.get('backbone') != 'A4' or provenance.get('smoke_only') or provenance.get('dataset') != args.dataset:
            raise ValueError('Formal A5 requires complete A4 exports, not R0 or smoke candidates')
        prepared = dict(version='A5-Jev',dataset=args.dataset,source_hashes=source_hashes(),
                        candidate_manifest_hash=sha256(args.candidates/'candidate_manifest.json'),splits={},cap_usd=args.cap_usd)
        for split in ('validation','test'):
            data = Dataset(args.dataset,args.data,args.candidates,split)
            if sha256(data.candidate_path) != provenance['candidate_hashes'][split] or sha256(data.events_path) != provenance['sample_hash']:
                raise ValueError('Export/data provenance mismatch')
            if sha256(data.query_path) != provenance['query_hashes'][split]:
                raise ValueError('Query data differs from exported data')
            prepared['splits'][split] = data.provenance()
        data = Dataset(args.dataset,args.data,args.candidates,'validation')
        lengths = [len(json.dumps(data.request(i)[0],ensure_ascii=False)) for i in range(len(data.queries))]
        if max(lengths)>32000: raise ValueError('Request too long')
        prepared['validation_estimate_one_token_per_character_usd'] = sum(lengths)*PRICE
        save(prepared_path,prepared)
        save(args.output/'example_request.json',data.request(0)[0])
        print(json.dumps(prepared,indent=2)); return
    prepared = check_prepare(args)
    split = 'test' if args.stage in ('query-test','report') else 'validation'
    data = Dataset(args.dataset,args.data,args.candidates,split)
    check_data(data,prepared)
    if args.stage in ('query-validation','query-test'):
        if split == 'test':
            frozen = load_json(args.output/'frozen.json')
            if frozen['prepared_hash'] != sha256(prepared_path): raise ValueError('Frozen run mismatch')
            if not frozen['eligible_for_test']: raise RuntimeError('Validation rejected; test queries disabled')
            if frozen['validation_predictions_hash'] != sha256(args.output/'validation_predictions.json'): raise ValueError('Validation predictions changed')
        path = args.output/f'{split}_predictions.json'
        if path.exists():
            verify_predictions(data,path,prepared); print('Complete cached predictions verified'); return
        client = Client(args.output,prepared['cap_usd'],args.offline)
        predictions = query(data,client)
        save(path,dict(provenance=data.provenance(),predictions=predictions,cost_usd=client.total())); return
    if args.stage == 'select':
        if (args.output/'frozen.json').exists(): raise FileExistsError('Selection already frozen')
        predictions = verify_predictions(data,args.output/'validation_predictions.json',prepared)
        best,trials = select_parameters(data.batch,data.categories,predictions,data.partitions=='selection')
        result = assess(data,predictions,best['weight'],best['temperature'])
        c,a = result['calibration'],result['audit']
        eligible = best['weight']>0 and c['net_top1']>=0 and c['delta_mrr']>=0 and a['net_top1']>0 and a['delta_mrr']>=0
        save(args.output/'frozen.json',dict(version='A5-Jev',weight=best['weight'],temperature=best['temperature'],smoothing=.02,
            eligible_for_test=eligible,prepared_hash=sha256(prepared_path),validation_predictions_hash=sha256(args.output/'validation_predictions.json')))
        save(args.output/'validation_report.json',dict(selected=result,trials=trials,eligible_for_test=eligible))
        print(json.dumps(dict(weight=best['weight'],temperature=best['temperature'],eligible_for_test=eligible,result=result),indent=2)); return
    frozen = load_json(args.output/'frozen.json')
    if not frozen['eligible_for_test'] or frozen['prepared_hash'] != sha256(prepared_path):
        raise ValueError('Test not enabled or frozen provenance mismatch')
    if frozen['validation_predictions_hash'] != sha256(args.output/'validation_predictions.json'): raise ValueError('Validation changed')
    if (args.output/'test_report.json').exists(): raise FileExistsError('Final test report already exists')
    predictions = verify_predictions(data,args.output/'test_predictions.json',prepared)
    result = assess(data,predictions,frozen['weight'],frozen['temperature'])
    save(args.output/'test_report.json',dict(version='A5-Jev',frozen_hash=sha256(args.output/'frozen.json'),
        predictions_hash=sha256(args.output/'test_predictions.json'),selected=result,active_integration=False))
    print(json.dumps(result,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','query-validation','select','query-test','report'])
    parser.add_argument('--dataset',required=True,choices=['ca','tky','nyc'])
    parser.add_argument('--data',required=True,type=Path,help='Dataset preprocessed directory')
    parser.add_argument('--candidates',required=True,type=Path,help='A4 export directory')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--cap-usd',type=float,default=10.0,help='Total per-run cap, frozen by prepare (includes test)')
    parser.add_argument('--offline',action='store_true',help='Require cached API responses; never call network')
    args = parser.parse_args()
    if not np.isfinite(args.cap_usd) or args.cap_usd<=0: parser.error('cap must be finite and positive')
    args.output.mkdir(parents=True,exist_ok=True)
    lock = args.output/'.active.lock'
    try:
        with lock.open('x',encoding='utf-8') as f:
            import os
            f.write(str(os.getpid()))
    except FileExistsError:
        parser.error('Run directory locked; check process before removing a stale .active.lock')
    try:
        execute(args)
    finally:
        lock.unlink()


if __name__ == '__main__': main()
