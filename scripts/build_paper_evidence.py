"""Offline paper tables and controlled post-hoc analyses; no training or API."""
from __future__ import annotations
import copy
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a5_jev.core import GRID, fuse, softmax
from a5_jev.data import Dataset, sha256, save
from rule_teacher.core import DatasetRuleStatistics, DEFAULT_RULE_NAMES, normalize_rows, rerank_candidates
from run_a5_jev import verify_predictions

SNAP = ROOT / 'server_artifacts/paper_snapshot_20260921'
OUT = ROOT / 'docs/paper/generated'
KEYS = ['recall_at_1','recall_at_5','recall_at_10','recall_at_20',
        'ndcg_at_1','ndcg_at_5','ndcg_at_10','ndcg_at_20','mrr']
RUNS = {'nyc':('nyc_a5_jev_force_test','user-authorized audit override'),
        'ca':('ca_a5_jev','original validation selection'),
        'tky':('tky_a5_jev_v1','revised calibration-safe selection')}

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def ranks(ids, data):
    hits=ids == data.batch.labels[:,None]
    return np.where(hits.any(1),hits.argmax(1)+1,data.batch.base_label_ranks).astype(float)

def values(r):
    return np.column_stack([*(r<=k for k in [1,5,10,20]),
        *(np.where(r<=k,1/np.log2(r+1),0) for k in [1,5,10,20]),1/r])

def metrics(ids,data,mask=None):
    v=values(ranks(ids,data))
    if mask is not None: v=v[mask]
    return dict(zip(KEYS,map(float,v.mean(0))))

def paired(ids,data,bootstrap=False,mask=None):
    if mask is None: mask=np.ones(len(ids),bool)
    a,b=ranks(ids,data)[mask],ranks(data.batch.candidate_ids,data)[mask]
    av,bv=values(a),values(b)
    result={'n':int(mask.sum()),'metrics':dict(zip(KEYS,map(float,av.mean(0)))),
        'backbone':dict(zip(KEYS,map(float,bv.mean(0)))),
        'delta':dict(zip(KEYS,map(float,(av-bv).mean(0)))),
        'rescued':int(((a==1)&(b!=1)).sum()),'lost':int(((a!=1)&(b==1)).sum())}
    if bootstrap:
        users,inverse=np.unique(data.queries.UserId.to_numpy()[mask],return_inverse=True)
        counts=np.bincount(inverse)
        totals=np.zeros((len(users),len(KEYS)))
        np.add.at(totals,inverse,av-bv)
        rng=np.random.default_rng(20260921)
        draws=[]
        for _ in range(2000):
            sample=rng.integers(0,len(users),len(users))
            draws.append(totals[sample].sum(0)/counts[sample].sum())
        bounds=np.quantile(draws,[.025,.975],axis=0).T
        result['user_cluster_ci95']=dict(zip(KEYS,bounds.tolist()))
        result['bootstrap']={'seed':20260921,'repeats':2000,'users':len(users)}
    return result

def table(headers,rows):
    def cell(x): return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |',
        '| '+' | '.join(['---']*len(headers))+' |',
        *['| '+' | '.join(map(cell,row))+' |' for row in rows]])+'\n'

def ledger(path):
    if not path.exists(): return {'available':False}
    latest={}; attempts={}
    for line in path.read_text().splitlines():
        e=json.loads(line); latest[e['attempt']]=e
        if e.get('status')=='reserved': attempts.setdefault(e['request_hash'],set()).add(e['attempt'])
    successes=[e for e in latest.values() if e['status']=='success']
    delays=[e['latency_s'] for e in successes if 'latency_s' in e]
    return {'available':True,'attempts':len(latest),'successful_attempts':len(successes),
        'requests_with_retry':sum(len(v)>1 for v in attempts.values()),
        'unresolved_or_failed_attempts':len(latest)-len(successes),
        'input_tokens':sum(e.get('input_tokens',0) for e in successes),
        'estimated_usd':sum(e.get('charge_usd',0) for e in latest.values()),
        'latency_s_p50_p95':np.quantile(delays,[.5,.95]).tolist() if delays else None,
        'note':'Historical client estimate; entire run ledger, not necessarily test-only.'}

def load_split(dataset,split,candidates,output,prepared,manifest):
    d=Dataset(dataset,ROOT/'data'/dataset/'preprocessed',candidates,split)
    assert sha256(d.events_path)==manifest['sample_hash']
    assert sha256(d.query_path)==manifest['query_hashes'][split]
    assert sha256(d.candidate_path)==manifest['candidate_hashes'][split]
    assert d.provenance()==prepared['splits'][split]
    n=len(pd.read_csv(d.query_path))
    idx=d.batch.sample_indices
    assert len(d.queries)==n
    if idx is not None: assert np.array_equal(np.sort(idx),np.arange(n))
    assert not np.any(np.diff(d.batch.model_scores,axis=1)>0)
    pred=verify_predictions(d,output/f'{split}_predictions.json',prepared)
    return d,pred

def alternatives(d,pred,weight,temp,train):
    result={}
    def apply(name,p=pred,w=weight,t=temp,s=.02):
        result[name]=fuse(d.batch.candidate_ids,d.batch.model_scores,d.categories,p,w,t,s)
    apply('without_Jev',w=0)
    uniform=[dict(p,categories=p['categories'],probabilities=[1/len(p['categories'])]*len(p['categories'])) for p in pred]
    apply('uniform_category',p=uniform)
    apply('without_smoothing',s=0)
    apply('temperature_1',t=1)
    apply('category_mass_replacement',w=1)
    for seed in range(5):
        rng=np.random.default_rng(seed)
        shuffled=[dict(p,probabilities=rng.permutation(p['probabilities']).tolist()) for p in pred]
        apply(f'shuffled_category_seed{seed}',p=shuffled)
    result['without_category_mass_correction']=d.batch.candidate_ids.copy()
    for i,p in enumerate(pred):
        q=np.array(p['probabilities'],float)
        if q.sum()==0: continue
        q=.98*q/q.sum()+.02/len(q)
        lookup=dict(zip(p['categories'],q))
        score=d.batch.model_scores[i]+weight/temp*np.log([lookup[c] for c in d.categories[i]])
        result['without_category_mass_correction'][i]=d.batch.candidate_ids[i,np.argsort(-score,kind='stable')]
    global_counts=train.groupby('CategoryText').size().to_dict()
    user_counts=train.groupby(['UserId','CategoryText']).size().to_dict()
    for name,user_specific in [('global_category_prior',False),('user_category_prior',True)]:
        prior=[]
        for user,p in zip(d.queries.UserId,pred):
            q=np.array([(user_counts.get((user,c),0) if user_specific else global_counts.get(c,0))+1.0 for c in p['categories']])
            prior.append(dict(p,probabilities=(q/q.sum()).tolist()))
        apply(name,p=prior)
    return result

def probability_diagnostics(d,pred):
    correct=[]; confidences=[]; nll=[]; brier=[]; outside=[]
    for label,cats,p in zip(d.batch.labels,d.categories,pred):
        truth=d.meta.get(int(label),'unknown')
        target=p['categories'].index(truth) if truth in p['categories'] else len(p['categories'])
        q=np.r_[p['probabilities'],p['outside_probability']].astype(float)
        assert np.isfinite(q).all() and (q>=0).all() and np.isclose(q.sum(),1,atol=1e-4)
        q/=q.sum()
        correct.append(int(q.argmax()==target)); confidences.append(q.max())
        nll.append(-np.log(max(q[target],1e-15)))
        brier.append(float((q*q).sum()-2*q[target]+1))
        outside.append(float(q[-1]))
    conf=np.array(confidences); accuracy=np.array(correct)
    bins=[]; ece=0.
    for j in range(10):
        mask=(conf>=j/10)&((conf<(j+1)/10) if j<9 else (conf<=1))
        if mask.any():
            gap=abs(accuracy[mask].mean()-conf[mask].mean()); ece+=mask.mean()*gap
            bins.append({'lower':j/10,'n':int(mask.sum()),'accuracy':float(accuracy[mask].mean()),'confidence':float(conf[mask].mean())})
    return {'n':len(pred),'category_accuracy':float(accuracy.mean()),'nll':float(np.mean(nll)),
        'brier_multiclass_sum':float(np.mean(brier)),'ece_10_equal_width':float(ece),
        'mean_outside_probability':float(np.mean(outside)),'bins':bins}

def select_rule(d,features):
    mask=d.partitions=='selection'; b=d.batch
    stacked=np.stack([features[n][mask] for n in DEFAULT_RULE_NAMES],axis=-1)
    base=normalize_rows(b.model_scores)[mask]; ids=b.candidate_ids[mask]; labels=b.labels[mask]
    fallback=b.base_label_ranks[mask]
    rng=np.random.default_rng(20260723)
    weights=[np.ones(5)/5,*np.eye(5),*rng.dirichlet(np.ones(5),size=256)]
    best=None
    for wi,w in enumerate(weights):
        rule=(stacked*w).sum(2)
        for strength in np.linspace(0,1,21):
            if strength==0 and wi>0: continue
            order=np.argsort(-((1-strength)*base+strength*rule),axis=1,kind='stable')
            ranked=np.take_along_axis(ids,order,1)
            hits=ranked==labels[:,None]
            r=np.where(hits.any(1),hits.argmax(1)+1,fallback)
            objective=float((r==1).mean()+.25*(1/r).mean())
            key=(objective,-float(strength),-wi)
            if best is None or key>best[0]: best=(key,w.copy(),float(strength))
    return {'weights':best[1].tolist(),'strength':best[2],'selection_objective':best[0][0],
            'weight_candidates':len(weights),'search_seed':20260723,'selection_users_only':True}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results={}; main_rows=[]; ablation_rows=[]; rule_rows=[]; stats_rows=[]; latency_rows=[]
    for ds,(run,policy) in RUNS.items():
        start=time.perf_counter(); print('Processing',ds,flush=True)
        c=SNAP/'artifacts'/f'{ds}_a4_candidates'; o=SNAP/'artifacts'/run
        manifest=read(c/'candidate_manifest.json'); prepared=read(o/'prepared.json'); frozen=read(o/'frozen.json')
        assert sha256(c/'candidate_manifest.json')==prepared['candidate_manifest_hash']
        assert sha256(o/'prepared.json')==frozen['prepared_hash']
        assert sha256(o/'validation_predictions.json')==frozen['validation_predictions_hash']
        w,t=frozen['weight'],frozen['temperature']
        val,vpred=load_split(ds,'validation',c,o,prepared,manifest)
        train=pd.concat(val.history.values(),ignore_index=True)
        events=train.copy(); train=train[train.SplitTag.str.lower()=='train']
        stats=DatasetRuleStatistics.from_sample_frame(ds,events)
        vf=stats.score_candidates(val.queries,val.batch.candidate_ids)
        rule=select_rule(val,vf)
        save(OUT/f'{ds}_rule_selected.json',rule) # freeze before loading test
        test,pred=load_split(ds,'test',c,o,prepared,manifest)
        full=fuse(test.batch.candidate_ids,test.batch.model_scores,test.categories,pred,w,t)
        assert np.array_equal(np.sort(full,axis=1),np.sort(test.batch.candidate_ids,axis=1))
        report=read(o/'test_report.json'); expected=report['selected']['all']['metrics']
        assert sha256(o/'frozen.json')==report['frozen_hash']
        assert sha256(o/'test_predictions.json')==report['predictions_hash']
        primary=paired(full,test,bootstrap=True)
        assert all(abs(primary['metrics'][k]-expected[k])<1e-12 for k in KEYS)
        for model,m in [('A4 paired',primary['backbone']),('A4 + Jev',primary['metrics'])]:
            main_rows.append([ds.upper(),model,policy if model.endswith('Jev') else 'same candidates',len(pred),*[f'{m[k]:.6f}' for k in KEYS]])
        supplement={}
        for split,d,p in [('validation',val,vpred),('test',test,pred)]:
            variants=alternatives(d,p,w,t,train)
            variants['full']=fuse(d.batch.candidate_ids,d.batch.model_scores,d.categories,p,w,t)
            supplement[split]={}
            for name,ids in variants.items():
                result=paired(ids,d); supplement[split][name]=result
                assert result['delta']['recall_at_20']==0
                if split=='test': ablation_rows.append([ds.upper(),name,*[f'{result["metrics"][k]:.6f}' for k in ['recall_at_1','recall_at_5','recall_at_10','ndcg_at_10','mrr']]])
        tf=stats.score_candidates(test.queries,test.batch.candidate_ids)
        rids,_=rerank_candidates(test.batch,tf,DEFAULT_RULE_NAMES,np.array(rule['weights']),rule['strength'])
        teacher=paired(rids,test,bootstrap=True)
        vids,_=rerank_candidates(val.batch,vf,DEFAULT_RULE_NAMES,np.array(rule['weights']),rule['strength'])
        rule['validation_partitions']={p:paired(vids,val,mask=val.partitions==p) for p in ['selection','calibration','audit']}
        for name,result in [('A4',{'metrics':primary['backbone']}),('rule teacher',teacher),('Jev',primary)]:
            rule_rows.append([ds.upper(),name,*[f'{result["metrics"][k]:.6f}' for k in ['recall_at_1','recall_at_5','recall_at_10','ndcg_at_10','mrr']]])
        counts=train.groupby('UserId').size(); edges=np.quantile(counts,[.25,.5,.75])
        freq=test.queries.UserId.map(counts).fillna(0).to_numpy(); bins=np.searchsorted(edges,freq,side='right')
        catn=np.array([len(set(cats)) for cats in test.categories]); buckets={}
        masks={f'train_user_frequency_Q{j+1}':bins==j for j in range(4)}
        masks.update({f'candidate_categories_{lo}_{hi}':(catn>=lo)&(catn<=hi) for lo,hi in [(1,1),(2,4),(5,9),(10,20)]})
        seen=set(zip(train.UserId.astype(int),train.PoiId.astype(int)))
        repeated=np.array([(int(u),int(p)) in seen for u,p in zip(test.queries.UserId,test.batch.labels)])
        masks.update({'seen_user_POI_in_train':repeated,'unseen_user_POI_in_train':~repeated})
        for name,mask in masks.items():
            if mask.any(): buckets[name]=paired(full,test,mask=mask)
        before=ranks(test.batch.candidate_ids,test); after=ranks(full,test); cases={}
        for name,mask in [('rescued',(before!=1)&(after==1)),('lost',(before==1)&(after!=1))]:
            cases[name]=[{'query_index':int(i),'before_rank':int(before[i]),'after_rank':int(after[i]),
                'truth_category_train_mapping':test.meta.get(int(test.batch.labels[i]),'unknown'),
                'base_top1_category':test.categories[i][0]} for i in np.flatnonzero(mask)[:3]]
        cost=ledger(o/'ledger.jsonl')
        latency_rows.append([ds.upper(),run,cost.get('successful_attempts'),cost.get('requests_with_retry'),cost.get('input_tokens'),cost.get('estimated_usd'),cost.get('latency_s_p50_p95')])
        data_stats={'users_events':int(events.UserId.nunique()),'pois_events':int(events.PoiId.nunique()),
            'categories_events':int(events.PoiCategoryId.nunique()),'events':len(events),
            'train_events':len(train),
            'validation_queries':len(val.queries),'test_queries':len(test.queries),'test_users':int(test.queries.UserId.nunique())}
        stats_rows.append([ds.upper(),*data_stats.values()])
        results[ds]={'policy':policy,'frozen':frozen,'candidate_manifest':manifest,'source_report_hash':sha256(o/'test_report.json'),
            'primary':primary,'posthoc_ablation':supplement,'rule_teacher':{'selection':rule,'test':teacher},
            'stratified':buckets,'train_user_frequency_quartiles':edges.tolist(),'cases':cases,
            'probability_diagnostics':probability_diagnostics(test,pred),'ledger':cost,
            'dataset_statistics':data_stats,'offline_analysis_seconds':time.perf_counter()-start}
        save(OUT/f'{ds}_evidence.json',results[ds]); print(ds,'done',round(time.perf_counter()-start,1),'s',flush=True)
    save(OUT/'evidence.json',{'protocol':'SUPPLEMENT_PROTOCOL.md','python':platform.python_version(),'datasets':results})
    headers=['Dataset','Method','Protocol','N',*KEYS]
    (OUT/'main_results.md').write_text('# 配对 test 主结果\n\n单 checkpoint；各数据集选参政策不同。NYC 为 audit override。\n\n'+table(headers,main_rows),encoding='utf-8')
    (OUT/'fusion_ablations.md').write_text('# 固定参数融合消融（事后分析）\n\n'+table(['Dataset','Variant','R1','R5','R10','NDCG10','MRR'],ablation_rows),encoding='utf-8')
    (OUT/'rule_comparison.md').write_text('# 同候选规则教师对照（事后补充）\n\n权重只用 selection 分区，目标 R1+.25MRR；不属于等调参预算对照。\n\n'+table(['Dataset','Method','R1','R5','R10','NDCG10','MRR'],rule_rows),encoding='utf-8')
    (OUT/'dataset_statistics.md').write_text('# 本地核验的数据统计\n\nTrain events 为训练事件数，不是仅末位置评估的查询数；Events 包含全部 SplitTag。\n\n'+table(['Dataset','Users in events','POIs in events','Categories','Events','Train events','Validation queries','Test queries','Test users'],stats_rows),encoding='utf-8')
    (OUT/'api_efficiency.md').write_text('# API 成本与延迟\n\nledger 按 attempt 最终状态计一次；CA 包含 validation+test，TKY/NYC 当前目录主要是 test；不可直接横比总量。\n\n'+table(['Dataset','Run','Successes','Retried requests','Input tokens','Estimated USD','Latency p50/p95 seconds'],latency_rows),encoding='utf-8')
    # Reusable LaTeX fragment, ratios rather than percentages.
    tex=['% Generated; single checkpoint, heterogeneous selection policies.',r'\begin{tabular}{llrrrrrrrrrr}',r'\hline', 'Dataset & Method & N & '+ ' & '.join(k.replace('_',r'\_') for k in KEYS)+r' \\',r'\hline']
    for row in main_rows: tex.append(' & '.join(map(str,[row[0],row[1],row[3],*row[4:]]))+r' \\')
    tex += [r'\hline',r'\end{tabular}']
    (OUT/'main_results.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')

if __name__=='__main__': main()
