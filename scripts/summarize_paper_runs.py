"""Build provenance, training configuration and historical run tables, offline."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SNAP=ROOT/'server_artifacts/paper_snapshot_20260921'
OUT=ROOT/'docs/paper/generated'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def table(h,rows): return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])+'\n'
def main():
    manifest=read(SNAP/'snapshot_manifest.json')
    for f in manifest:
        p=SNAP/f['path']
        assert p.stat().st_size==f['bytes'],p
        assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],p
    selected={'NYC':'20260723_155322_nyc_sthgcn_seed80786525','CA':'20260921_001455_ca_sthgcn_seed27486607','TKY':'20260921_023057_tky_sthgcn_seed54607333'}
    configs={}; rows=[]
    for ds,run in selected.items():
        s=read(SNAP/'runs'/run/'summary.json'); c=read(SNAP/'runs'/run/'config.json'); configs[ds]=c
        seconds=(datetime.fromisoformat(s['finished_at'])-datetime.fromisoformat(s['started_at'])).total_seconds()
        rows.append([ds,run,int(seconds),s['test_metrics']['hparam/num_params'],c['seed'],c['sizes'],c['warm_up_steps'],c['valid_steps'],c['cooldown_rate']])
    (OUT/'training_runs.md').write_text('# 已完成 A4 训练记录\n\n计时为 summary 的开始至结束，含运行内验证/测试；不是纯 GPU kernel 时间。\n\n'+table(['Dataset','Run','Seconds','Parameters','Seed','Neighbors','LR warmup','Validate interval','Cooldown'],rows),encoding='utf-8')
    labels={'005631':'R1','014239':'R1.1','020254':'R1.2','130050':'A1','134839':'A2','143805':'A3','155322':'A4','165638':'B-Tanh (old A5)'}
    rows=[]
    for p in sorted((SNAP/'runs').glob('20260723_*/summary.json')):
        stamp=p.parent.name.split('_')[1]
        if stamp not in labels: continue
        s=read(p); m=s['test_metrics']; keys=['Recall@1','Recall@5','Recall@10','Recall@20','NDCG@5','NDCG@10','NDCG@20','MRR']
        rows.append([labels[stamp],p.parent.name,*[f'{m["hparam/"+k]:.6f}' for k in keys]])
    (OUT/'historical_nyc.md').write_text('# NYC 历史模型演进（非严格单因素消融）\n\n来自训练 summary；不可与另一次采样导出的 A5 配对表混算差值。\n\n'+table(['Variant','Run','R1','R5','R10','R20','N5','N10','N20','MRR'],rows),encoding='utf-8')
    code=[]
    for f in manifest:
        if f['path'].endswith('.py'):
            local=ROOT/f['path']; match=local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest()==f['sha256']
            code.append({**f,'matches_current_local':match})
    (OUT/'provenance.json').write_text(json.dumps({'verified_snapshot_files':len(manifest),'training_configs':configs,'source_code':code},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Verified',len(manifest),'snapshot files; generated runtime and history tables.')
if __name__=='__main__': main()
