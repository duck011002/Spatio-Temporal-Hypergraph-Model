"""Readable supplemental tables from frozen offline evidence; no API."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'docs/paper/generated'
def table(h,rows): return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])+'\n'
def main():
    evidence=json.loads((OUT/'evidence.json').read_text(encoding='utf-8'))['datasets']
    strata=[]; intervals=[]; probs=[]; cases=[]; gates=[]
    for ds,e in evidence.items():
        for name,v in e['stratified'].items():
            strata.append([ds,name,v['n'],*[f'{v["delta"][k]*100:.3f}' for k in ['recall_at_1','recall_at_5','recall_at_10','ndcg_at_10','mrr']]])
        for k,ci in e['primary']['user_cluster_ci95'].items():
            intervals.append([ds,k,f'{100*e["primary"]["delta"][k]:.4f}',f'{100*ci[0]:.4f}',f'{100*ci[1]:.4f}'])
        p=e['probability_diagnostics']; probs.append([ds,p['n'],*[f'{p[k]:.6f}' for k in ['category_accuracy','nll','brier_multiclass_sum','ece_10_equal_width','mean_outside_probability']]])
        for kind,values in e['cases'].items():
            for v in values: cases.append([ds,kind,v['query_index'],v['before_rank'],v['after_rank'],v['truth_category_train_mapping'],v['base_top1_category']])
        for split,v in e['rule_teacher']['selection']['validation_partitions'].items():
            gates.append([ds,split,v['n'],v['rescued'],v['lost'],f'{v["delta"]["recall_at_1"]:.6f}',f'{v["delta"]["mrr"]:.6f}'])
    outputs={
        'stratified':('分组配对增益（pp）','稀疏组不是零历史冷启动；桶边界来自训练频次，全部非空桶都报告。',['Dataset','Group','N','ΔR1','ΔR5','ΔR10','ΔN10','ΔMRR'],strata),
        'confidence_intervals':('用户聚类bootstrap','2000次，seed20260921；百分点；描述性未校正区间，不替代训练seed方差。',['Dataset','Metric','Δ pp','CI low','CI high'],intervals),
        'probability_diagnostics':('类别概率诊断','候选类别加outside作为分类集合；类别数可变，不宜直接跨数据集比难度。',['Dataset','N','Accuracy','NLL','Brier','ECE10','Mean outside'],probs),
        'cases':('系统选取的成功/失败例子','各取查询原顺序最早三个；仅匿名索引/类别，不发布轨迹。',['Dataset','Type','Query index','Old rank','New rank','Truth category','Base top1 category'],cases),
        'rule_validation_partitions':('规则教师的验证分区','selection选择后原样报告calibration/audit；不是按test选参。',['Dataset','Partition','N','Rescued','Lost','ΔR1','ΔMRR'],gates)}
    for name,(title,note,headers,rows) in outputs.items():
        (OUT/(name+'.md')).write_text('# '+title+'\n\n'+note+'\n\n'+table(headers,rows),encoding='utf-8')
    print('Rendered',len(outputs),'supplemental tables.')
if __name__=='__main__': main()
