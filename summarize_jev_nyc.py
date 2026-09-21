"""Generate the auditable NYC Jev report from cached completed evaluations."""
import json
from pathlib import Path

import numpy as np

from run_jev_nyc import OUT, ROOT, save


def main():
    validation = json.loads((OUT/'validation_report.json').read_text())
    test = json.loads((OUT/'test_report.json').read_text())
    frozen = json.loads((OUT/'frozen_selection.json').read_text())
    entries = [json.loads(line) for line in (OUT/'api_ledger.jsonl').read_text().splitlines()]
    latest = {item['attempt']:item for item in entries}
    successes = [v for v in latest.values() if v['status']=='success']
    latencies = [v['latency_s'] for v in successes]
    usage = dict(attempts=len(latest),successful_requests=len(successes),
        uncertain_or_failed_attempts=len(latest)-len(successes),
        input_tokens=sum(v['input_tokens'] for v in successes),
        successful_estimated_cost_usd=sum(v['charge_usd'] for v in successes),
        conservative_budget_charge_usd=sum(v['charge_usd'] for v in latest.values()),
        latency_p50_s=float(np.median(latencies)),latency_p95_s=float(np.quantile(latencies,.95)))
    report_dir = ROOT/'docs/results/jev_nyc_20260920'
    save(report_dir/'summary.json',dict(validation=validation,test=test,selection=frozen,usage=usage))
    lines = ['# NYC Jev 概率仲裁实验结果', '',
        '实验日期：2026-09-20。固定模型：`jev-1.13.0`。冻结 NYC A4 / 规则教师，使用归档候选在 CPU 上评估。', '',
        '**评估性质：探索性。** NYC validation/test 已用于历史研究；本次用户级 selection/calibration/audit 隔离不抹去历史使用，也不能消除原规则权重在 validation 上选择的影响。', '',
        '## 选择结果', '',
        f'- 选择输入：`{frozen["prompt"]}`；冻结切换阈值：`{frozen["threshold"]}`。',
        f'- 按预注册 calibration/audit 条件允许部署：**{frozen["eligible_for_deployment"]}**。',
        '- 只有 Jev 对 A4 候选给出足够概率时，将它提到规则排序首位；其余样本保留规则排序。neither 回退，不丢弃候选。',
        '- 主结果使用新规则之外的固定仲裁层，没有更新 A4 或 MoE 权重。', '',
        '## 全集指标', '',
        '| 分区 | 方法 | N | Top-1正确数 | R@1 | R@5 | R@20 | MRR | 相对规则净正确数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def add_row(split,name,row,metric_key='metrics',net=None):
        m=row[metric_key]
        net = row.get('paired_vs_rule',{}).get('net_top1','—') if net is None else net
        lines.append(f'| {split} | {name} | {row["n"]} | {round(row["n"]*m["recall_at_1"])} | {m["recall_at_1"]:.6f} | {m["recall_at_5"]:.6f} | {m["recall_at_20"]:.6f} | {m["mrr"]:.6f} | {net} |')
    for split,container in [('validation',validation['selected']['all']),('historical test',test['selected']['all'])]:
        add_row(split,'A4',container,'base',round(container['n']*(container['base']['recall_at_1']-container['rule']['recall_at_1'])))
        add_row(split,'A4 + frozen rules',container,'rule',0)
        if split=='validation':
            for variant,r in validation['raw_fixed_threshold_05'].items():
                add_row(split,f'Jev {variant}, t=0.5',r['all'])
            add_row(split,'Jev selected gate',container)
            add_row(split,'Local supervised logistic gate',validation['local_classifier']['all'])
            add_row(split,'Margin gate',validation['margin_gate']['all'])
        else:
            add_row(split,'Jev selected gate',container)
            add_row(split,'Jev selected prompt, t=0.5',test['raw_fixed_threshold_05']['all'])
            add_row(split,'Local supervised logistic gate',test['local_classifier']['all'])
            add_row(split,'Margin gate',test['margin_gate']['all'])
    lines += ['', '旧 Qwen binary-v1 在同一 validation 缓存上的历史结果：376/1400（R@1=0.268571）；本次输入更丰富且有 neither 选项，缺少逐条 Qwen 响应，因此不是仅替换模型的严格消融。', '',
        '## 冻结 Jev 门控的分区审计', '',
        '| 分区 | N | 冲突数 | 切换数 | 挽救 | 损失 | 净变化 | R@1 差值95%用户聚类bootstrap区间 | MRR差值 |',
        '|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for part,row in validation['selected'].items():
        p=row['paired_vs_rule']
        lines.append(f'| {part} | {row["n"]} | {row["n_conflicts"]} | {row["switches"]} | {p["rescued"]} | {p["lost"]} | {p["net_top1"]} | {p["user_cluster_ci_r1"]} | {p["delta_mrr"]:.6f} |')
    p=test['selected']['all']['paired_vs_rule']
    lines += ['',f'历史 test 冻结门控：挽救 {p["rescued"]}，损失 {p["lost"]}，净变化 {p["net_top1"]}；R@1 差值区间 {p["user_cluster_ci_r1"]}，MRR 差值区间 {p["user_cluster_ci_mrr"]}。',
        '区间不校正既往模型选择或本轮多重比较，不能当作确认性显著性证据。', '',
        '## 概率可靠性与选项顺序', '',
        '三类概率的真实结局定义为：A4候选正确 / 规则候选正确 / 两者都错。下表覆盖所有冲突，而非只挑至少一个正确的样本。', '',
        '| validation 输入 | 三类准确率 | Brier ↓ | NLL ↓ | ECE ↓ | 32条交换顺序argmax翻转率 | 平均概率绝对变化 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for variant,report in validation['raw_fixed_threshold_05'].items():
        rel=report['all']['probability_reliability']; perm=validation['permutation'][variant]
        lines.append(f'| {variant} | {rel["accuracy"]:.6f} | {rel["brier"]:.6f} | {rel["nll"]:.6f} | {rel["ece"]:.6f} | {perm["argmax_flip_rate"]:.6f} | {perm["mean_absolute_probability_change"]:.6f} |')
    lines += ['', 'ECE使用预先固定分箱；样本有限，概率评估仅说明此具体推荐任务的表现。接口返回的confidence不直接用作真实准确率。', '',
        '## 判断与下一步优化建议', '',
        '本轮不足以支持将 Jev 仲裁加入最终模型：validation 全集净增1条，audit净减1条；历史test净增2条且区间跨零，本地监督分类器得到相同Top-1和MRR。小幅test正向不能推翻事先冻结的audit拒绝条件。', '',
        '在407条validation冲突中，真实结局为A4正确62、规则正确75、两者都错270。context版只将32条判为neither，其中30条正确；平均neither概率仅0.194，实际占比0.663。输入给出了相对偏好线索，却不足以预测匿名地点的真实下一访问，原始概率不能直接作为此任务的正确率。', '',
        '优先级建议（未执行的新实验）：', '',
        '1. 利用已缓存的Jev分布，比较同一监督预算下的“本地特征门”与“本地特征+Jev分布门”。使用训练侧留出或交叉拟合；必须证明Jev相对于本地门的增量，不只比较规则。',
        '2. 把目标改为规则干预的相对收益：估计保留规则与切换A4的期望差；两者都错的样本可为无Top-1收益，不把所有冲突强制二选一。用独立校准数据学习分布偏差和回退策略。',
        '3. 将Jev拆成时间-类别匹配、近期行为连续性等窄问题，输出作为特征，交给本地门控学习权重；窄问题仍需实测，不能预设语义分数有效。',
        '4. 当前二候选仲裁不能挽救两个候选都错误的270条。可在新的开发协议下考虑Top-K候选或基于类别概率的残差，但要与纯本地reranker比较，并保留候选覆盖上限。',
        '5. 若希望依赖外部知识，考虑补充有依据的地点名称/功能描述；目前类别、距离和匿名访问统计的信息边界仍在。',
        '6. 下一轮效果选择不能沿用刚观察的historical test；需要训练侧开发、锁定新的时间留出，或在尚未完成的CA/TKY最终A4上预先登记验证。', '',
        '## 成本与复现', '',
        f'- 成功请求：{usage["successful_requests"]}；失败/结果不明尝试：{usage["uncertain_or_failed_attempts"]}；输入 token：{usage["input_tokens"]}。',
        f'- 按公布单价估算费用：${usage["successful_estimated_cost_usd"]:.6f}；含不明尝试的保守预算记账：${usage["conservative_budget_charge_usd"]:.6f}；运行上限 $1。',
        f'- 并发上限2；本机端到端 p50={usage["latency_p50_s"]:.3f}s，p95={usage["latency_p95_s"]:.3f}s。',
        '- 结果来自实际 API 调用，未用手工构造概率代替。API key仅通过进程环境提供，未写入项目文件。',
        '- 代码：`run_jev_nyc.py`；检查：`tests/test_jev_nyc.py`；协议：`docs/15_jev_nyc_protocol.md`。',
        '- 完整缓存、请求散列、费用账本、冻结参数、逐样本概率：`server_artifacts/jev_nyc_20260920/`。',
        '- 可提交汇总：`docs/results/jev_nyc_20260920/summary.json`。',
        '- 本地分类器使用selection真实标签，Jev没有接收这些标签；两者监督资源不同。',
        '- 训练统计、候选与历史时间使用沿用原实验定义，不对旧协议作重新解释。',
        '- test结果不用于修改阈值、提示词、选择其他快照，后续新方案需要新的开发和验收协议。', '']
    path = ROOT/'docs/16_jev_nyc_results.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(report=str(path),usage=usage),indent=2))


if __name__ == '__main__':
    main()
