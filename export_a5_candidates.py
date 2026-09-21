"""Export a frozen A4 once, without fitting a rule teacher."""
import argparse
from itertools import islice
from pathlib import Path
from a5_jev.data import save,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-f','--yaml-file',required=True)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--max-batches',type=int,help='Smoke export only; cannot feed formal A5')
    args=parser.parse_args()
    if args.max_batches is not None and args.max_batches<=0: parser.error('max-batches must be positive')
    if args.output.exists() and any(args.output.iterdir()): parser.error('Use a fresh output directory; never overwrite frozen candidates')
    from run_rule_teacher import _initialize_inference,_export_candidates
    import torch
    cfg,model,validation,test=_initialize_inference(args.yaml_file)
    m=cfg.model_args
    if not (getattr(m,'use_moe',False) and m.moe_num_experts==12 and m.moe_target_num_groups==6
            and m.moe_top_k==2 and m.moe_rank==32 and m.moe_aux_loss_free
            and m.moe_adaptive_shared_gate and m.moe_adaptive_residual_gate
            and m.moe_adaptive_grouping and m.moe_use_shared_expert
            and m.moe_loss_weight==0 and m.moe_post_group_loss_weight==0
            and not getattr(m,'moe_residual_gate_max_delta',0)
            and not getattr(m,'use_llm_semantic_expert',False)):
        raise ValueError('Expected A4 aux-free dual-gate architecture')
    ckpt=args.checkpoint/'checkpoint.pt' if args.checkpoint.is_dir() else args.checkpoint
    checkpoint_hash=sha256(ckpt)
    # Checkpoint is a locally produced, trusted training artifact (contains optimizer state).
    state=torch.load(ckpt,map_location=model.device,weights_only=False)
    metadata=state.get('experiment_metadata',{})
    if metadata.get('dataset_name') != cfg.dataset_args.dataset_name:
        raise ValueError('Checkpoint lacks matching dataset provenance; use a newly recorded A4 checkpoint')
    smoke_only=bool(args.max_batches or metadata.get('smoke_only') or getattr(cfg.run_args,'smoke_only',False))
    if metadata.get('smoke_only') and not args.max_batches:
        raise ValueError('Smoke checkpoint requires a bounded smoke export')
    for name in [k for k in vars(m) if k.startswith('moe_') or k in ('use_moe','embed_size','st_embed_size')]:
        if metadata.get(name)!=getattr(m,name): raise ValueError(f'Checkpoint/config mismatch: {name}')
    model.load_state_dict(state['model_state_dict'],strict=True)
    args.output.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name,sampler in [('validation',validation),('test',test)]:
        batch=_export_candidates(model,islice(sampler,args.max_batches) if args.max_batches else sampler,20)
        path=args.output/f'{name}_candidates.npz'; batch.save(str(path)); hashes[name]=sha256(path)
    root=Path(__file__).resolve().parent; data=root/'data'/cfg.dataset_args.dataset_name/'preprocessed'
    if sha256(ckpt)!=checkpoint_hash: raise RuntimeError('Checkpoint changed during export')
    save(args.output/'candidate_manifest.json',dict(backbone='A4',dataset=cfg.dataset_args.dataset_name,
        smoke_only=smoke_only,checkpoint=str(ckpt.resolve()),checkpoint_hash=checkpoint_hash,
        config_hash=sha256(root/'conf'/args.yaml_file),candidate_hashes=hashes,sample_hash=sha256(data/'sample.csv'),
        query_hashes={s:sha256(data/n) for s,n in [('validation','validate_sample.csv'),('test','test_sample.csv')]},
        seed=int(cfg.run_args.seed),note='A4 and A5 paired metrics must use these same candidate exports.'))
    print('A4 export complete:',args.output)


if __name__=='__main__': main()
