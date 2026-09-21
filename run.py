import logging
import os
import os.path as osp
import datetime
import time
import torch
import random
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
from preprocess import preprocess
from utils import seed_torch, set_logger, Cfg, count_parameters, test_step, save_model
from layer import NeighborSampler
from dataset import LBSNDataset
from model import STHGCN, SequentialTransformer
from utils.run_reporter import RunReporter, configure_console_logging


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--yaml_file', help='The configuration file.', required=True)
    parser.add_argument('--multi_run_mode', help='Run multiple experiments with the same config.', action='store_true')
    parser.add_argument(
        '--backbone-checkpoint',
        default=None,
        help=(
            'A frozen backbone checkpoint file or directory. Unlike '
            'init_checkpoint, optimizer and training-step state are not resumed.'
        ),
    )
    parser.add_argument(
        '--semantic-init-snapshot',
        default=None,
        help='Optional lightweight LLM semantic snapshot to continue training.',
    )
    parser.add_argument(
        '--learning-rate-override',
        type=float,
        default=None,
    )
    parser.add_argument(
        '--max-steps-override',
        type=int,
        default=None,
    )
    args = parser.parse_args()
    conf_file = args.yaml_file

    cfg = Cfg(conf_file)
    eval_max_batches = getattr(cfg.run_args, 'eval_max_batches', None)
    if eval_max_batches is not None and not getattr(cfg.run_args, 'smoke_only', False):
        raise ValueError('Partial evaluation requires smoke_only: true; cannot be used for formal training.')
    if args.learning_rate_override is not None:
        cfg.run_args.learning_rate = args.learning_rate_override
    if args.max_steps_override is not None:
        cfg.run_args.max_steps = args.max_steps_override

    sizes = [int(i) for i in cfg.model_args.sizes.split('-')]
    cfg.model_args.sizes = sizes

    # cuda setting
    if int(cfg.run_args.gpu) >= 0:
        device = 'cuda:' + str(cfg.run_args.gpu)
    else:
        device = 'cpu'
    cfg.run_args.device = device

    # for multiple runs, seed is replaced with random value
    if args.multi_run_mode:
        cfg.run_args.seed = None
    if cfg.run_args.seed is None:
        seed = random.randint(0, 100000000)
    else:
        seed = int(cfg.run_args.seed)

    seed_torch(seed)

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg.run_args.save_path = f'tensorboard/{current_time}/{cfg.dataset_args.dataset_name}'
    cfg.run_args.log_path = f'log/{current_time}/{cfg.dataset_args.dataset_name}'

    if not osp.isdir(cfg.run_args.save_path):
        os.makedirs(cfg.run_args.save_path)
    if not osp.isdir(cfg.run_args.log_path):
        os.makedirs(cfg.run_args.log_path)

    set_logger(cfg.run_args)
    summary_writer = SummaryWriter(log_dir=cfg.run_args.save_path)

    hparam_dict = {}
    for group, hparam in cfg.__dict__.items():
        hparam_dict.update(hparam.__dict__)
    hparam_dict['seed'] = seed
    hparam_dict['sizes'] = '-'.join([str(item) for item in cfg.model_args.sizes])

    run_reporter = RunReporter.create(
        root='runs',
        dataset=cfg.dataset_args.dataset_name,
        model=cfg.model_args.model_name,
        seed=seed,
        hparams=hparam_dict,
        log_path=cfg.run_args.log_path,
        tensorboard_path=cfg.run_args.save_path
    )
    configure_console_logging()
    logging.info(f'[Run] Artifacts: {run_reporter.run_dir}')

    # Preprocess data
    setup_progress = tqdm(
        total=3,
        desc='Setup',
        unit='stage',
        dynamic_ncols=True
    )
    setup_progress.set_postfix_str('preprocess')
    stage_started_at = time.perf_counter()
    run_reporter.stage_started('preprocess')
    preprocess(cfg)
    run_reporter.stage_finished('preprocess', time.perf_counter() - stage_started_at)
    setup_progress.update(1)

    # Initialize dataset
    setup_progress.set_postfix_str('load dataset')
    stage_started_at = time.perf_counter()
    run_reporter.stage_started('dataset')
    lbsn_dataset = LBSNDataset(cfg)
    cfg.dataset_args.spatial_slots = lbsn_dataset.spatial_slots
    cfg.dataset_args.num_user = lbsn_dataset.num_user
    cfg.dataset_args.num_poi = lbsn_dataset.num_poi
    cfg.dataset_args.num_category = lbsn_dataset.num_category
    cfg.dataset_args.padding_poi_id = lbsn_dataset.padding_poi_id
    cfg.dataset_args.padding_user_id = lbsn_dataset.padding_user_id
    cfg.dataset_args.padding_poi_category = lbsn_dataset.padding_poi_category
    cfg.dataset_args.padding_hour_id = lbsn_dataset.padding_hour_id
    cfg.dataset_args.padding_weekday_id = lbsn_dataset.padding_weekday_id
    run_reporter.stage_finished('dataset', time.perf_counter() - stage_started_at)
    setup_progress.update(1)

    # Initialize neighbor sampler(dataloader)
    setup_progress.set_postfix_str('build samplers')
    stage_started_at = time.perf_counter()
    run_reporter.stage_started('samplers')
    sampler_train, sampler_validate, sampler_test = None, None, None

    if cfg.run_args.do_train:
        sampler_train = NeighborSampler(
            lbsn_dataset.x,
            lbsn_dataset.edge_index,
            lbsn_dataset.edge_attr,
            intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
            inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
            edge_t=lbsn_dataset.edge_t,
            edge_delta_t=lbsn_dataset.edge_delta_t,
            edge_type=lbsn_dataset.edge_type,
            sizes=sizes,
            sample_idx=lbsn_dataset.sample_idx_train,
            node_idx=lbsn_dataset.node_idx_train,
            edge_delta_s=lbsn_dataset.edge_delta_s,
            max_time=lbsn_dataset.max_time_train,
            label=lbsn_dataset.label_train,
            batch_size=cfg.run_args.batch_size,
            num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
            shuffle=True,
            pin_memory=True
        )

    if cfg.run_args.do_validate:
        sampler_validate = NeighborSampler(
            lbsn_dataset.x,
            lbsn_dataset.edge_index,
            lbsn_dataset.edge_attr,
            intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
            inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
            edge_t=lbsn_dataset.edge_t,
            edge_delta_t=lbsn_dataset.edge_delta_t,
            edge_type=lbsn_dataset.edge_type,
            sizes=sizes,
            sample_idx=lbsn_dataset.sample_idx_valid,
            node_idx=lbsn_dataset.node_idx_valid,
            edge_delta_s=lbsn_dataset.edge_delta_s,
            max_time=lbsn_dataset.max_time_valid,
            label=lbsn_dataset.label_valid,
            batch_size=cfg.run_args.eval_batch_size,
            num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
            shuffle=False,
            pin_memory=True
        )

    if cfg.run_args.do_test:
        sampler_test = NeighborSampler(
            lbsn_dataset.x,
            lbsn_dataset.edge_index,
            lbsn_dataset.edge_attr,
            intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
            inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
            edge_t=lbsn_dataset.edge_t,
            edge_delta_t=lbsn_dataset.edge_delta_t,
            edge_type=lbsn_dataset.edge_type,
            sizes=sizes,
            sample_idx=lbsn_dataset.sample_idx_test,
            node_idx=lbsn_dataset.node_idx_test,
            edge_delta_s=lbsn_dataset.edge_delta_s,
            max_time=lbsn_dataset.max_time_test,
            label=lbsn_dataset.label_test,
            batch_size=cfg.run_args.eval_batch_size,
            num_workers=0 if device == 'cpu' else cfg.run_args.num_workers,
            shuffle=False,
            pin_memory=True
        )

    run_reporter.stage_finished('samplers', time.perf_counter() - stage_started_at)
    setup_progress.update(1)
    setup_progress.close()

    if cfg.model_args.model_name == 'sthgcn':
        model = STHGCN(cfg)
    elif cfg.model_args.model_name == 'seq_transformer':
        model = SequentialTransformer(cfg)
    else:
        raise NotImplementedError(
            f'[Training] Model {cfg.model_args.name}, please choose from ["sthgcn", "seq_transformer"]'
        )

    model = model.to(device)
    backbone_checkpoint = args.backbone_checkpoint or getattr(
        cfg.run_args,
        'backbone_checkpoint',
        None,
    )
    if backbone_checkpoint and cfg.run_args.init_checkpoint:
        raise ValueError(
            'backbone_checkpoint and init_checkpoint are mutually exclusive.'
        )
    if (
        cfg.run_args.do_train
        and getattr(model, 'use_llm_semantic_expert', False)
        and getattr(model, 'llm_semantic_freeze_backbone', False)
        and not backbone_checkpoint
        and not cfg.run_args.init_checkpoint
    ):
        raise ValueError(
            'Frozen LLM semantic training requires --backbone-checkpoint '
            '(or a full semantic init_checkpoint).'
        )
    if backbone_checkpoint:
        checkpoint_path = str(backbone_checkpoint)
        if osp.isdir(checkpoint_path):
            checkpoint_path = osp.join(checkpoint_path, 'checkpoint.pt')
        logging.info('[LLM Semantic] Loading frozen backbone: %s', checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_backbone_state_dict(state_dict)
    if args.semantic_init_snapshot:
        if not getattr(model, 'use_llm_semantic_expert', False):
            raise ValueError(
                '--semantic-init-snapshot requires the semantic expert.'
            )
        semantic_snapshot = torch.load(
            args.semantic_init_snapshot,
            map_location=device,
        )
        model.llm_semantic_expert.load_state_dict(
            semantic_snapshot['semantic_state_dict']
        )
        logging.info(
            '[LLM Semantic] Loaded continuation snapshot at semantic step %s: %s',
            semantic_snapshot.get('step'),
            args.semantic_init_snapshot,
        )
    if (
        getattr(model, 'use_llm_semantic_expert', False)
        and getattr(model, 'llm_semantic_freeze_backbone', False)
    ):
        model.freeze_backbone_for_llm_semantic()
        logging.info('[LLM Semantic] Backbone frozen; training semantic expert only.')
    logging.info(f'[Training] Seed: {seed}')
    logging.info('[Training] Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info(f'[Training] Parameter {name}: {param.size()}, require_grad = {param.requires_grad}')
    logging.info(f'[Training] #Parameters: {count_parameters(model)}')
    if getattr(model, 'use_moe', False):
        logging.info(
            '[MoE] Enabled: context=%s experts=%s top_k=%s rank=%s '
            'balance_weight=%s post_group_balance_weight=%s '
            'adaptive_grouping=%s group_warmup_steps=%s target_groups=%s '
            'aux_loss_free=%s router_bias_update_rate=%s '
            'adaptive_shared_gate=%s adaptive_residual_gate=%s '
            'residual_gate_max_delta=%s',
            model.moe.router_context,
            model.moe.num_experts,
            model.moe.top_k,
            cfg.model_args.moe_rank,
            model.moe_loss_weight,
            model.moe_post_group_loss_weight,
            model.moe.adaptive_grouping,
            model.moe.group_warmup_steps,
            model.moe.target_num_groups,
            model.moe.aux_loss_free,
            model.moe.router_bias_update_rate,
            model.moe.adaptive_shared_gate,
            model.moe.adaptive_residual_gate,
            model.moe.residual_gate_max_delta,
        )
    if getattr(model, 'use_llm_semantic_expert', False):
        semantic_expert = model.llm_semantic_expert
        logging.info(
            '[LLM Semantic] Enabled: mode=%s profile_dim=%s rank=%s '
            'max_scale=%s distill_weight=%s warmup_steps=%s '
            'profile_coverage=%.6f',
            semantic_expert.mode,
            semantic_expert.profile_dim,
            semantic_expert.rank,
            semantic_expert.max_scale,
            model.llm_semantic_distill_weight,
            semantic_expert.distill_warmup_steps,
            float(semantic_expert.profile_available.float().mean().item()),
        )

    if cfg.run_args.do_train:
        current_learning_rate = cfg.run_args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )
        if cfg.run_args.warm_up_steps:
            warm_up_steps = cfg.run_args.warm_up_steps
        else:
            warm_up_steps = cfg.run_args.max_steps // 2

        init_step = 0
        if cfg.run_args.init_checkpoint:
            # Restore model from checkpoint directory
            # manually set in yml
            logging.info(f'[Training] Loading checkpoint %s...' % cfg.run_args.init_checkpoint)
            checkpoint = torch.load(
                osp.join(cfg.run_args.init_checkpoint, 'checkpoint.pt'),
                map_location=device,
            )
            init_step = checkpoint['step']
            model.load_state_dict(checkpoint['model_state_dict'])
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            cooldown_rate = checkpoint['cooldown_rate']
            sizes = checkpoint['sizes']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            if backbone_checkpoint:
                logging.info('[Training] Initializing semantic expert on frozen backbone.')
            else:
                logging.info(f'[Training] Randomly Initializing Model...')
            init_step = 0
        step = init_step

        # Set valid dataloader as it would be evaluated during training
        logging.info(f'[Training] Initial learning rate: {current_learning_rate}')

        # Training Loop
        best_metrics = float('-inf')
        early_stop_patience = int(getattr(cfg.run_args, 'early_stop_patience', 0) or 0)
        epochs_without_improvement = 0
        global_step = 0
        training_started_at = time.perf_counter()
        run_reporter.stage_started('training', planned_epochs=cfg.run_args.epoch)
        for eph in range(cfg.run_args.epoch):
            training_logs = []
            running_loss_total = 0.0
            last_validation = None
            epoch_improved = False
            epoch_started_at = time.perf_counter()
            if global_step >= cfg.run_args.max_steps:
                break
            progress = tqdm(
                sampler_train,
                total=len(sampler_train),
                desc=f'Train {eph + 1}/{cfg.run_args.epoch}',
                unit='batch',
                dynamic_ncols=True,
                leave=True
            )
            for data in progress:
                model.train()
                split_index = torch.max(data.adjs_t[1].storage.row()).tolist()
                data = data.to(device)
                input_data = {
                    'x': data.x,
                    'x_target': data.x_target,
                    'edge_index': data.adjs_t,
                    'edge_attr': data.edge_attrs,
                    'split_index': split_index,
                    'delta_ts': data.edge_delta_ts,
                    'delta_ss': data.edge_delta_ss,
                    'edge_type': data.edge_types
                }

                out, loss = model(input_data, label=data.y[:, 0])
                loss_value = float(loss.detach().cpu().item())
                if not torch.isfinite(loss).all():
                    raise FloatingPointError('Training loss is not finite')
                training_logs.append(loss_value)
                running_loss_total += loss_value
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                semantic_snapshot_steps = int(
                    getattr(
                        cfg.run_args,
                        'llm_semantic_snapshot_steps',
                        0,
                    )
                    or 0
                )
                if (
                    semantic_snapshot_steps > 0
                    and getattr(model, 'use_llm_semantic_expert', False)
                    and global_step % semantic_snapshot_steps == 0
                ):
                    semantic_snapshot_path = osp.join(
                        cfg.run_args.save_path,
                        f'llm_semantic_step_{global_step}.pt',
                    )
                    torch.save(
                        {
                            'step': global_step,
                            'semantic_state_dict': (
                                model.llm_semantic_expert.state_dict()
                            ),
                            'backbone_checkpoint': backbone_checkpoint,
                        },
                        semantic_snapshot_path,
                    )
                    logging.info(
                        '[LLM Semantic] Saved lightweight snapshot: %s',
                        semantic_snapshot_path,
                    )
                summary_writer.add_scalar(f'train/loss_step', loss_value, global_step)
                if getattr(model, 'last_moe_aux_loss', None) is not None:
                    summary_writer.add_scalar(
                        'train/moe_balance_loss_step',
                        float(model.last_moe_aux_loss.cpu().item()),
                        global_step
                    )
                    summary_writer.add_scalar(
                        'train/moe_balance_weight_step',
                        float(model.last_moe_loss_weight),
                        global_step
                    )
                semantic_diagnostics = getattr(
                    model,
                    'get_llm_semantic_diagnostics',
                    lambda: None,
                )()
                if semantic_diagnostics is not None:
                    for diagnostic_name, diagnostic_value in semantic_diagnostics.items():
                        summary_writer.add_scalar(
                            f'train/llm_semantic_{diagnostic_name}_step',
                            float(diagnostic_value.cpu().item()),
                            global_step,
                        )
                if cfg.run_args.do_validate and global_step % cfg.run_args.valid_steps == 0:
                    logging.info(f'[Evaluating] Evaluating on Valid Dataset...')

                    logging.info(f'[Evaluating] Epoch {eph}, step {global_step}:')
                    recall_res, ndcg_res, map_res, mrr_res, eval_loss = test_step(
                        model,
                        data=sampler_validate,
                        desc=f'Validate {eph + 1}/{cfg.run_args.epoch}',
                        max_batches=eval_max_batches,
                    )
                    summary_writer.add_scalar(f'validate/Recall@1', 100*recall_res[1], global_step)
                    summary_writer.add_scalar(f'validate/Recall@5', 100*recall_res[5], global_step)
                    summary_writer.add_scalar(f'validate/Recall@10', 100*recall_res[10], global_step)
                    summary_writer.add_scalar(f'validate/Recall@20', 100*recall_res[20], global_step)
                    summary_writer.add_scalar(f'validate/MRR', mrr_res, global_step)
                    summary_writer.add_scalar(f'validate/eval_loss', eval_loss, global_step)
                    summary_writer.add_scalar('train/learning_rate', current_learning_rate, global_step)

                    last_validation = {
                        'global_step': global_step,
                        'loss': float(eval_loss),
                        'recall_at_1': float(recall_res[1]),
                        'recall_at_5': float(recall_res[5]),
                        'recall_at_10': float(recall_res[10]),
                        'recall_at_20': float(recall_res[20]),
                        'mrr': float(mrr_res),
                        'ndcg_at_1': float(ndcg_res[1]),
                        'ndcg_at_5': float(ndcg_res[5]),
                        'ndcg_at_10': float(ndcg_res[10]),
                        'ndcg_at_20': float(ndcg_res[20]),
                        'map_at_1': float(map_res[1]),
                        'map_at_5': float(map_res[5]),
                        'map_at_10': float(map_res[10]),
                        'map_at_20': float(map_res[20])
                    }
                    run_reporter.validation(
                        epoch=eph + 1,
                        global_step=global_step,
                        metrics=last_validation
                    )

                    metrics = 4 * recall_res[1] + recall_res[20]

                    # save model based on compositional recall metrics
                    if metrics > best_metrics:
                        save_variable_list = {
                            'step': global_step,
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps,
                            'cooldown_rate': cfg.run_args.cooldown_rate,
                            'sizes': sizes
                        }
                        logging.info(f'[Training] Save model at step {global_step} epoch {eph}')
                        save_model(model, optimizer, save_variable_list, cfg.run_args, hparam_dict)
                        best_metrics = metrics
                        epoch_improved = True

                # learning rate schedule
                if global_step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    logging.info(f'[Training] Change learning_rate to {current_learning_rate} at step {global_step}')
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=current_learning_rate
                    )
                    warm_up_steps = warm_up_steps * cfg.run_args.cooldown_rate

                if global_step >= cfg.run_args.max_steps:
                    break
                global_step += 1
                progress.set_postfix(
                    loss=f'{running_loss_total / len(training_logs):.4f}',
                    lr=f'{current_learning_rate:.2e}'
                )

            epoch_loss = running_loss_total / len(training_logs)
            logging.info(f'[Training] Average train loss at step {global_step} is {epoch_loss}:')
            summary_writer.add_scalar('train/loss_epoch', epoch_loss, eph)
            run_reporter.epoch(
                epoch=eph + 1,
                global_step=global_step,
                train_loss=epoch_loss,
                learning_rate=current_learning_rate,
                duration_s=time.perf_counter() - epoch_started_at,
                validation=last_validation
            )
            if early_stop_patience > 0:
                if epoch_improved:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                logging.info(
                    '[Training] Early stopping: %s/%s epochs without improvement',
                    epochs_without_improvement,
                    early_stop_patience
                )
                if epochs_without_improvement >= early_stop_patience:
                    logging.info(
                        '[Training] Early stopping triggered after epoch %s',
                        eph + 1
                    )
                    break
        run_reporter.stage_finished('training', time.perf_counter() - training_started_at)

    if cfg.run_args.do_test:
        logging.info('[Evaluating] Start evaluating on test set...')

        checkpoint = torch.load(
            osp.join(cfg.run_args.save_path, 'checkpoint.pt'),
            map_location=device,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        test_started_at = time.perf_counter()
        run_reporter.stage_started('test')
        recall_res, ndcg_res, map_res, mrr_res, loss = test_step(
            model,
            sampler_test,
            desc='Test',
            max_batches=eval_max_batches,
        )
        num_params = count_parameters(model)
        metric_dict = {
            'hparam/num_params': num_params,
            'hparam/Recall@1': recall_res[1],
            'hparam/Recall@5': recall_res[5],
            'hparam/Recall@10': recall_res[10],
            'hparam/Recall@20': recall_res[20],
            'hparam/NDCG@1': ndcg_res[1],
            'hparam/NDCG@5': ndcg_res[5],
            'hparam/NDCG@10': ndcg_res[10],
            'hparam/NDCG@20': ndcg_res[20],
            'hparam/MAP@1': map_res[1],
            'hparam/MAP@5': map_res[5],
            'hparam/MAP@10': map_res[10],
            'hparam/MAP@20': map_res[20],
            'hparam/MRR': mrr_res,
        }
        logging.info(f'[Evaluating] Test evaluation result : {metric_dict}')
        summary_writer.add_hparams(hparam_dict, metric_dict)
        run_reporter.stage_finished(
            'test',
            time.perf_counter() - test_started_at,
            metrics=metric_dict
        )
        run_reporter.finish(test_metrics=metric_dict)
    else:
        run_reporter.finish()

    summary_writer.close()
