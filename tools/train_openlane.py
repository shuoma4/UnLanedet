#!/usr/bin/env python
"""
Training Script for OpenLane Dataset with Temporal Modeling.

This script provides flexible training options including:
- GPU selection
- Temporal module configuration
- Multi-GPU training
- Resume from checkpoint
- Evaluation during training

Usage:
    # Single GPU training (default)
    python tools/train_openlane.py --config config/clrnet/resnet34_openlane.py

    # Specify GPU
    python tools/train_openlane.py --config config/clrnet/resnet34_openlane.py --gpu 0

    # Multi-GPU training
    python tools/train_openlane.py --config config/clrnet/resnet34_openlane.py --gpus 0,1,2,3

    # With temporal module
    python tools/train_openlane.py --config config/clrnet/resnet34_openlane.py --temporal convlstm

    # Resume training
    python tools/train_openlane.py --config config/clrnet/resnet34_openlane.py --resume output/checkpoint.pth

    # Evaluation only
    python tools/train_openlane.py --config config/clrnet/resnet34_openlane.py --eval-only --weights output/best_model.pth
"""

import argparse
import logging
import os
import sys
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from unlanedet.config import LazyConfig, instantiate
from unlanedet.engine import launch
from unlanedet.checkpoint import Checkpointer, BestCheckPointer
from unlanedet.evaluation import inference_on_dataset, print_csv_format
from unlanedet.utils import comm

logger = logging.getLogger("unlanedet")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train OpenLane lane detection model")

    # Config
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )

    # GPU selection
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use (single GPU mode, default: 0)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for multi-GPU training (e.g., 0,1,2,3)"
    )

    # Temporal module
    parser.add_argument(
        "--temporal",
        type=str,
        default=None,
        choices=['convlstm', 'attention', '3dconv', 'gru'],
        help="Type of temporal module to use"
    )
    parser.add_argument(
        "--temporal-window",
        type=int,
        default=5,
        help="Temporal window size (default: 5)"
    )

    # Checkpoint and resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pretrained weights"
    )

    # Training mode
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation, no training"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)"
    )

    # Other
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options at the end of the command"
    )

    return parser.parse_args()


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )


def set_gpus(gpu_ids):
    """Set visible GPUs."""
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        logger.info(f"Setting CUDA_VISIBLE_DEVICES={gpu_ids}")
    else:
        # Use GPU 0 by default
        # Don't set CUDA_VISIBLE_DEVICES, let PyTorch use default GPU 0
        pass


def do_test(cfg, model, is_distributed=False):
    """Run evaluation."""
    if "evaluator" not in cfg.dataloader:
        logger.warning("No evaluator found in config, skipping evaluation")
        return {}

    # Convert to SyncBatchNorm if in distributed mode
    if is_distributed:
        has_sync_bn = any(isinstance(module, torch.nn.SyncBatchNorm)
                         for module in model.modules())
        if not has_sync_bn:
            logger.info("Converting model to use SyncBatchNorm for evaluation")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    ret = inference_on_dataset(
        model,
        instantiate(cfg.dataloader.test),
        instantiate(cfg.dataloader.evaluator),
    )
    print_csv_format(ret)
    return ret


def do_train(args, cfg, is_distributed=False):
    """Run training."""
    from unlanedet.engine.defaults import create_ddp_model
    from unlanedet.engine import AMPTrainer, SimpleTrainer, hooks

    # Create model
    logger.info("Building model...")
    model = instantiate(cfg.model)
    logger.info(f"Model:\n{model}")

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load pretrained weights if specified
    if hasattr(args, 'weights') and args.weights is not None:
        logger.info(f"Loading pretrained weights from {args.weights}")
        Checkpointer(model).load(args.weights)

    # Load resume checkpoint if specified
    if args.resume is not None:
        logger.info(f"Resuming training from {args.resume}")
        Checkpointer(model).load(args.resume)

    # Convert to DDP if distributed
    if is_distributed:
        logger.info("Converting model to use SyncBatchNorm")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Create optimizer
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)

    # Create data loader
    train_loader = instantiate(cfg.dataloader.train)

    # Wrap model for distributed training
    model = create_ddp_model(model, **cfg.train.ddp)

    # Create trainer
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(
        model, train_loader, optimizer
    )

    # Setup checkpointing
    checkpointer = BestCheckPointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    # Register hooks
    trainer.register_hooks([
        hooks.IterationTimer(),
        hooks.SetEpochHook(train_loader),
        hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
        hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
        if comm.is_main_process() else None,
        hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model, is_distributed)),
        hooks.BestCheckpointer(
            checkpointer=checkpointer,
            eval_period=cfg.train.eval_period,
            val_metric=cfg.dataloader.evaluator.metric
        ) if comm.is_main_process() else None,
    ])

    # Train
    logger.info("Starting training...")
    if args.resume is not None and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def apply_temporal_config(cfg, args):
    """Apply temporal module configuration."""
    if args.temporal is None:
        return cfg

    logger.info(f"Enabling temporal module: {args.temporal} (window={args.temporal_window})")

    # Update config with temporal settings
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'head'):
        # Store temporal config
        if not hasattr(cfg.model, 'temporal'):
            cfg.model.temporal = {}
        cfg.model.temporal['type'] = args.temporal
        cfg.model.temporal['window'] = args.temporal_window

    return cfg


def main(args):
    """Main training function."""
    setup_logging()

    # Handle GPU selection
    if args.gpus is not None:
        # Multi-GPU mode using --gpus
        gpu_ids = args.gpus
        num_gpus = len(args.gpus.split(','))
        is_distributed = num_gpus > 1
        logger.info(f"Multi-GPU training on GPUs: {gpu_ids} ({num_gpus} GPUs)")
        set_gpus(gpu_ids)
    elif args.num_gpus > 1:
        # Multi-GPU mode using --num-gpus
        gpu_ids = ','.join(str(i) for i in range(args.num_gpus))
        num_gpus = args.num_gpus
        is_distributed = True
        logger.info(f"Multi-GPU training on GPUs: {gpu_ids} ({num_gpus} GPUs)")
        set_gpus(gpu_ids)
    else:
        # Single GPU mode
        gpu_ids = str(args.gpu)
        num_gpus = 1
        is_distributed = False
        logger.info(f"Single GPU training on GPU {args.gpu}")
        # Only set CUDA_VISIBLE_DEVICES if explicit GPU is specified
        if args.gpu != 0:
            set_gpus(gpu_ids)

    # Load config
    logger.info(f"Loading config from {args.config}")
    cfg = LazyConfig.load(args.config)

    # Apply config overrides
    if args.opts:
        cfg = LazyConfig.apply_overrides(cfg, args.opts)

    # Apply temporal configuration
    cfg = apply_temporal_config(cfg, args)

    # Update output directory
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    else:
        # Generate default output directory name
        config_name = Path(args.config).stem
        if args.temporal:
            output_name = f"{config_name}_{args.temporal}_t{args.temporal_window}"
        else:
            output_name = config_name
        cfg.train.output_dir = str(PROJECT_ROOT / "output" / output_name)

    # Create output directory
    os.makedirs(cfg.train.output_dir, exist_ok=True)

    # Log configuration
    logger.info(f"Output directory: {cfg.train.output_dir}")
    logger.info(f"Config:\n{LazyConfig.to_py(cfg)}")

    # Evaluation only
    if args.eval_only:
        logger.info("Running evaluation only")
        model = instantiate(cfg.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if args.weights is None:
            logger.error("--weights is required for --eval-only")
            sys.exit(1)

        logger.info(f"Loading weights from {args.weights}")
        Checkpointer(model).load(args.weights)

        do_test(cfg, model, is_distributed)
        return

    # Training
    if is_distributed:
        # Launch distributed training
        launch(
            main_func=do_train,
            num_gpus_per_machine=num_gpus,
            num_machines=1,
            machine_rank=0,
            dist_url="auto",
            args=(args, cfg, True),
            max_retries=5,  # 添加最大重试次数
        )
    else:
        # Single GPU training
        do_train(args, cfg, is_distributed=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
