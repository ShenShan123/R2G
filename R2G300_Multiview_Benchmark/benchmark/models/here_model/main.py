#!/usr/bin/env python3
"""Train native-view HGT/RGCN/HAN models on one independent stage dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dataset import (
    STAGES,
    TASK_REGISTRY,
    VIEWS,
    discover_records,
    filter_split_without_targets,
    fit_quantile_loss_weighting,
    fit_transform_from_records,
    get_task_spec,
    graph_structure_summary,
    load_native_graph,
    split_records_by_family,
)
from model import SUPPORTED_MODELS, NativeTaskHead, build_model
from train import make_smoke_subgraph, smoke_train_one_graph, train_multidesign

HERE = Path(__file__).resolve().parent

# 每个视图独立放在本工程 Dataset/<VIEW>，不依赖原服务器路径。
STAGE_DATASET_PATHS = {
    stage: {view: HERE / "Dataset" / view for view in VIEWS} for stage in STAGES
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="route")
    parser.add_argument("--view", choices=(*VIEWS, "ALL"), default="B")
    parser.add_argument(
        "--graph-path",
        type=Path,
        action="append",
        default=[],
        help="Explicit heterograph.pt; repeat for formal multi-design training",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Dataset root using <family>/<version>/<stage>/heterograph.pt; "
            "repeat when multiple roots are needed"
        ),
    )
    parser.add_argument("--task", choices=tuple(TASK_REGISTRY), default="wirelength")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="hgt")
    parser.add_argument("--include-geometry-edges", action="store_true")
    parser.add_argument(
        "--edge-pair-mode",
        choices=("directed", "symmetric"),
        default="directed",
        help="Use an exchange-invariant endpoint readout for unordered edge tasks",
    )
    parser.add_argument(
        "--deduplicate-undirected-targets",
        action="store_true",
        help="Supervise only one stored orientation of each unordered edge pair",
    )
    parser.add_argument(
        "--cc-dynamic-pair-geometry",
        action="store_true",
        help="Derive causal Net-pair geometry in memory from existing Pin coordinates",
    )
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Bounded real-graph forward/backward test; not a paper data split",
    )
    parser.add_argument("--smoke-max-nodes-per-type", type=int, default=512)
    parser.add_argument("--smoke-seed-targets", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-head-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-bases", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--feature-transform",
        choices=("signed_log1p", "raw"),
        default="signed_log1p",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--test-every",
        type=int,
        default=5,
        help=(
            "Evaluate and log the current checkpoint on the held-out test set "
            "every N epochs; 0 disables periodic test monitoring. Test metrics "
            "never participate in checkpoint selection."
        ),
    )
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--loss", choices=("smooth_l1", "mse"), default="smooth_l1")
    parser.add_argument(
        "--target-transform-override",
        choices=("native", "standard", "log1p_standard", "log10_standard"),
        default="native",
        help="Ablation-only target transform override; native uses the task registry",
    )
    parser.add_argument("--max-targets-per-design", type=int, default=0)
    parser.add_argument(
        "--train-max-targets-per-design",
        type=int,
        default=0,
        help="Training-only target cap; validation/test remain complete",
    )
    parser.add_argument(
        "--target-sampling",
        choices=("uniform", "stratified"),
        default="uniform",
        help="Training-only sampling strategy when the training target cap is active",
    )
    parser.add_argument(
        "--irdrop-quantile-weighting",
        action="store_true",
        help="Weight train-only IR target intervals P0/P50/P90/P95 by 1/2/3/5",
    )
    parser.add_argument("--shuffle-train-designs", action="store_true")
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.0,
        help="ReduceLROnPlateau factor; 0 disables the scheduler",
    )
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument(
        "--checkpoint-metric",
        choices=("loss", "r2"),
        default="loss",
        help="Validation metric used exclusively for checkpoint selection",
    )
    parser.add_argument(
        "--target-chunk-size",
        type=int,
        default=131072,
        help=(
            "Maximum supervised rows materialized by the task head at once; "
            "0 disables chunking. This does not subsample labels."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "Independent design-family split seed. Defaults to --seed for "
            "backward compatibility; set it explicitly to vary only the data "
            "partition while holding model initialization fixed."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=HERE / "Result")
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def configure_logger(output_dir: Path) -> logging.Logger:
    # 一个运行只写一个新目录，避免相同配置覆盖既有日志和检查点。
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger("r2g2_native")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(output_dir / "train.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu")
    return torch.device(value)


def _input_roots(args, view: str) -> list[Path]:
    if args.graph_path:
        return args.graph_path
    if args.data_root:
        return args.data_root
    return [STAGE_DATASET_PATHS[args.stage][view]]


def _jsonable_args(args) -> dict:
    return {
        key: [str(item) for item in value]
        if isinstance(value, list)
        else str(value)
        if isinstance(value, Path)
        else value
        for key, value in vars(args).items()
    }


def _structure_audit(args, views: tuple[str, ...]) -> dict:
    report = {}
    for view in views:
        records = discover_records(
            _input_roots(args, view), view_type=view, stage=args.stage
        )
        if not records:
            raise FileNotFoundError(
                f"No {view}/{args.stage} heterograph.pt was discovered"
            )
        report[view] = {
            "record_count": len(records),
            "representative": graph_structure_summary(records[0].graph_path),
        }
    return report


def main() -> None:
    args = parse_args()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    views = VIEWS if args.view == "ALL" else (args.view,)
    run_name = args.run_name or (
        f"{args.model}_{args.task}_{args.view}_{args.stage}_seed{args.seed}"
        + ("_smoke" if args.smoke_test else "")
        + f"_L{args.num_layers}H{args.num_head_layers}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("--run-name 必须是单个目录名")
    output_dir = args.output_dir.expanduser().resolve() / run_name
    logger = configure_logger(output_dir)
    logger.info(
        "Native R2G2.0 experiment | stage=%s view=%s task=%s model=%s | "
        "model_seed=%d split_seed=%d",
        args.stage,
        args.view,
        args.task,
        args.model,
        args.seed,
        split_seed,
    )
    structure = _structure_audit(args, views)
    results_document = {
        "schema": "r2g2_training_results_v2",
        "status": "complete" if args.inspect_only else "running",
        "config": _jsonable_args(args),
        "input_structure": structure,
    }
    for view in views:
        first = structure[view]["representative"]
        logger.info(
            "%s input=%s | nodes=%s | message_edges=%s",
            view,
            first["path"],
            {k: v["count"] for k, v in first["nodes"].items()},
            first["message_edge_types"],
        )
    if args.inspect_only:
        results_path = output_dir / "results.json"
        results_path.write_text(
            json.dumps(results_document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Structure audit completed: %s", results_path)
        return
    if args.view == "ALL":
        raise ValueError(
            "--view ALL is structure-audit only; train one native view at a time"
        )

    spec = get_task_spec(args.task)
    if args.target_transform_override != "native":
        if args.task != "cc":
            raise ValueError(
                "target transform override is currently reserved for task=cc"
            )
        spec = replace(spec, target_transform=args.target_transform_override)
    if not spec.available_in(args.view):
        raise ValueError(
            f"Task {spec.name!r} is unavailable in {args.view} view by topology contract"
        )
    if args.irdrop_quantile_weighting and args.task != "irdrop":
        raise ValueError("--irdrop-quantile-weighting is valid only for task=irdrop")
    if args.deduplicate_undirected_targets and args.task != "cc":
        raise ValueError("--deduplicate-undirected-targets is valid only for task=cc")
    if args.cc_dynamic_pair_geometry and args.task != "cc":
        raise ValueError("--cc-dynamic-pair-geometry is valid only for task=cc")
    if args.edge_pair_mode == "symmetric" and args.task != "cc":
        raise ValueError("--edge-pair-mode symmetric is currently reserved for task=cc")
    records = discover_records(
        _input_roots(args, args.view), view_type=args.view, stage=args.stage
    )
    logger.info(
        "Discovered %d records; target schemas=%s; unit=%s; transform=%s",
        len(records),
        spec.schemas(args.view),
        spec.unit,
        spec.target_transform,
    )
    device = choose_device(args.device)
    logger.info("Device=%s | geometry_edges=%s", device, args.include_geometry_edges)
    if device.type == "cuda":
        # Each batch worker exposes exactly one physical GPU through
        # CUDA_VISIBLE_DEVICES.  The no-argument API is also compatible with
        # the project's PyTorch 2.7 build, whose explicit device path rejects
        # an otherwise valid cuda:0 argument.
        torch.cuda.reset_peak_memory_stats()
        logger.info(
            "GPU=%s | CUDA_VISIBLE_DEVICES=%s",
            torch.cuda.get_device_name(device),
            os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        )

    init_sample = load_native_graph(
        records[0].graph_path,
        task=spec,
        expected_view=args.view,
        expected_stage=args.stage,
        include_geometry=args.include_geometry_edges,
    )
    if args.smoke_test:
        if len(records) != 1:
            logger.warning("Smoke test uses only the first of %d records", len(records))
        init_sample = make_smoke_subgraph(
            init_sample,
            max_nodes_per_type=args.smoke_max_nodes_per_type,
            max_seed_targets=args.smoke_seed_targets,
        )
        logger.warning(
            "Single-design row split is active only for connectivity testing; "
            "it must not be reported as validation/test performance"
        )

    model = build_model(
        args.model,
        view_type=args.view,
        node_types=tuple(init_sample.graph.node_types),
        edge_types=init_sample.message_edge_types,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        feature_transform=args.feature_transform,
        heads=args.heads,
        num_bases=args.num_bases,
    ).to(device)
    head = NativeTaskHead(
        args.hidden_dim,
        args.num_head_layers,
        edge_pair_mode=args.edge_pair_mode,
        cc_dynamic_pair_geometry=args.cc_dynamic_pair_geometry,
    ).to(device)

    if args.smoke_test:
        result = smoke_train_one_graph(
            sample=init_sample,
            model=model,
            head=head,
            spec=spec,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            max_targets=args.max_targets_per_design,
            seed=args.seed,
            logger=logger,
        )
        parameter_count = sum(
            p.numel() for module in (model, head) for p in module.parameters()
        )
        metrics = {
            "mode": "single_design_smoke",
            "view": args.view,
            "stage": args.stage,
            "task": args.task,
            "parameter_count": parameter_count,
            **{key: value for key, value in result.items() if key != "best_state"},
        }
    else:
        split = split_records_by_family(records, split_seed)
        split, skipped_no_target = filter_split_without_targets(
            split,
            spec,
            include_geometry=args.include_geometry_edges,
        )
        split_summary = split.summary()
        split_summary["discovered_record_count"] = len(records)
        split_summary["skipped_no_target"] = {
            name: {
                "record_count": len(sample_ids),
                "sample_ids": sample_ids,
            }
            for name, sample_ids in skipped_no_target.items()
        }
        logger.info(
            "Split | train=%d records/%d families | "
            "validation=%d records/%d families | test=%d records/%d families | "
            "skipped_no_target=%s",
            split_summary["train"]["record_count"],
            split_summary["train"]["family_count"],
            split_summary["val"]["record_count"],
            split_summary["val"]["family_count"],
            split_summary["test"]["record_count"],
            split_summary["test"]["family_count"],
            {name: len(values) for name, values in skipped_no_target.items()},
        )
        transform, _, _ = fit_transform_from_records(
            split.train, spec, include_geometry=args.include_geometry_edges
        )
        loss_weighting = None
        if args.irdrop_quantile_weighting:
            loss_weighting = fit_quantile_loss_weighting(
                split.train,
                spec,
                include_geometry=args.include_geometry_edges,
                quantiles=(0.50, 0.90, 0.95),
                multipliers=(1.0, 2.0, 3.0, 5.0),
            )
            logger.info(
                "IR train-only quantile weighting | thresholds_mV=%s | weights=%s",
                loss_weighting.thresholds,
                loss_weighting.multipliers,
            )
        # Materialize lazy per-schema projections before optimizer creation.
        init_sample.to(device)
        with torch.no_grad():
            model(init_sample.graph)
        parameter_count = sum(
            p.numel() for module in (model, head) for p in module.parameters()
        )
        del init_sample
        result = train_multidesign(
            model=model,
            head=head,
            split=split,
            spec=spec,
            transform=transform,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            patience=args.patience,
            max_targets=args.max_targets_per_design,
            train_max_targets=args.train_max_targets_per_design,
            target_chunk_size=args.target_chunk_size,
            include_geometry=args.include_geometry_edges,
            test_every=args.test_every,
            logger=logger,
            target_sampling=args.target_sampling,
            deduplicate_undirected=args.deduplicate_undirected_targets,
            loss_weighting=loss_weighting,
            shuffle_train=args.shuffle_train_designs,
            seed=args.seed,
            scheduler_factor=args.scheduler_factor,
            scheduler_patience=args.scheduler_patience,
            min_lr=args.min_lr,
            checkpoint_metric=args.checkpoint_metric,
        )
        metrics = {
            "mode": "formal_design_family_split",
            "view": args.view,
            "stage": args.stage,
            "task": args.task,
            "parameter_count": parameter_count,
            "split": {
                "summary": split_summary,
                "sample_ids": split.ids(),
                "contract": (
                    "Train, validation, and test are pairwise disjoint by "
                    "sample ID and design family."
                ),
            },
            "target_transform": vars(transform),
            "loss_weighting": (
                vars(loss_weighting) if loss_weighting is not None else None
            ),
            **{key: value for key, value in result.items() if key != "best_state"},
        }

    if device.type == "cuda":
        metrics["peak_gpu_memory_mib"] = {
            "allocated": torch.cuda.max_memory_allocated() / (1024**2),
            "reserved": torch.cuda.max_memory_reserved() / (1024**2),
        }
        logger.info(
            "CUDA peak | allocated=%.1f MiB | reserved=%.1f MiB",
            metrics["peak_gpu_memory_mib"]["allocated"],
            metrics["peak_gpu_memory_mib"]["reserved"],
        )
    results_document["status"] = "complete"
    results_document["metrics"] = metrics
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(results_document, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    torch.save(result["best_state"], output_dir / "best_checkpoint.pt")
    logger.info(
        "PASS | parameters=%d | results=%s | checkpoint=%s",
        parameter_count,
        results_path,
        output_dir / "best_checkpoint.pt",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("r2g2_native").exception("Experiment failed")
        raise
