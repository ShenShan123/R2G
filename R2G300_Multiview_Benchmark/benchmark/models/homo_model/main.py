"""新版同构工程入口：统一配置 → 数据 → 编码/GNN/Head → 训练及保存。"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dataset import (
    STAGES,
    TASK_REGISTRY,
    convert_sample_to_homogeneous,
    discover_records,
    filter_split_without_targets,
    fit_transform_records,
    get_task_spec,
    graph_structure_summary,
    load_native_graph,
    schema_from_sample,
    split_by_family,
)
from model import HomogeneousGNN
from train import train_formal, train_smoke

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, action="append", default=[])
    parser.add_argument("--graph-path", type=Path, action="append", default=[])
    parser.add_argument("--view", choices=("B", "C", "D", "E", "F"), default="B")
    parser.add_argument("--stage", choices=STAGES, default="route")
    parser.add_argument("--task", choices=tuple(TASK_REGISTRY), default="wirelength")
    parser.add_argument(
        "--model", choices=("gat", "gine", "resgatedgcn"), default="gine"
    )
    parser.add_argument("--include-geometry-edges", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-head-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="Recompute GNN layers during backward to bound full-graph activation memory.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--test-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--loss", choices=("smooth_l1", "mse"), default="smooth_l1")
    parser.add_argument("--target-chunk-size", type=int, default=65536)
    parser.add_argument("--max-targets-per-design", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-targets", type=int, default=512)
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
    parser.add_argument(
        "--amp",
        choices=("none", "bfloat16"),
        default="none",
        help=(
            "CUDA autocast policy. bfloat16 reduces full-graph activation memory "
            "without changing model width, graph topology, or supervision."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "Result")
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        result = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if result.type == "cuda":
            torch.cuda.set_device(result)
        return result
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    result = torch.device(value)
    if result.type == "cuda":
        torch.cuda.set_device(result)
    return result


def configure_logger(directory: Path) -> logging.Logger:
    # 一个运行只写一个新目录，不覆盖已有结果。
    directory.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger("r2g2_homogeneous")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(directory / "train.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def input_roots(args) -> list[Path]:
    if args.graph_path:
        return args.graph_path
    if args.data_root:
        return args.data_root
    return [HERE / "Dataset" / args.view]


def jsonable_args(args) -> dict:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, list):
            result[key] = [str(item) for item in value]
        else:
            result[key] = value
    return result


def main() -> None:
    args = parse_args()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    run_name = args.run_name or (
        f"{args.model}_{args.task}_HOMO_{args.view}_{args.stage}_seed{args.seed}"
        + ("_smoke" if args.smoke_test else "")
        + f"_L{args.num_layers}H{args.num_head_layers}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("--run-name 必须是单个目录名")
    output = args.output_dir.expanduser().resolve() / run_name
    logger = configure_logger(output)
    logger.info(
        "Native R2G2.0 homogeneous experiment | stage=%s view=%s task=%s model=%s | "
        "model_seed=%d split_seed=%d",
        args.stage,
        args.view,
        args.task,
        args.model,
        args.seed,
        split_seed,
    )
    spec = get_task_spec(args.task)
    if not spec.available_in(args.view):
        raise ValueError(f"Task {args.task!r} is unavailable in {args.view} view")
    records = discover_records(input_roots(args), view_type=args.view, stage=args.stage)
    if not records:
        raise FileNotFoundError("No matching heterograph.pt was discovered")
    logger.info(
        "Discovered %d records; target schemas=%s; unit=%s; transform=%s",
        len(records),
        spec.schemas(args.view),
        spec.unit,
        spec.target_transform,
    )
    device = choose_device(args.device)
    logger.info(
        "Device=%s | geometry_edges=%s | amp=%s",
        device,
        args.include_geometry_edges,
        args.amp,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    split = None
    split_document = None
    skipped_no_target = None
    representative_record = records[0]
    if not args.smoke_test:
        # Assign families using the complete catalog before removing graphs
        # that have no surviving supervision in a reduced view.
        raw_split = split_by_family(records, split_seed)
        split, skipped_no_target = filter_split_without_targets(
            raw_split, spec, args.include_geometry_edges
        )
        split_document = split.summary()
        split_document["discovered_record_count"] = len(records)
        split_document["skipped_no_target"] = {
            name: {
                "record_count": len(sample_ids),
                "sample_ids": sample_ids,
            }
            for name, sample_ids in skipped_no_target.items()
        }
        representative_record = split.train[0]
        logger.info(
            "Split | train=%d records/%d families | validation=%d records/%d families | "
            "test=%d records/%d families | skipped_no_target=%s",
            split_document["train"]["record_count"],
            split_document["train"]["family_count"],
            split_document["val"]["record_count"],
            split_document["val"]["family_count"],
            split_document["test"]["record_count"],
            split_document["test"]["family_count"],
            {name: len(values) for name, values in skipped_no_target.items()},
        )

    first_sample = load_native_graph(
        representative_record.graph_path,
        task=spec,
        expected_view=args.view,
        expected_stage=args.stage,
        include_geometry=args.include_geometry_edges,
    )
    schema = schema_from_sample(first_sample)
    first_homogeneous = convert_sample_to_homogeneous(first_sample, schema)
    logger.info(
        "Conversion=%s | hetero_nodes=%s | homogeneous_nodes=%d | "
        "homogeneous_edges=%d | node_types=%d | edge_types=%d | targets=%d/%s",
        first_homogeneous.conversion_method,
        {
            key: int(first_sample.graph[key].num_nodes)
            for key in first_sample.graph.node_types
        },
        int(first_homogeneous.num_nodes),
        int(first_homogeneous.num_edges),
        len(schema.node_type_names),
        len(schema.edge_type_names),
        int(first_homogeneous.target_y.numel()),
        first_homogeneous.target_kind,
    )
    model = HomogeneousGNN(
        schema=schema,
        model_name=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        head_layers=args.num_head_layers,
        heads=args.heads,
        dropout=args.dropout,
        activation_checkpointing=args.activation_checkpointing,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    if args.smoke_test:
        first_homogeneous = first_homogeneous.to(device)
        result, transform = train_smoke(
            data=first_homogeneous, model=model, spec=spec, args=args, logger=logger
        )
        mode = "single_design_smoke"
    else:
        assert split is not None
        transform = fit_transform_records(
            split.train, spec, args.include_geometry_edges
        )
        result = train_formal(
            model=model,
            schema=schema,
            split=split,
            spec=spec,
            transform=transform,
            args=args,
            device=device,
            logger=logger,
        )
        mode = "formal_design_family_split"

    metrics = {
        "mode": mode,
        "view": args.view,
        "stage": args.stage,
        "task": args.task,
        "model": args.model,
        "parameter_count": parameter_count,
        "split": split_document or result.get("split"),
        "target_transform": vars(transform),
        **{key: value for key, value in result.items() if key != "best_state"},
    }
    if device.type == "cuda":
        metrics["peak_gpu_memory_mib"] = {
            "allocated": torch.cuda.max_memory_allocated(device) / (1024**2),
            "reserved": torch.cuda.max_memory_reserved(device) / (1024**2),
        }
    document = {
        "schema": "r2g2_homogeneous_training_results_v2",
        "status": "complete",
        "config": jsonable_args(args),
        "input_structure": {
            "representative_heterogeneous": graph_structure_summary(
                representative_record.graph_path
            ),
            "homogeneous_conversion": {
                "method": first_homogeneous.conversion_method,
                "num_nodes": int(first_homogeneous.num_nodes),
                "num_edges": int(first_homogeneous.num_edges),
                "node_type_count": len(schema.node_type_names),
                "edge_type_count": len(schema.edge_type_names),
                "schema": schema.as_dict(),
            },
        },
        "metrics": metrics,
    }
    (output / "results.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    torch.save(result["best_state"], output / "best_checkpoint.pt")
    logger.info(
        "PASS | parameters=%d | results=%s | checkpoint=%s",
        parameter_count,
        output / "results.json",
        output / "best_checkpoint.pt",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("r2g2_homogeneous").exception("Experiment failed")
        raise
