"""训练核心：逐设计全图传播，分块预测头，验证选模后评估 Test。"""

from __future__ import annotations

import copy
import math
import random
import time

import torch
from dataset import (
    TargetTransform,
    convert_sample_to_homogeneous,
    fit_transform_chunks,
    load_native_graph,
)
from torch.nn import functional as F


class MetricAccumulator:
    def __init__(self) -> None:
        self.weight = self.loss_sum = 0.0
        self.abs_error = self.squared_error = 0.0
        self.target_sum = self.target_squared_sum = 0.0

    def update(self, loss, prediction, target, weight) -> None:
        prediction = prediction.detach().double()
        target = target.detach().double()
        weight = weight.detach().double()
        amount = float(weight.sum())
        error = prediction - target
        self.weight += amount
        self.loss_sum += float(loss.detach()) * amount
        self.abs_error += float((error.abs() * weight).sum())
        self.squared_error += float((error.square() * weight).sum())
        self.target_sum += float((target * weight).sum())
        self.target_squared_sum += float((target.square() * weight).sum())

    def metrics(self) -> dict[str, float]:
        if self.weight <= 0:
            return {key: math.nan for key in ("loss", "mae", "rmse", "r2")}
        sst = self.target_squared_sum - self.target_sum**2 / self.weight
        return {
            "loss": self.loss_sum / self.weight,
            "mae": self.abs_error / self.weight,
            "rmse": math.sqrt(self.squared_error / self.weight),
            "r2": 1.0 - self.squared_error / sst if sst > 0 else math.nan,
        }


def weighted_loss(prediction, target, weight, loss_name: str):
    if loss_name == "smooth_l1":
        rows = F.smooth_l1_loss(prediction, target, reduction="none")
    else:
        rows = F.mse_loss(prediction, target, reduction="none")
    return (rows * weight).sum() / weight.sum().clamp_min(1.0e-12)


def limited_rows(count: int, maximum: int, device=None) -> torch.Tensor:
    if maximum <= 0 or count <= maximum:
        return torch.arange(count, device=device)
    return torch.linspace(0, count - 1, maximum, device=device).round().long()


def run_one_graph(
    *,
    data,
    model,
    transform,
    optimizer,
    loss_name,
    target_chunk_size,
    selected_rows=None,
    amp="none",
) -> MetricAccumulator:
    training = optimizer is not None
    data = data.to(next(model.parameters()).device)
    if selected_rows is None:
        rows = torch.arange(data.target_y.numel(), device=data.target_y.device)
    else:
        rows = selected_rows.to(data.target_y.device)
    if not rows.numel():
        raise ValueError("No supervised row was selected")
    chunk_size = rows.numel() if target_chunk_size <= 0 else target_chunk_size
    chunks = [
        rows[start : start + chunk_size] for start in range(0, rows.numel(), chunk_size)
    ]
    total_weight = data.target_weight[rows].sum()
    accumulator = MetricAccumulator()
    if training:
        optimizer.zero_grad(set_to_none=True)
    amp_enabled = amp == "bfloat16" and data.x.device.type == "cuda"
    with torch.autocast(
        device_type=data.x.device.type,
        dtype=torch.bfloat16 if amp_enabled else torch.float32,
        enabled=amp_enabled,
    ):
        embeddings = model.encode(data)
        prediction_embeddings = (
            embeddings.detach().requires_grad_(True) if training else embeddings
        )
        for chunk in chunks:
            prediction = model.predict(data, prediction_embeddings, chunk)
            raw_target = data.target_y[chunk]
            weight = data.target_weight[chunk]
            loss = weighted_loss(
                prediction, transform.transform(raw_target), weight, loss_name
            )
            if training:
                (loss * weight.sum() / total_weight).backward()
            accumulator.update(loss, transform.inverse(prediction), raw_target, weight)
    if training:
        if prediction_embeddings.grad is None:
            raise RuntimeError("Task head produced no embedding gradient")
        torch.autograd.backward(embeddings, prediction_embeddings.grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return accumulator


def merge_accumulators(
    destination: MetricAccumulator, source: MetricAccumulator
) -> None:
    for name in (
        "weight",
        "loss_sum",
        "abs_error",
        "squared_error",
        "target_sum",
        "target_squared_sum",
    ):
        setattr(destination, name, getattr(destination, name) + getattr(source, name))


def run_records(
    *,
    records,
    model,
    schema,
    spec,
    transform,
    optimizer,
    loss_name,
    target_chunk_size,
    max_targets,
    include_geometry,
    epoch_seed=None,
    amp="none",
) -> dict[str, float]:
    ordered = list(records)
    if optimizer is not None and epoch_seed is not None:
        random.Random(epoch_seed).shuffle(ordered)
    model.train(optimizer is not None)
    total = MetricAccumulator()
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for record in ordered:
            sample = load_native_graph(
                record.graph_path,
                task=spec,
                expected_view=record.view_type,
                expected_stage=record.stage,
                include_geometry=include_geometry,
            )
            data = convert_sample_to_homogeneous(sample, schema)
            rows = limited_rows(data.target_y.numel(), max_targets)
            result = run_one_graph(
                data=data,
                model=model,
                transform=transform,
                optimizer=optimizer,
                loss_name=loss_name,
                target_chunk_size=target_chunk_size,
                selected_rows=rows,
                amp=amp,
            )
            merge_accumulators(total, result)
            del sample, data, result
    return total.metrics()


def format_metrics(metrics: dict[str, float]) -> str:
    return " | ".join(
        f"{key}={metrics[key]:.8g}" for key in ("loss", "mae", "rmse", "r2")
    )


def train_formal(
    *,
    model,
    schema,
    split,
    spec,
    transform,
    args,
    device,
    logger,
) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train = run_records(
            records=split.train,
            model=model,
            schema=schema,
            spec=spec,
            transform=transform,
            optimizer=optimizer,
            loss_name=args.loss,
            target_chunk_size=args.target_chunk_size,
            max_targets=args.max_targets_per_design,
            include_geometry=args.include_geometry_edges,
            epoch_seed=args.seed + epoch,
            amp=args.amp,
        )
        val = run_records(
            records=split.val,
            model=model,
            schema=schema,
            spec=spec,
            transform=transform,
            optimizer=None,
            loss_name=args.loss,
            target_chunk_size=args.target_chunk_size,
            max_targets=args.max_targets_per_design,
            include_geometry=args.include_geometry_edges,
            amp=args.amp,
        )
        logger.info("Epoch %03d | train      | %s", epoch, format_metrics(train))
        logger.info("Epoch %03d | validation | %s", epoch, format_metrics(val))
        item = {"epoch": epoch, "train": train, "validation": val}
        if args.test_every > 0 and epoch % args.test_every == 0:
            test_monitor = run_records(
                records=split.test,
                model=model,
                schema=schema,
                spec=spec,
                transform=transform,
                optimizer=None,
                loss_name=args.loss,
                target_chunk_size=args.target_chunk_size,
                max_targets=args.max_targets_per_design,
                include_geometry=args.include_geometry_edges,
                amp=args.amp,
            )
            item["test_monitor"] = test_monitor
            logger.info(
                "Epoch %03d | test-monitor | %s",
                epoch,
                format_metrics(test_monitor),
            )
        history.append(item)
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced")
    model.load_state_dict(best_state)
    test = run_records(
        records=split.test,
        model=model,
        schema=schema,
        spec=spec,
        transform=transform,
        optimizer=None,
        loss_name=args.loss,
        target_chunk_size=args.target_chunk_size,
        max_targets=args.max_targets_per_design,
        include_geometry=args.include_geometry_edges,
        amp=args.amp,
    )
    logger.info("Final best epoch %03d | test | %s", best_epoch, format_metrics(test))
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "test": test,
        "history": history,
        "seconds": time.perf_counter() - started,
        "best_state": best_state,
    }


def train_smoke(*, data, model, spec, args, logger) -> tuple[dict, TargetTransform]:
    count = data.target_y.numel()
    maximum = min(count, args.smoke_max_targets)
    generator = torch.Generator().manual_seed(args.seed)
    rows = torch.randperm(count, generator=generator)[:maximum]
    n_train = max(1, int(maximum * 0.6))
    n_val = max(1, int(maximum * 0.2))
    split = {
        "train": rows[:n_train],
        "val": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val :],
    }
    if not split["test"].numel():
        raise ValueError("Smoke graph needs at least five valid targets")
    transform = fit_transform_chunks(
        [(data.target_y[split["train"]], data.target_weight[split["train"]])],
        spec.target_transform,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history = []
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_acc = run_one_graph(
            data=data,
            model=model,
            transform=transform,
            optimizer=optimizer,
            loss_name=args.loss,
            target_chunk_size=args.target_chunk_size,
            selected_rows=split["train"],
            amp=args.amp,
        )
        model.eval()
        with torch.no_grad():
            val_acc = run_one_graph(
                data=data,
                model=model,
                transform=transform,
                optimizer=None,
                loss_name=args.loss,
                target_chunk_size=args.target_chunk_size,
                selected_rows=split["val"],
                amp=args.amp,
            )
            test_acc = run_one_graph(
                data=data,
                model=model,
                transform=transform,
                optimizer=None,
                loss_name=args.loss,
                target_chunk_size=args.target_chunk_size,
                selected_rows=split["test"],
                amp=args.amp,
            )
        train = train_acc.metrics()
        val = val_acc.metrics()
        test_monitor = test_acc.metrics()
        logger.info("Epoch %03d | train      | %s", epoch, format_metrics(train))
        logger.info("Epoch %03d | validation | %s", epoch, format_metrics(val))
        logger.info(
            "Epoch %03d | test-monitor | %s", epoch, format_metrics(test_monitor)
        )
        history.append(
            {
                "epoch": epoch,
                "train": train,
                "validation": val,
                "test_monitor": test_monitor,
            }
        )
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final = run_one_graph(
            data=data,
            model=model,
            transform=transform,
            optimizer=None,
            loss_name=args.loss,
            target_chunk_size=args.target_chunk_size,
            selected_rows=split["test"],
            amp=args.amp,
        ).metrics()
    logger.info("Final best epoch %03d | test | %s", best_epoch, format_metrics(final))
    return (
        {
            "warning": "Single-design target-row split: smoke test only",
            "split": {key: int(value.numel()) for key, value in split.items()},
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "test": final,
            "history": history,
            "seconds": time.perf_counter() - started,
            "best_state": best_state,
        },
        transform,
    )
