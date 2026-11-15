"""训练核心：目标分块、加权损失、反向传播、验证选模及最终 Test。"""

from __future__ import annotations

import copy
import math
import random
import time

import torch
from dataset import (
    FittedTargetTransform,
    NativeGraphSample,
    NativeRecord,
    QuantileLossWeighting,
    TargetLocation,
    TaskSpec,
    _fit_weighted_statistics,
    _limit_indices,
    load_native_graph,
    target_tensors,
)
from model import NativeTaskHead
from torch import nn
from torch.nn import functional as F


def _deduplicate_undirected_indices(
    data, location: TargetLocation, valid: torch.Tensor
) -> torch.Tensor:
    """Return one valid stored orientation for every unordered edge pair."""

    if location.kind != "edge" or location.store_key[0] != location.store_key[2]:
        return torch.nonzero(valid, as_tuple=False).flatten()
    edge_index = data[location.store_key].edge_index
    edge_count = edge_index.size(1)
    if edge_count % 2 == 0 and torch.equal(
        edge_index[:, 0::2], edge_index[:, 1::2].flip(0)
    ):
        even = torch.arange(0, edge_count, 2, device=valid.device)
        odd = even + 1
        usable = valid[even] | valid[odd]
        return torch.where(valid[even], even, odd)[usable]

    indices = torch.nonzero(valid, as_tuple=False).flatten()
    source, target = edge_index[:, indices]
    low = torch.minimum(source, target)
    high = torch.maximum(source, target)
    pair_key = low * int(data[location.store_key[0]].num_nodes) + high
    order = torch.argsort(pair_key)
    sorted_key = pair_key[order]
    first = torch.ones_like(sorted_key, dtype=torch.bool)
    first[1:] = sorted_key[1:] != sorted_key[:-1]
    return indices[order[first]]


def _stratified_indices(
    values: torch.Tensor,
    indices: torch.Tensor,
    maximum: int,
    bins: int,
    transform: FittedTargetTransform,
) -> torch.Tensor:
    """Select approximately equal counts across transformed target quantiles."""

    if maximum <= 0 or indices.numel() <= maximum:
        return indices
    if bins < 2:
        raise ValueError("stratified target sampling needs at least two bins")
    estimate = indices[
        torch.linspace(
            0,
            indices.numel() - 1,
            min(indices.numel(), 65536),
            device=indices.device,
        )
        .round()
        .long()
    ]
    transformed = transform.transform(values)
    boundaries = torch.quantile(
        transformed[estimate],
        torch.linspace(0, 1, bins + 1, device=values.device)[1:-1],
    )
    assignments = torch.bucketize(transformed[indices], boundaries)
    selected = []
    base, remainder = divmod(maximum, bins)
    for bin_id in range(bins):
        candidates = indices[assignments == bin_id]
        quota = base + (1 if bin_id < remainder else 0)
        if candidates.numel() > quota:
            positions = (
                torch.linspace(0, candidates.numel() - 1, quota, device=indices.device)
                .round()
                .long()
            )
            candidates = candidates[positions]
        selected.append(candidates)
    return torch.cat(selected)


def _target_indices(
    data,
    location: TargetLocation,
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    maximum: int,
    sampling: str,
    transform: FittedTargetTransform,
    deduplicate_undirected: bool,
) -> torch.Tensor:
    indices = (
        _deduplicate_undirected_indices(data, location, valid)
        if deduplicate_undirected
        else torch.nonzero(valid, as_tuple=False).flatten()
    )
    if sampling == "stratified":
        return _stratified_indices(values, indices, maximum, 8, transform)
    if sampling != "uniform":
        raise ValueError(f"Unknown target sampling mode {sampling!r}")
    if maximum <= 0 or indices.numel() <= maximum:
        return indices
    positions = (
        torch.linspace(0, indices.numel() - 1, maximum, device=indices.device)
        .round()
        .long()
    )
    return indices[positions]


def _weighted_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    if loss_name == "smooth_l1":
        per_row = F.smooth_l1_loss(prediction, target, reduction="none")
    elif loss_name == "mse":
        per_row = F.mse_loss(prediction, target, reduction="none")
    else:
        raise ValueError(f"Unknown loss {loss_name!r}")
    return (per_row * weight).sum() / weight.sum().clamp_min(1.0e-12)


class WeightedRegressionAccumulator:
    def __init__(self) -> None:
        self.weight = self.loss_weight = 0.0
        self.loss_sum = self.abs_error = self.squared_error = 0.0
        self.target_sum = self.target_squared_sum = 0.0
        self.rows = 0

    def update(self, loss, prediction, target, weight) -> None:
        prediction = prediction.detach().double()
        target = target.detach().double()
        weight = weight.detach().double()
        amount = float(weight.sum())
        error = prediction - target
        self.rows += int(target.numel())
        self.weight += amount
        self.loss_sum += float(loss.detach()) * amount
        self.abs_error += float((error.abs() * weight).sum())
        self.squared_error += float((error.square() * weight).sum())
        self.target_sum += float((target * weight).sum())
        self.target_squared_sum += float((target.square() * weight).sum())

    def metrics(self) -> dict[str, float | int]:
        if self.weight <= 0:
            return {"loss": math.nan, "mae": math.nan, "rmse": math.nan, "r2": math.nan}
        sst = self.target_squared_sum - self.target_sum**2 / self.weight
        return {
            "loss": self.loss_sum / self.weight,
            "mae": self.abs_error / self.weight,
            "rmse": math.sqrt(self.squared_error / self.weight),
            "r2": 1.0 - self.squared_error / sst if sst > 0 else math.nan,
        }


def _format_metrics(metrics: dict[str, float | int]) -> str:
    return " | ".join(
        f"{name}={float(metrics[name]):.8g}" for name in ("loss", "mae", "rmse", "r2")
    )


def _sample_predictions(
    sample: NativeGraphSample,
    embeddings: dict[str, torch.Tensor],
    head: NativeTaskHead,
    transform: FittedTargetTransform,
    *,
    max_targets: int,
    loss_name: str,
    allowed_indices: dict[TargetLocation, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    losses = []
    outputs = []
    for location in sample.target_locations:
        values, valid, weights = target_tensors(sample.graph, location)
        if allowed_indices is None:
            indices = _limit_indices(valid, max_targets)
        else:
            candidates = allowed_indices.get(
                location, values.new_empty(0, dtype=torch.long)
            )
            indices = candidates[valid[candidates]]
            if max_targets > 0:
                indices = indices[:max_targets]
        if not indices.numel():
            continue
        prediction = head.predict(sample.graph, embeddings, location, indices)
        raw_target = values[indices]
        weight = weights[indices]
        loss = _weighted_loss(
            prediction, transform.transform(raw_target), weight, loss_name
        )
        losses.append((loss, weight.sum()))
        outputs.append((transform.inverse(prediction), raw_target, weight))
    if not losses:
        raise ValueError("No valid target remained after masking/sampling")
    total_weight = torch.stack([weight for _, weight in losses]).sum()
    total_loss = sum(loss * weight for loss, weight in losses) / total_weight
    return total_loss, outputs


def run_design_epoch(
    *,
    records: list[NativeRecord],
    model: nn.Module,
    head: NativeTaskHead,
    spec: TaskSpec,
    transform: FittedTargetTransform,
    device: torch.device,
    loss_name: str,
    optimizer: torch.optim.Optimizer | None,
    max_targets: int,
    train_max_targets: int,
    target_chunk_size: int,
    include_geometry: bool,
    target_sampling: str,
    deduplicate_undirected: bool,
    loss_weighting: QuantileLossWeighting | None,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    head.train(training)
    accumulator = WeightedRegressionAccumulator()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for record in records:
            sample = load_native_graph(
                record.graph_path,
                task=spec,
                expected_view=record.view_type,
                expected_stage=record.stage,
                include_geometry=include_geometry,
            ).to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            embeddings = model(sample.graph)
            batches: list[tuple[TargetLocation, torch.Tensor]] = []
            target_payload: dict[
                TargetLocation,
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            ] = {}
            total_weight = embeddings[next(iter(embeddings))].new_zeros(())
            for location in sample.target_locations:
                values, valid, weights = target_tensors(sample.graph, location)
                maximum = (
                    train_max_targets
                    if training and train_max_targets > 0
                    else max_targets
                )
                indices = _target_indices(
                    sample.graph,
                    location,
                    values,
                    valid,
                    maximum=maximum,
                    sampling=target_sampling if training else "uniform",
                    transform=transform,
                    deduplicate_undirected=deduplicate_undirected,
                )
                if not indices.numel():
                    continue
                loss_weights = weights
                if training and loss_weighting is not None:
                    loss_weights = weights * loss_weighting.weights_for(values)
                target_payload[location] = (values, weights, loss_weights)
                chunk_size = (
                    indices.numel() if target_chunk_size <= 0 else target_chunk_size
                )
                for start in range(0, indices.numel(), chunk_size):
                    chunk = indices[start : start + chunk_size]
                    batches.append((location, chunk))
                    total_weight = total_weight + loss_weights[chunk].sum()
            if not batches or not bool(total_weight > 0):
                raise ValueError("No valid target remained after masking/sampling")

            # Large pair tasks (notably Cc) can contain millions of supervised
            # rows.  Materialize only one task-head chunk at a time.  During
            # training the head sees detached embedding leaves, accumulates the
            # exact gradient for every row, then sends the summed embedding
            # gradients through the GNN backbone once.  This is mathematically
            # equivalent to one full readout loss but avoids retaining every
            # 4H pair tensor in GPU memory.
            prediction_embeddings = embeddings
            if training:
                prediction_embeddings = {
                    key: value.detach().requires_grad_(True)
                    for key, value in embeddings.items()
                }
            for location, indices in batches:
                values, metric_weights, loss_weights = target_payload[location]
                prediction = head.predict(
                    sample.graph, prediction_embeddings, location, indices
                )
                raw_target = values[indices]
                metric_weight = metric_weights[indices]
                loss_weight = loss_weights[indices]
                local_loss = _weighted_loss(
                    prediction,
                    transform.transform(raw_target),
                    loss_weight,
                    loss_name,
                )
                if training:
                    scaled_loss = local_loss * loss_weight.sum() / total_weight
                    scaled_loss.backward()
                metric_loss = _weighted_loss(
                    prediction,
                    transform.transform(raw_target),
                    metric_weight,
                    loss_name,
                )
                accumulator.update(
                    metric_loss,
                    transform.inverse(prediction),
                    raw_target,
                    metric_weight,
                )
                del (
                    prediction,
                    raw_target,
                    metric_weight,
                    loss_weight,
                    local_loss,
                    metric_loss,
                )
            if training:
                backbone_outputs = []
                backbone_gradients = []
                for node_type, value in embeddings.items():
                    gradient = prediction_embeddings[node_type].grad
                    if gradient is not None:
                        backbone_outputs.append(value)
                        backbone_gradients.append(gradient)
                if not backbone_outputs:
                    raise RuntimeError("Task head produced no embedding gradients")
                torch.autograd.backward(backbone_outputs, backbone_gradients)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(head.parameters()), 1.0
                )
                optimizer.step()
            del (
                sample,
                embeddings,
                prediction_embeddings,
                batches,
                target_payload,
                total_weight,
            )
    return accumulator.metrics()


def train_multidesign(
    *,
    model,
    head,
    split,
    spec,
    transform,
    device,
    epochs,
    lr,
    weight_decay,
    loss_name,
    patience,
    max_targets,
    target_chunk_size,
    include_geometry,
    test_every,
    logger,
    train_max_targets=0,
    target_sampling="uniform",
    deduplicate_undirected=False,
    loss_weighting=None,
    shuffle_train=False,
    seed=42,
    scheduler_factor=0.0,
    scheduler_patience=8,
    min_lr=1.0e-6,
    checkpoint_metric="loss",
) -> dict:
    split.validate()
    if test_every < 0:
        raise ValueError("test_every must be non-negative")
    if checkpoint_metric not in {"loss", "r2"}:
        raise ValueError("checkpoint_metric must be 'loss' or 'r2'")
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = None
    if scheduler_factor > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_lr,
        )
    best_value = math.inf if checkpoint_metric == "loss" else -math.inf
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_records = list(split.train)
        if shuffle_train:
            random.Random(seed + epoch).shuffle(train_records)
        train = run_design_epoch(
            records=train_records,
            model=model,
            head=head,
            spec=spec,
            transform=transform,
            device=device,
            loss_name=loss_name,
            optimizer=optimizer,
            max_targets=max_targets,
            train_max_targets=train_max_targets,
            target_chunk_size=target_chunk_size,
            include_geometry=include_geometry,
            target_sampling=target_sampling,
            deduplicate_undirected=deduplicate_undirected,
            loss_weighting=loss_weighting,
        )
        val = run_design_epoch(
            records=split.val,
            model=model,
            head=head,
            spec=spec,
            transform=transform,
            device=device,
            loss_name=loss_name,
            optimizer=None,
            max_targets=max_targets,
            train_max_targets=0,
            target_chunk_size=target_chunk_size,
            include_geometry=include_geometry,
            target_sampling="uniform",
            deduplicate_undirected=deduplicate_undirected,
            loss_weighting=None,
        )
        if scheduler is not None:
            scheduler.step(float(val["loss"]))
        candidate = float(val[checkpoint_metric])
        improved = (
            candidate < best_value
            if checkpoint_metric == "loss"
            else candidate > best_value
        )
        if improved:
            best_value = candidate
            best_loss = float(val["loss"])
            best_epoch = epoch
            stale = 0
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
            }
        else:
            stale += 1
        epoch_result = {"epoch": epoch, "train": train, "validation": val}
        logger.info("Epoch %03d | train      | %s", epoch, _format_metrics(train))
        logger.info("Epoch %03d | validation | %s", epoch, _format_metrics(val))
        if test_every > 0 and epoch % test_every == 0:
            monitored_test = run_design_epoch(
                records=split.test,
                model=model,
                head=head,
                spec=spec,
                transform=transform,
                device=device,
                loss_name=loss_name,
                optimizer=None,
                max_targets=max_targets,
                train_max_targets=0,
                target_chunk_size=target_chunk_size,
                include_geometry=include_geometry,
                target_sampling="uniform",
                deduplicate_undirected=deduplicate_undirected,
                loss_weighting=None,
            )
            epoch_result["test_monitor"] = monitored_test
            logger.info(
                "Epoch %03d | test-monitor | %s",
                epoch,
                _format_metrics(monitored_test),
            )
        history.append(epoch_result)
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint")
    model.load_state_dict(best_state["model"])
    head.load_state_dict(best_state["head"])
    test = run_design_epoch(
        records=split.test,
        model=model,
        head=head,
        spec=spec,
        transform=transform,
        device=device,
        loss_name=loss_name,
        optimizer=None,
        max_targets=max_targets,
        train_max_targets=0,
        target_chunk_size=target_chunk_size,
        include_geometry=include_geometry,
        target_sampling="uniform",
        deduplicate_undirected=deduplicate_undirected,
        loss_weighting=None,
    )
    logger.info(
        "Final best epoch %03d | test | %s",
        best_epoch,
        _format_metrics(test),
    )
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "checkpoint_metric": checkpoint_metric,
        "best_val_metric": best_value,
        "test": test,
        "history": history,
        "best_state": best_state,
        "seconds": time.perf_counter() - started,
    }


def make_smoke_subgraph(
    sample: NativeGraphSample,
    *,
    max_nodes_per_type: int,
    max_seed_targets: int,
) -> NativeGraphSample:
    """Create a bounded, target-containing subgraph from the real large graph."""

    data = sample.graph
    selected = {
        node_type: torch.zeros(data[node_type].num_nodes, dtype=torch.bool)
        for node_type in data.node_types
    }
    for location in sample.target_locations:
        _, valid, _ = target_tensors(data, location)
        indices = _limit_indices(valid, max_seed_targets)
        if location.kind == "node":
            selected[location.store_key][indices] = True
        else:
            source, _, target = location.store_key
            endpoints = data[location.store_key].edge_index[:, indices]
            selected[source][endpoints[0]] = True
            selected[target][endpoints[1]] = True

    # One bounded expansion over core message relations supplies context.
    for edge_type in sample.message_edge_types:
        if edge_type[1].startswith("rev_"):
            continue
        edge_index = data[edge_type].edge_index
        incident = (
            selected[edge_type[0]][edge_index[0]]
            | selected[edge_type[2]][edge_index[1]]
        )
        edge_ids = _limit_indices(incident, max_nodes_per_type * 2)
        selected[edge_type[0]][edge_index[0, edge_ids]] = True
        selected[edge_type[2]][edge_index[1, edge_ids]] = True

    subsets = {}
    for node_type, mask in selected.items():
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if not indices.numel():
            indices = torch.arange(min(8, data[node_type].num_nodes))
        if indices.numel() > max_nodes_per_type:
            positions = (
                torch.linspace(0, indices.numel() - 1, max_nodes_per_type)
                .round()
                .long()
            )
            indices = indices[positions]
        subsets[node_type] = indices

    # PyG HeteroData.subgraph assumes every attribute classified as a node/edge
    # attribute supports tensor indexing.  The production graphs intentionally
    # retain Python name lists (inst_name, pin_name, source_net_id, ...), which
    # are useful for audit but not model input.  Remove only those per-entity
    # non-tensor lists from this temporary smoke graph.
    for node_type in data.node_types:
        store = data[node_type]
        for key, value in list(store.items()):
            if torch.is_tensor(value) or isinstance(value, (str, bytes)):
                continue
            try:
                per_entity = len(value) == store.num_nodes
            except TypeError:
                per_entity = False
            if per_entity:
                del store[key]
    for edge_type in data.edge_types:
        store = data[edge_type]
        edge_count = store.edge_index.size(1)
        for key, value in list(store.items()):
            if torch.is_tensor(value) or isinstance(value, (str, bytes)):
                continue
            try:
                per_entity = len(value) == edge_count
            except TypeError:
                per_entity = False
            if per_entity:
                del store[key]
    sample.graph = data.subgraph(subsets)
    return sample


def smoke_train_one_graph(
    *,
    sample,
    model,
    head,
    spec,
    device,
    epochs,
    lr,
    weight_decay,
    loss_name,
    max_targets,
    seed,
    logger,
) -> dict:
    """Connectivity test only: row split inside one design is not a paper split."""

    generator = torch.Generator().manual_seed(seed)
    split_indices: dict[str, dict[TargetLocation, torch.Tensor]] = {
        "train": {},
        "val": {},
        "test": {},
    }
    training_chunks = []
    for location in sample.target_locations:
        values, valid, weights = target_tensors(sample.graph, location)
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        n_train = max(1, int(indices.numel() * 0.6))
        n_val = max(1, int(indices.numel() * 0.2))
        parts = {
            "train": indices[:n_train],
            "val": indices[n_train : n_train + n_val],
            "test": indices[n_train + n_val :],
        }
        for name, part in parts.items():
            split_indices[name][location] = part
        training_chunks.append((values[parts["train"]], weights[parts["train"]]))
    transform, _, _ = _fit_weighted_statistics(training_chunks, spec.target_transform)
    sample.to(device)
    with torch.no_grad():
        model(sample.graph)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(sample.graph)
        loss, _ = _sample_predictions(
            sample,
            embeddings,
            head,
            transform,
            max_targets=max_targets,
            loss_name=loss_name,
            allowed_indices=split_indices["train"],
        )
        loss.backward()
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in list(model.parameters()) + list(head.parameters())
        )
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(head.parameters()), 1.0
        )
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.detach()),
                "finite_gradients": finite_gradients,
            }
        )
        logger.info(
            "Smoke epoch %d | loss=%.6f | finite_gradients=%s",
            epoch,
            float(loss.detach()),
            finite_gradients,
        )

    evaluations = {}
    model.eval()
    head.eval()
    with torch.no_grad():
        embeddings = model(sample.graph)
        for name in ("train", "val", "test"):
            loss, outputs = _sample_predictions(
                sample,
                embeddings,
                head,
                transform,
                max_targets=max_targets,
                loss_name=loss_name,
                allowed_indices=split_indices[name],
            )
            accumulator = WeightedRegressionAccumulator()
            for prediction, target, weight in outputs:
                accumulator.update(loss, prediction, target, weight)
            evaluations[name] = accumulator.metrics()
    return {
        "warning": "Single-design row split: connectivity test only, not a valid experiment",
        "target_transform": vars(transform),
        "history": history,
        "evaluations": evaluations,
        "best_state": {"model": model.state_dict(), "head": head.state_dict()},
    }
