"""数据入口：任务标签契约、图读取、监督位置与有效掩码。"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable

import torch
from torch_geometric.data import HeteroData

# 1. 任务定义：物理单位、标签列名、目标变换。
VIEWS = ("B", "C", "D", "E", "F")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    unit: str
    target_transform: str
    schemas_by_view: dict[str, tuple[str, ...]]
    description: str

    def schemas(self, view_type: str) -> tuple[str, ...]:
        return self.schemas_by_view.get(view_type.upper(), ())

    def available_in(self, view_type: str) -> bool:
        return bool(self.schemas(view_type))


def _views(*names: str, schema: str) -> dict[str, tuple[str, ...]]:
    return {name: (schema,) for name in names}


TASK_REGISTRY: dict[str, TaskSpec] = {
    "wirelength": TaskSpec(
        "wirelength",
        "um",
        "log1p_standard",
        {
            **_views("B", "C", schema="routed_wirelength_um"),
            **_views("D", "E", "F", schema="net::routed_wirelength_um"),
        },
        "Route DEF wirelength: Net nodes in B/C, weighted Net-derived edges in D/E/F.",
    ),
    "cg": TaskSpec(
        "cg",
        "pF",
        "log10_standard",
        {
            **_views("B", "C", schema="ground_cap_pF"),
            **_views("D", "E", "F", schema="net::ground_cap_pF"),
        },
        "SPEF ground capacitance: Net nodes in B/C, weighted Net-derived edges in D/E/F.",
    ),
    "congestion": TaskSpec(
        "congestion",
        "dimensionless",
        "log1p_standard",
        _views("B", "C", "D", "F", schema="cell_congestion"),
        "Route congestion on Gate nodes; unavailable in Pin-only E view.",
    ),
    "irdrop": TaskSpec(
        "irdrop",
        "mV",
        "log1p_standard",
        {
            **_views("B", "C", "D", "F", schema="ir_drop_mV"),
            "E": ("gate::ir_drop_mV",),
        },
        "IR drop on Gate nodes, or weighted cell-arc edges in E view.",
    ),
    "setup_slack": TaskSpec(
        "setup_slack",
        "ns",
        "standard",
        _views(*VIEWS, schema="setup_slack_ns"),
        "OpenSTA setup slack wherever Pin/IO-Pin supervision survives the view.",
    ),
    "hold_slack": TaskSpec(
        "hold_slack",
        "ns",
        "standard",
        _views(*VIEWS, schema="hold_slack_ns"),
        "OpenSTA hold slack wherever Pin/IO-Pin supervision survives the view.",
    ),
    "setup_delay": TaskSpec(
        "setup_delay",
        "ns",
        "standard",
        {
            "B": ("setup_delay_ns",),
            "D": ("timing::setup_delay_ns",),
            "E": ("timing::setup_delay_ns",),
        },
        "Timing-path setup delay; absent when the view cannot represent Pin-Pin timing.",
    ),
    "hold_delay": TaskSpec(
        "hold_delay",
        "ns",
        "standard",
        {
            "B": ("hold_delay_ns",),
            "D": ("timing::hold_delay_ns",),
            "E": ("timing::hold_delay_ns",),
        },
        "Timing-path hold delay; absent when the view cannot represent Pin-Pin timing.",
    ),
    "cc": TaskSpec(
        "cc",
        "pF",
        "log10_standard",
        _views("B", "C", schema="coupling_cap_pF"),
        "SPEF Net-Net coupling capacitance; only views retaining Net nodes support it.",
    ),
    "reff": TaskSpec(
        "reff",
        "ohm",
        "log10_standard",
        {
            "B": ("effective_resistance_ohm",),
            "D": ("rc::effective_resistance_ohm",),
            "E": ("rc::effective_resistance_ohm",),
        },
        "SPEF effective resistance; only views retaining Pin-level endpoints support it.",
    ),
}


def get_task_spec(name: str) -> TaskSpec:
    normalized = name.strip().lower()
    if normalized not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task {name!r}; supported: {', '.join(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[normalized]


# 2. 原生图读取：传播边与监督边分开处理。
STAGES = ("floorplan", "placement", "cts", "route")
VIEWS = ("B", "C", "D", "E", "F")
EdgeType = tuple[str, str, str]


def relation_to_string(edge_type: EdgeType) -> str:
    return "|".join(edge_type)


def relation_from_string(value: str | Iterable[str]) -> EdgeType:
    parts = tuple(value.split("|")) if isinstance(value, str) else tuple(value)
    if len(parts) != 3:
        raise ValueError(f"Invalid edge type: {value!r}")
    return parts  # type: ignore[return-value]


def graph_view_type(data: HeteroData) -> str:
    value = data._global_store.get("view_type")
    if value is None:
        # B is the only native contract predating the explicit field.
        required = {"gate", "pin", "net", "io_pin"}
        if required <= set(data.node_types):
            return "B"
        raise ValueError("Graph has no view_type and is not recognizable as B view")
    value = str(value).upper()
    if value not in VIEWS:
        raise ValueError(f"Unsupported view_type={value!r}")
    return value


def graph_stage(data: HeteroData) -> str:
    value = data._global_store.get("prediction_stage")
    if value is None:
        value = data._global_store.get("stage")
    value = str(value).lower()
    if value == "routing":
        value = "route"
    if value not in STAGES:
        raise ValueError(f"Unsupported/missing prediction stage: {value!r}")
    return value


def message_edge_types(
    data: HeteroData, *, include_geometry: bool = False
) -> list[EdgeType]:
    raw = data._global_store.get("message_passing_edge_types")
    if not raw:
        raw = data._global_store.get("default_message_passing_edge_types", [])
    result = [relation_from_string(value) for value in raw]
    if include_geometry:
        result.extend(
            relation_from_string(value)
            for value in data._global_store.get(
                "optional_message_passing_edge_types", []
            )
        )
    missing = [edge_type for edge_type in result if edge_type not in data.edge_types]
    if missing:
        raise ValueError(f"Declared message relations missing from graph: {missing}")
    # Preserve order and avoid duplicates.
    return list(dict.fromkeys(result))


@dataclass(frozen=True)
class TargetLocation:
    kind: str
    store_key: Hashable
    schema: str
    column: int


def locate_task_targets(
    data: HeteroData, spec: TaskSpec, view_type: str | None = None
) -> list[TargetLocation]:
    view = view_type or graph_view_type(data)
    allowed = set(spec.schemas(view))
    if not allowed:
        raise ValueError(f"Task {spec.name!r} is unavailable in {view} view")
    locations: list[TargetLocation] = []
    for node_type in data.node_types:
        store = data[node_type]
        for column, schema in enumerate(list(getattr(store, "y_schema", []))):
            if schema in allowed:
                locations.append(TargetLocation("node", node_type, schema, column))
    for edge_type in data.edge_types:
        store = data[edge_type]
        for column, schema in enumerate(list(getattr(store, "edge_y_schema", []))):
            if schema in allowed:
                locations.append(TargetLocation("edge", edge_type, schema, column))
    if not locations:
        raise ValueError(
            f"{view} view declares task {spec.name!r}, but none of {sorted(allowed)} "
            "was found in node y_schema or edge_y_schema"
        )
    return locations


def target_tensors(
    data: HeteroData, location: TargetLocation
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    store = data[location.store_key]
    if location.kind == "node":
        values = store.y[:, location.column].float()
        valid = store.y_valid_mask[:, location.column].bool()
    else:
        values = store.edge_y[:, location.column].float()
        mask = getattr(store, "edge_y_mask", None)
        if mask is None:
            mask = store.edge_y_valid_mask
        valid = mask[:, location.column].bool()
    valid = valid & torch.isfinite(values)

    # Only replicated canonical entities need de-duplication weights.  Exact
    # timing/RC pair labels sharing the D/E edge table remain one observation.
    use_entity_weight = (
        location.kind == "edge"
        and (
            location.schema.startswith("net::") or location.schema.startswith("gate::")
        )
        and hasattr(store, "entity_weight")
    )
    if use_entity_weight:
        weights = store.entity_weight.float().reshape(-1)
    elif location.kind == "edge" and location.schema == "coupling_cap_pF":
        # B/C store symmetric Net-Net Cc supervision in both directions.  Give
        # every unordered physical pair a total weight of one, while retaining
        # both orientations for the endpoint-pair predictor.  The common fast
        # path handles adjacent (u,v)/(v,u) rows without a costly global sort;
        # the fallback also covers non-adjacent or repeated unordered pairs.
        edge_index = store.edge_index
        edge_count = edge_index.size(1)
        adjacent_reciprocals = edge_count % 2 == 0 and torch.equal(
            edge_index[:, 0::2], edge_index[:, 1::2].flip(0)
        )
        if adjacent_reciprocals:
            weights = torch.full_like(values, 0.5)
        else:
            source, target = edge_index
            low = torch.minimum(source, target)
            high = torch.maximum(source, target)
            pair_key = low * int(data[location.store_key[0]].num_nodes) + high
            _, inverse, counts = torch.unique(
                pair_key, sorted=False, return_inverse=True, return_counts=True
            )
            weights = counts[inverse].to(values.dtype).reciprocal()
    else:
        weights = torch.ones_like(values)
    valid &= torch.isfinite(weights) & (weights > 0)
    return values, valid, weights


def _add_reverse_message_relations(
    data: HeteroData, forward: list[EdgeType]
) -> list[EdgeType]:
    result = list(forward)
    for source_type, relation, target_type in forward:
        edge_type = (source_type, relation, target_type)
        reverse = (target_type, f"rev_{relation}", source_type)
        if reverse in data.edge_types:
            result.append(reverse)
            continue
        source = data[edge_type]
        target = data[reverse]
        target.edge_index = source.edge_index.flip(0)
        if hasattr(source, "edge_attr"):
            target.edge_attr = source.edge_attr
            target.edge_schema = list(getattr(source, "edge_schema", []))
        target.is_runtime_reverse = True
        result.append(reverse)
    return result


@dataclass
class NativeGraphSample:
    graph: HeteroData
    path: Path
    view_type: str
    stage: str
    message_edge_types: tuple[EdgeType, ...]
    target_locations: tuple[TargetLocation, ...]

    def to(self, device: torch.device | str) -> "NativeGraphSample":
        self.graph = self.graph.to(device)
        return self


def load_native_graph(
    path: str | Path,
    *,
    task: TaskSpec,
    expected_view: str | None = None,
    expected_stage: str | None = None,
    include_geometry: bool = False,
    add_reverse_edges: bool = True,
    prune_unused_relations: bool = True,
) -> NativeGraphSample:
    path = Path(path).expanduser().resolve()
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, HeteroData):
        raise TypeError(f"{path} does not contain torch_geometric.data.HeteroData")
    view = graph_view_type(data)
    stage = graph_stage(data)
    if expected_view and view != expected_view.upper():
        raise ValueError(f"Expected {expected_view} view but {path} stores {view}")
    normalized_stage = "route" if expected_stage == "routing" else expected_stage
    if normalized_stage and stage != normalized_stage:
        raise ValueError(f"Expected {normalized_stage} stage but {path} stores {stage}")

    forward_relations = message_edge_types(data, include_geometry=include_geometry)
    locations = locate_task_targets(data, task, view)
    target_edges = {
        location.store_key for location in locations if location.kind == "edge"
    }
    if prune_unused_relations:
        keep = set(forward_relations) | target_edges
        for edge_type in list(data.edge_types):
            if edge_type not in keep:
                del data[edge_type]
    runtime_relations = (
        _add_reverse_message_relations(data, forward_relations)
        if add_reverse_edges
        else forward_relations
    )
    return NativeGraphSample(
        graph=data,
        path=path,
        view_type=view,
        stage=stage,
        message_edge_types=tuple(runtime_relations),
        target_locations=tuple(locations),
    )


@dataclass(frozen=True)
class NativeRecord:
    graph_path: Path
    view_type: str
    stage: str
    design_name: str
    design_family: str

    @property
    def sample_id(self) -> str:
        return f"{self.design_name}/{self.stage}/{self.view_type}"


def record_from_path(path: str | Path) -> NativeRecord:
    path = Path(path).expanduser().resolve()
    data = torch.load(path, map_location="cpu", weights_only=False)
    view = graph_view_type(data)
    stage = graph_stage(data)
    design_name = str(data._global_store.get("design_name") or path.parent.name)
    family = design_name.split("__", 1)[0]
    del data
    return NativeRecord(path, view, stage, design_name, family)


def records_from_manifest(
    manifest: Path, *, view_type: str, stage: str
) -> list[NativeRecord]:
    """统一多视图目录：以显式 design/version 字段划分，不能从视图目录猜设计族。"""
    root = manifest.parent.resolve()
    view_type = view_type.upper()
    records = []
    seen = set()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"design", "version", "stage", view_type}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest missing columns: {required}: {manifest}")
        for row in reader:
            if row["stage"] != stage:
                continue
            family, version = row["design"], row["version"]
            if not family or not version:
                raise ValueError(f"Empty design/version in {manifest}")
            path = (root / row[view_type]).resolve()
            if root not in path.parents or not path.is_file():
                raise ValueError(f"Missing graph or path outside dataset: {path}")
            key = (family, version)
            if key in seen:
                raise ValueError(f"Duplicate design/version in {manifest}: {key}")
            seen.add(key)
            records.append(
                NativeRecord(path, view_type, stage, f"{family}__{version}", family)
            )
    return sorted(records, key=lambda record: record.sample_id)


def discover_records(
    roots: Iterable[str | Path], *, view_type: str, stage: str
) -> list[NativeRecord]:
    records: list[NativeRecord] = []
    candidates: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        manifest = root / "manifest.csv" if root.is_dir() else root
        if manifest.name == "manifest.csv" and manifest.is_file():
            records.extend(
                records_from_manifest(manifest, view_type=view_type, stage=stage)
            )
            continue
        if root.is_file():
            record = record_from_path(root)
            if record.view_type == view_type.upper() and record.stage == stage:
                records.append(record)
        elif root.is_dir():
            candidates.update(root.rglob("heterograph.pt"))
    for path in sorted(candidates):
        # Current batch layout:
        #   <family>/<version>/<stage>/heterograph.pt
        # Legacy converted datasets may contain an extra ``stages`` directory:
        #   <family>/<version>/stages/<stage>/heterograph.pt
        # Derive the catalog record from either path and defer strict graph
        # metadata validation to load_native_graph.  This avoids deserializing
        # all four stages (1200 large PTs for the B-v2 catalog) merely to select
        # one 300-graph prediction stage.
        if path.parent.name != stage:
            continue
        if path.parent.parent.name == "stages":
            version_dir = path.parent.parent.parent
            family_dir = version_dir.parent
        else:
            version_dir = path.parent.parent
            family_dir = version_dir.parent
        if family_dir.name and version_dir.name:
            records.append(
                NativeRecord(
                    path,
                    view_type.upper(),
                    stage,
                    f"{family_dir.name}__{version_dir.name}",
                    family_dir.name,
                )
            )
        else:
            record = record_from_path(path)
            if record.view_type == view_type.upper() and record.stage == stage:
                records.append(record)
    return records


def graph_structure_summary(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    data = torch.load(path, map_location="cpu", weights_only=False)
    view = graph_view_type(data)
    summary = {
        "path": str(path),
        "view": view,
        "stage": graph_stage(data),
        "nodes": {},
        "edges": {},
        "message_edge_types": [
            relation_to_string(value) for value in message_edge_types(data)
        ],
        "optional_message_edge_types": list(
            data._global_store.get("optional_message_passing_edge_types", [])
        ),
    }
    for node_type in data.node_types:
        store = data[node_type]
        summary["nodes"][node_type] = {
            "count": int(store.num_nodes),
            "x_dim": int(store.x.size(1)),
            "y_schema": list(getattr(store, "y_schema", [])),
        }
    for edge_type in data.edge_types:
        store = data[edge_type]
        summary["edges"][relation_to_string(edge_type)] = {
            "count": int(store.edge_index.size(1)),
            "edge_attr_dim": int(store.edge_attr.size(1))
            if hasattr(store, "edge_attr")
            else 0,
            "edge_y_schema": list(getattr(store, "edge_y_schema", [])),
            "has_entity_weight": hasattr(store, "entity_weight"),
        }
    return summary


# 3. 先按设计族划分，再仅用训练标签拟合变换。
@dataclass
class DatasetSplit:
    train: list[NativeRecord]
    val: list[NativeRecord]
    test: list[NativeRecord]

    def ids(self) -> dict[str, list[str]]:
        return {
            "train": [record.sample_id for record in self.train],
            "val": [record.sample_id for record in self.val],
            "test": [record.sample_id for record in self.test],
        }

    def families(self) -> dict[str, list[str]]:
        return {
            name: sorted({record.design_family for record in records})
            for name, records in (
                ("train", self.train),
                ("val", self.val),
                ("test", self.test),
            )
        }

    def validate(self) -> None:
        """Enforce a reproducible family-level split with an untouched test set."""

        if not self.train or not self.val or not self.test:
            raise ValueError("Train, validation, and test splits must all be non-empty")
        ids = {name: set(values) for name, values in self.ids().items()}
        families = {name: set(values) for name, values in self.families().items()}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            duplicate_ids = ids[left] & ids[right]
            duplicate_families = families[left] & families[right]
            if duplicate_ids or duplicate_families:
                raise ValueError(
                    f"Dataset split leakage between {left} and {right}: "
                    f"sample_ids={sorted(duplicate_ids)}, "
                    f"families={sorted(duplicate_families)}"
                )

    def summary(self) -> dict[str, dict[str, int | list[str]]]:
        families = self.families()
        return {
            name: {
                "record_count": len(records),
                "family_count": len(families[name]),
                "families": families[name],
            }
            for name, records in (
                ("train", self.train),
                ("val", self.val),
                ("test", self.test),
            )
        }


def split_records_by_family(records: list[NativeRecord], seed: int) -> DatasetSplit:
    families = sorted({record.design_family for record in records})
    if len(families) < 3:
        raise ValueError(
            "Formal training needs at least three independent design families; "
            "use --smoke-test only for the temporary bp_multi_top graph"
        )
    random.Random(seed).shuffle(families)
    val_count = max(1, round(len(families) * 0.1))
    test_count = max(1, round(len(families) * 0.1))
    val = set(families[:val_count])
    test = set(families[val_count : val_count + test_count])
    train = set(families) - val - test
    split = DatasetSplit(
        [r for r in records if r.design_family in train],
        [r for r in records if r.design_family in val],
        [r for r in records if r.design_family in test],
    )
    split.validate()
    return split


def record_target_count(
    record: NativeRecord,
    spec: TaskSpec,
    *,
    include_geometry: bool,
) -> int:
    """Count valid supervision rows without moving or converting the graph."""

    sample = load_native_graph(
        record.graph_path,
        task=spec,
        expected_view=record.view_type,
        expected_stage=record.stage,
        include_geometry=include_geometry,
        add_reverse_edges=False,
    )
    count = 0
    for location in sample.target_locations:
        _, valid, _ = target_tensors(sample.graph, location)
        count += int(valid.sum())
    del sample
    return count


def filter_split_without_targets(
    split: DatasetSplit,
    spec: TaskSpec,
    *,
    include_geometry: bool,
) -> tuple[DatasetSplit, dict[str, list[str]]]:
    """Remove zero-supervision graphs after fixing the family assignment.

    Reduced views can legitimately eliminate every target of a task for an
    individual design (for example F-view Slack when no labelled IO-Pin
    survives).  The complete catalog is split first so missing supervision
    cannot move a family between train, validation, and test.
    """

    kept: dict[str, list[NativeRecord]] = {}
    skipped: dict[str, list[str]] = {}
    for name, records in (
        ("train", split.train),
        ("val", split.val),
        ("test", split.test),
    ):
        kept[name] = []
        skipped[name] = []
        for record in records:
            if record_target_count(record, spec, include_geometry=include_geometry) > 0:
                kept[name].append(record)
            else:
                skipped[name].append(record.sample_id)
    filtered = DatasetSplit(kept["train"], kept["val"], kept["test"])
    filtered.validate()
    return filtered, skipped


@dataclass(frozen=True)
class FittedTargetTransform:
    kind: str
    mean: float
    std: float
    epsilon: float = 1.0e-12

    def _base(self, values: torch.Tensor) -> torch.Tensor:
        if self.kind == "standard":
            return values
        if bool((values < 0).any()):
            raise ValueError(f"{self.kind} received a negative physical target")
        if self.kind == "log1p_standard":
            return torch.log1p(values)
        if self.kind == "log10_standard":
            return torch.log10(values + self.epsilon)
        raise ValueError(f"Unknown target transform {self.kind!r}")

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (self._base(values) - self.mean) / self.std

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        base = values * self.std + self.mean
        if self.kind == "standard":
            return base
        if self.kind == "log1p_standard":
            return torch.expm1(base)
        if self.kind == "log10_standard":
            return torch.pow(10.0, base) - self.epsilon
        raise ValueError(f"Unknown target transform {self.kind!r}")


def _fit_weighted_statistics(
    chunks: list[tuple[torch.Tensor, torch.Tensor]], kind: str
) -> tuple[FittedTargetTransform, float, int]:
    provisional = FittedTargetTransform(kind, 0.0, 1.0)
    weight_sum = total = total_sq = 0.0
    row_count = 0
    for values, weights in chunks:
        base = provisional._base(values.double())
        weights = weights.double()
        weight_sum += float(weights.sum())
        total += float((base * weights).sum())
        total_sq += float((base.square() * weights).sum())
        row_count += int(values.numel())
    if weight_sum <= 0:
        raise ValueError("No valid training target is available")
    mean = total / weight_sum
    variance = max(total_sq / weight_sum - mean * mean, 1.0e-16)
    return FittedTargetTransform(kind, mean, math.sqrt(variance)), weight_sum, row_count


def fit_transform_from_records(
    records: list[NativeRecord],
    spec: TaskSpec,
    *,
    include_geometry: bool,
) -> tuple[FittedTargetTransform, float, int]:
    chunks = []
    for record in records:
        sample = load_native_graph(
            record.graph_path,
            task=spec,
            expected_view=record.view_type,
            expected_stage=record.stage,
            include_geometry=include_geometry,
            add_reverse_edges=False,
        )
        for location in sample.target_locations:
            values, valid, weights = target_tensors(sample.graph, location)
            if bool(valid.any()):
                chunks.append((values[valid], weights[valid]))
        del sample
    return _fit_weighted_statistics(chunks, spec.target_transform)


@dataclass(frozen=True)
class QuantileLossWeighting:
    """Training-only target weights fitted exclusively from train records."""

    thresholds: tuple[float, ...]
    multipliers: tuple[float, ...]

    def weights_for(self, values: torch.Tensor) -> torch.Tensor:
        if len(self.multipliers) != len(self.thresholds) + 1:
            raise ValueError("quantile loss weighting has an invalid shape")
        result = torch.full_like(values, self.multipliers[0])
        for threshold, multiplier in zip(self.thresholds, self.multipliers[1:]):
            result = torch.where(values > threshold, multiplier, result)
        return result


def _limit_indices(mask: torch.Tensor, maximum: int) -> torch.Tensor:
    """确定性等间距取样；0 表示不截断，不消耗随机数状态。"""
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if maximum <= 0 or indices.numel() <= maximum:
        return indices
    positions = torch.linspace(0, indices.numel() - 1, maximum).round().long()
    return indices[positions]


def fit_quantile_loss_weighting(
    records: list[NativeRecord],
    spec: TaskSpec,
    *,
    include_geometry: bool,
    quantiles: tuple[float, ...],
    multipliers: tuple[float, ...],
    sample_per_record: int = 65536,
) -> QuantileLossWeighting:
    """Estimate raw-target quantiles without retaining every training label."""

    if any(not 0.0 < value < 1.0 for value in quantiles):
        raise ValueError("loss-weight quantiles must be strictly between 0 and 1")
    if tuple(sorted(quantiles)) != quantiles:
        raise ValueError("loss-weight quantiles must be strictly increasing")
    if len(multipliers) != len(quantiles) + 1 or any(v <= 0 for v in multipliers):
        raise ValueError("provide one positive multiplier per quantile interval")
    samples = []
    for record in records:
        sample = load_native_graph(
            record.graph_path,
            task=spec,
            expected_view=record.view_type,
            expected_stage=record.stage,
            include_geometry=include_geometry,
            add_reverse_edges=False,
        )
        for location in sample.target_locations:
            values, valid, _ = target_tensors(sample.graph, location)
            indices = _limit_indices(valid, sample_per_record)
            if indices.numel():
                samples.append(values[indices].float())
        del sample
    if not samples:
        raise ValueError("No valid training targets for quantile loss weighting")
    values = torch.cat(samples)
    thresholds = tuple(
        float(value)
        for value in torch.quantile(values, torch.tensor(quantiles)).tolist()
    )
    return QuantileLossWeighting(thresholds, multipliers)
