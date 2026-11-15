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


# 3. 官方 PyG 同构化；保留类型标识和独立监督索引。
@dataclass(frozen=True)
class HomogeneousSchema:
    view_type: str
    node_type_names: tuple[str, ...]
    edge_type_names: tuple[EdgeType, ...]
    node_widths: tuple[int, ...]
    edge_widths: tuple[int, ...]
    max_node_width: int
    max_edge_width: int
    global_width: int

    def as_dict(self) -> dict:
        return {
            "view_type": self.view_type,
            "node_type_names": list(self.node_type_names),
            "edge_type_names": ["|".join(value) for value in self.edge_type_names],
            "node_widths": list(self.node_widths),
            "edge_widths": list(self.edge_widths),
            "max_node_width": self.max_node_width,
            "max_edge_width": self.max_edge_width,
            "global_width": self.global_width,
        }


def _padded(value: torch.Tensor, width: int) -> torch.Tensor:
    value = value.float()
    if value.size(1) == width:
        return value
    padding = value.new_full((value.size(0), width - value.size(1)), float("nan"))
    return torch.cat((value, padding), dim=1)


def schema_from_sample(sample) -> HomogeneousSchema:
    data = sample.graph
    node_names = tuple(data.node_types)
    edge_names = tuple(sample.message_edge_types)
    node_widths = tuple(int(data[name].x.size(1)) for name in node_names)
    edge_widths = tuple(
        int(data[edge_type].edge_attr.size(1))
        if hasattr(data[edge_type], "edge_attr")
        else 0
        for edge_type in edge_names
    )
    global_features = data._global_store.get("global_features")
    return HomogeneousSchema(
        view_type=sample.view_type,
        node_type_names=node_names,
        edge_type_names=edge_names,
        node_widths=node_widths,
        edge_widths=edge_widths,
        max_node_width=max(node_widths),
        max_edge_width=max(1, max(edge_widths, default=0)),
        global_width=int(global_features.numel()) if global_features is not None else 0,
    )


def convert_sample_to_homogeneous(sample, schema: HomogeneousSchema | None = None):
    """Return one PyG ``Data`` graph plus native supervision tensors.

    Only declared message relations enter ``Data.edge_index``.  Timing, Cc,
    and Reff query relations are mapped to ``target_edge_index`` and never
    become message-passing edges merely because they carry supervision.
    """

    data = sample.graph
    schema = schema or schema_from_sample(sample)
    if tuple(data.node_types) != schema.node_type_names:
        raise ValueError("Node type order/schema changed across input graphs")
    if tuple(sample.message_edge_types) != schema.edge_type_names:
        raise ValueError("Message relation order/schema changed across input graphs")

    staged = HeteroData()
    offsets: dict[str, int] = {}
    total = 0
    for type_id, node_type in enumerate(schema.node_type_names):
        store = data[node_type]
        actual = int(store.x.size(1))
        if actual != schema.node_widths[type_id]:
            raise ValueError(
                f"{node_type}.x width changed: {actual} != {schema.node_widths[type_id]}"
            )
        offsets[node_type] = total
        total += int(store.num_nodes)
        staged[node_type].x = _padded(store.x, schema.max_node_width)

    for relation_id, edge_type in enumerate(schema.edge_type_names):
        source = data[edge_type]
        staged[edge_type].edge_index = source.edge_index
        if hasattr(source, "edge_attr"):
            edge_attr = source.edge_attr
        else:
            edge_attr = torch.empty((source.edge_index.size(1), 0))
        actual = int(edge_attr.size(1))
        if actual != schema.edge_widths[relation_id]:
            raise ValueError(
                f"{edge_type}.edge_attr width changed: "
                f"{actual} != {schema.edge_widths[relation_id]}"
            )
        staged[edge_type].edge_attr = _padded(edge_attr, schema.max_edge_width)

    # Official PyG heterogeneous-to-homogeneous conversion.  The generated
    # node_type and edge_type tensors retain native type/relation identity.
    homogeneous = staged.to_homogeneous(
        node_attrs=["x"],
        edge_attrs=["edge_attr"],
        add_node_type=True,
        add_edge_type=True,
    )

    node_indices = []
    edge_indices = []
    kind_codes = []
    targets = []
    weights = []
    target_kind = None
    target_schemas = []
    for location in sample.target_locations:
        if target_kind is None:
            target_kind = location.kind
        elif target_kind != location.kind:
            target_kind = "mixed"
        values, valid, entity_weight = target_tensors(data, location)
        local = torch.nonzero(valid, as_tuple=False).flatten()
        if not local.numel():
            continue
        targets.append(values[local])
        weights.append(entity_weight[local])
        if location.schema not in target_schemas:
            target_schemas.append(location.schema)
        if location.kind == "node":
            mapped = local + offsets[str(location.store_key)]
            node_indices.append(mapped)
            edge_indices.append(torch.full((2, local.numel()), -1, dtype=torch.long))
            kind_codes.append(torch.zeros(local.numel(), dtype=torch.long))
        else:
            source_type, _, target_type = location.store_key
            endpoints = data[location.store_key].edge_index[:, local].clone()
            endpoints[0] += offsets[source_type]
            endpoints[1] += offsets[target_type]
            edge_indices.append(endpoints)
            node_indices.append(torch.full((local.numel(),), -1, dtype=torch.long))
            kind_codes.append(torch.ones(local.numel(), dtype=torch.long))

    if not targets or target_kind is None:
        raise ValueError(f"No valid target exists in {sample.path}")
    homogeneous.target_kind = target_kind
    homogeneous.target_y = torch.cat(targets).float()
    homogeneous.target_weight = torch.cat(weights).float()
    mapped_nodes = torch.cat(node_indices).long()
    mapped_edges = torch.cat(edge_indices, dim=1).long()
    homogeneous.target_kind_code = torch.cat(kind_codes).long()
    if target_kind == "node":
        homogeneous.target_index = mapped_nodes
    elif target_kind == "edge":
        homogeneous.target_edge_index = mapped_edges
    else:
        # C-view Pin elimination moves Pin slack onto incidence edges while IO
        # slack remains a node target.  Preserve both forms in one task without
        # inventing mapping/query files or converting one supervision type into
        # the other.  -1 is only padding and is never indexed by predict().
        homogeneous.target_node_index = mapped_nodes
        homogeneous.target_edge_index = mapped_edges

    global_features = data._global_store.get("global_features")
    if global_features is not None:
        homogeneous.global_features = global_features.reshape(1, -1).float()
    homogeneous.view_type = sample.view_type
    homogeneous.prediction_stage = sample.stage
    homogeneous.design_name = str(
        data._global_store.get("design_name") or sample.path.parent.name
    )
    homogeneous.source_path = str(sample.path)
    homogeneous.target_schemas = target_schemas
    homogeneous.conversion_method = "torch_geometric.data.HeteroData.to_homogeneous"
    return homogeneous


# 4. 设计族划分与训练集标签变换。
@dataclass
class DatasetSplit:
    train: list[NativeRecord]
    val: list[NativeRecord]
    test: list[NativeRecord]

    def families(self) -> dict[str, list[str]]:
        return {
            key: sorted({record.design_family for record in records})
            for key, records in (
                ("train", self.train),
                ("val", self.val),
                ("test", self.test),
            )
        }

    def validate(self) -> None:
        if not self.train or not self.val or not self.test:
            raise ValueError("Train, validation and test must all be non-empty")
        families = {key: set(value) for key, value in self.families().items()}
        ids = {
            "train": {record.sample_id for record in self.train},
            "val": {record.sample_id for record in self.val},
            "test": {record.sample_id for record in self.test},
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if families[left] & families[right] or ids[left] & ids[right]:
                raise ValueError(f"Dataset leakage between {left} and {right}")

    def summary(self) -> dict:
        families = self.families()
        return {
            key: {
                "record_count": len(records),
                "family_count": len(families[key]),
                "families": families[key],
                "sample_ids": [record.sample_id for record in records],
            }
            for key, records in (
                ("train", self.train),
                ("val", self.val),
                ("test", self.test),
            )
        }


def split_by_family(records: list[NativeRecord], seed: int) -> DatasetSplit:
    families = sorted({record.design_family for record in records})
    if len(families) < 3:
        raise ValueError("Formal training requires at least three design families")
    random.Random(seed).shuffle(families)
    val_count = max(1, round(len(families) * 0.1))
    test_count = max(1, round(len(families) * 0.1))
    val = set(families[:val_count])
    test = set(families[val_count : val_count + test_count])
    train = set(families) - val - test
    result = DatasetSplit(
        [r for r in records if r.design_family in train],
        [r for r in records if r.design_family in val],
        [r for r in records if r.design_family in test],
    )
    result.validate()
    return result


@dataclass(frozen=True)
class TargetTransform:
    kind: str
    mean: float
    std: float
    epsilon: float = 1.0e-12

    def base(self, value: torch.Tensor) -> torch.Tensor:
        if self.kind == "standard":
            return value
        if bool((value < 0).any()):
            raise ValueError(f"{self.kind} received a negative target")
        if self.kind == "log1p_standard":
            return torch.log1p(value)
        if self.kind == "log10_standard":
            return torch.log10(value + self.epsilon)
        raise ValueError(f"Unknown target transform {self.kind!r}")

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        return (self.base(value) - self.mean) / self.std

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        base = value * self.std + self.mean
        if self.kind == "standard":
            return base
        if self.kind == "log1p_standard":
            return torch.expm1(base)
        if self.kind == "log10_standard":
            return torch.pow(10.0, base) - self.epsilon
        raise ValueError(f"Unknown target transform {self.kind!r}")


def fit_transform_chunks(chunks, kind: str) -> TargetTransform:
    provisional = TargetTransform(kind, 0.0, 1.0)
    weight_sum = total = total_sq = 0.0
    for values, weights in chunks:
        base = provisional.base(values.double())
        weights = weights.double()
        weight_sum += float(weights.sum())
        total += float((base * weights).sum())
        total_sq += float((base.square() * weights).sum())
    if weight_sum <= 0:
        raise ValueError("No valid training target")
    mean = total / weight_sum
    variance = max(total_sq / weight_sum - mean * mean, 1.0e-16)
    return TargetTransform(kind, mean, math.sqrt(variance))


def fit_transform_records(records, spec, include_geometry: bool) -> TargetTransform:
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
    return fit_transform_chunks(chunks, spec.target_transform)


def record_target_count(record, spec, include_geometry: bool) -> int:
    """Count valid supervision rows without constructing a homogeneous graph."""

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
    split: DatasetSplit, spec, include_geometry: bool
) -> tuple[DatasetSplit, dict[str, list[str]]]:
    """Remove only graphs with no valid rows after the family split.

    The family assignment is computed from the complete 300-design catalog
    first, so task-specific missing supervision cannot move a family between
    train, validation, and test.  This matters for F-view slack: designs with
    no surviving IO-Pin target are valid F graphs, but cannot contribute a
    supervised loss.
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
            if record_target_count(record, spec, include_geometry) > 0:
                kept[name].append(record)
            else:
                skipped[name].append(record.sample_id)
    filtered = DatasetSplit(kept["train"], kept["val"], kept["test"])
    filtered.validate()
    return filtered, skipped
