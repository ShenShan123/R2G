"""Native-view HGT, RGCN and HAN backbones plus node/edge task heads."""

from __future__ import annotations

import torch
from dataset import EdgeType, TargetLocation
from encoder import NativeViewFeatureEncoder
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HANConv, HGTConv, RGCNConv

SUPPORTED_MODELS = ("hgt", "rgcn", "han")
CC_PAIR_GEOMETRY_DIM = 21


def _schema_column(store, name: str) -> int:
    schema = list(getattr(store, "x_schema", []))
    if name not in schema:
        raise ValueError(f"{name!r} is missing from runtime feature schema")
    return schema.index(name)


def _cc_net_geometry(data) -> torch.Tensor:
    """Aggregate causal Net geometry from existing Pin/IO-Pin coordinates."""

    net_store = data["net"]
    cached = getattr(net_store, "cc_runtime_geometry", None)
    if cached is not None:
        return cached
    reference = net_store.x
    net_count = int(net_store.num_nodes)
    count = reference.new_zeros(net_count)
    coordinate_sum = reference.new_zeros((net_count, 2))
    coordinate_squared_sum = reference.new_zeros((net_count, 2))
    coordinate_min = reference.new_full((net_count, 2), float("inf"))
    coordinate_max = reference.new_full((net_count, 2), float("-inf"))

    for node_type in ("pin", "io_pin"):
        relation = (node_type, "connects_to", "net")
        if node_type not in data.node_types or relation not in data.edge_types:
            continue
        store = data[node_type]
        x_column = _schema_column(store, "pin_x_normalized")
        y_column = _schema_column(store, "pin_y_normalized")
        valid_column = _schema_column(store, "pin_position_valid")
        edge_index = data[relation].edge_index
        source = edge_index[0]
        target = edge_index[1]
        coordinates = store.x[source][:, [x_column, y_column]].float()
        valid = (store.x[source, valid_column] > 0.5) & torch.isfinite(coordinates).all(
            dim=1
        )
        if not bool(valid.any()):
            continue
        target = target[valid]
        coordinates = coordinates[valid]
        count.index_add_(0, target, torch.ones_like(target, dtype=count.dtype))
        coordinate_sum.index_add_(0, target, coordinates)
        coordinate_squared_sum.index_add_(0, target, coordinates.square())
        expanded_target = target[:, None].expand(-1, 2)
        coordinate_min.scatter_reduce_(
            0, expanded_target, coordinates, reduce="amin", include_self=True
        )
        coordinate_max.scatter_reduce_(
            0, expanded_target, coordinates, reduce="amax", include_self=True
        )

    valid_net = count > 0
    denominator = count.clamp_min(1.0)[:, None]
    mean = coordinate_sum / denominator
    variance = (coordinate_squared_sum / denominator - mean.square()).clamp_min(0.0)
    std = variance.sqrt()
    coordinate_min = torch.where(valid_net[:, None], coordinate_min, 0.0)
    coordinate_max = torch.where(valid_net[:, None], coordinate_max, 0.0)
    geometry = torch.cat(
        (
            mean,
            coordinate_min,
            coordinate_max,
            std,
            torch.log1p(count)[:, None],
            valid_net.to(reference.dtype)[:, None],
        ),
        dim=1,
    )
    net_store.cc_runtime_geometry = geometry
    return geometry


def _cc_pair_geometry(data, edge_index: torch.Tensor) -> torch.Tensor:
    """Build exchange-invariant relative geometry for selected Net pairs."""

    geometry = _cc_net_geometry(data)
    source = geometry[edge_index[0]]
    target = geometry[edge_index[1]]
    mean_delta = (source[:, :2] - target[:, :2]).abs()
    lower = torch.maximum(source[:, 2:4], target[:, 2:4])
    upper = torch.minimum(source[:, 4:6], target[:, 4:6])
    gap = (lower - upper).clamp_min(0.0)
    overlap = (upper - lower).clamp_min(0.0)
    source_size = (source[:, 4:6] - source[:, 2:4]).clamp_min(0.0)
    target_size = (target[:, 4:6] - target[:, 2:4]).clamp_min(0.0)
    std_sum = source[:, 6:8] + target[:, 6:8]
    std_difference = (source[:, 6:8] - target[:, 6:8]).abs()
    count_sum = source[:, 8:9] + target[:, 8:9]
    count_difference = (source[:, 8:9] - target[:, 8:9]).abs()
    both_valid = source[:, 9:10] * target[:, 9:10]
    pair = torch.cat(
        (
            mean_delta,
            mean_delta.sum(dim=1, keepdim=True),
            mean_delta.square().sum(dim=1, keepdim=True).sqrt(),
            gap,
            gap.square().sum(dim=1, keepdim=True).sqrt(),
            overlap,
            overlap.prod(dim=1, keepdim=True),
            source_size + target_size,
            (source_size - target_size).abs(),
            std_sum,
            std_difference,
            count_sum,
            count_difference,
            both_valid,
        ),
        dim=1,
    )
    if pair.size(1) != CC_PAIR_GEOMETRY_DIM:
        raise RuntimeError(
            f"Cc pair geometry expected {CC_PAIR_GEOMETRY_DIM}D, got {pair.size(1)}D"
        )
    return pair


def _mlp(input_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_head_layers must be >= 1")
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(num_layers - 1):
        layers.extend((nn.Linear(current, hidden_dim), nn.GELU()))
        current = hidden_dim
    layers.append(nn.Linear(current, 1))
    return nn.Sequential(*layers)


class NativeTaskHead(nn.Module):
    """Read native node targets or native endpoint-pair edge targets."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        *,
        edge_pair_mode: str = "directed",
        cc_dynamic_pair_geometry: bool = False,
    ) -> None:
        super().__init__()
        if edge_pair_mode not in {"directed", "symmetric"}:
            raise ValueError("edge_pair_mode must be 'directed' or 'symmetric'")
        self.edge_pair_mode = edge_pair_mode
        self.cc_dynamic_pair_geometry = cc_dynamic_pair_geometry
        self.node_head = _mlp(hidden_dim, hidden_dim, num_layers)
        pair_width = hidden_dim * (3 if edge_pair_mode == "symmetric" else 4)
        if cc_dynamic_pair_geometry:
            pair_width += CC_PAIR_GEOMETRY_DIM
        self.edge_head = _mlp(pair_width, hidden_dim, num_layers)

    def predict(
        self,
        data,
        embeddings: dict[str, torch.Tensor],
        location: TargetLocation,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if location.kind == "node":
            return self.node_head(embeddings[location.store_key][indices]).squeeze(-1)
        source_type, _, target_type = location.store_key
        edge_index = data[location.store_key].edge_index[:, indices]
        source = embeddings[source_type][edge_index[0]]
        target = embeddings[target_type][edge_index[1]]
        if self.edge_pair_mode == "symmetric":
            # RC coupling is an unordered physical relation.  This readout is
            # exactly invariant to swapping (u, v), unlike concatenating the
            # two endpoint embeddings in their stored direction.
            pair = torch.cat(
                (source + target, (source - target).abs(), source * target),
                dim=-1,
            )
        else:
            pair = torch.cat(
                (source, target, source * target, (source - target).abs()),
                dim=-1,
            )
        if self.cc_dynamic_pair_geometry:
            if location.schema != "coupling_cap_pF":
                raise ValueError(
                    "dynamic Cc pair geometry received a non-Cc target location"
                )
            pair = torch.cat((pair, _cc_pair_geometry(data, edge_index)), dim=-1)
        return self.edge_head(pair).squeeze(-1)


class _NativeBackbone(nn.Module):
    def __init__(
        self,
        *,
        view_type: str,
        node_types: tuple[str, ...],
        edge_types: tuple[EdgeType, ...],
        hidden_dim: int,
        feature_transform: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.encoder = NativeViewFeatureEncoder(
            view_type=view_type,
            node_types=node_types,
            message_edge_types=edge_types,
            hidden_dim=hidden_dim,
            feature_transform=feature_transform,
        )
        self.dropout = nn.Dropout(dropout)

    @property
    def metadata(self):
        return self.node_types, self.edge_types

    def edge_index_dict(self, data) -> dict[EdgeType, torch.Tensor]:
        return {edge_type: data[edge_type].edge_index for edge_type in self.edge_types}

    def _residual_update(self, previous, update, norms):
        return {
            node_type: (
                norms[node_type](value + self.dropout(F.gelu(update[node_type])))
                if node_type in update and update[node_type] is not None
                else value
            )
            for node_type, value in previous.items()
        }


class HGT(_NativeBackbone):
    def __init__(self, *, num_layers: int, heads: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.layers = nn.ModuleList(
            HGTConv(self.hidden_dim, self.hidden_dim, self.metadata, heads=heads)
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.ModuleDict(
                {name: nn.LayerNorm(self.hidden_dim) for name in self.node_types}
            )
            for _ in range(num_layers)
        )

    def forward(self, data) -> dict[str, torch.Tensor]:
        values = self.encoder(data)
        edge_index_dict = self.edge_index_dict(data)
        for layer, norms in zip(self.layers, self.norms):
            values = self._residual_update(
                values, layer(values, edge_index_dict), norms
            )
        return values


class HAN(_NativeBackbone):
    def __init__(self, *, num_layers: int, heads: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.layers = nn.ModuleList(
            HANConv(
                self.hidden_dim,
                self.hidden_dim,
                self.metadata,
                heads=heads,
                dropout=self.dropout.p,
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.ModuleDict(
                {name: nn.LayerNorm(self.hidden_dim) for name in self.node_types}
            )
            for _ in range(num_layers)
        )

    def forward(self, data) -> dict[str, torch.Tensor]:
        values = self.encoder(data)
        edge_index_dict = self.edge_index_dict(data)
        for layer, norms in zip(self.layers, self.norms):
            values = self._residual_update(
                values, layer(values, edge_index_dict), norms
            )
        return values


def _homogeneous_layout(data, values, node_types, edge_types):
    offsets: dict[str, int] = {}
    chunks: list[torch.Tensor] = []
    total = 0
    for node_type in node_types:
        offsets[node_type] = total
        chunks.append(values[node_type])
        total += values[node_type].size(0)
    edge_chunks = []
    relation_chunks = []
    for relation_id, edge_type in enumerate(edge_types):
        edge_index = data[edge_type].edge_index
        edge_chunks.append(
            torch.stack(
                (
                    edge_index[0] + offsets[edge_type[0]],
                    edge_index[1] + offsets[edge_type[2]],
                )
            )
        )
        relation_chunks.append(
            torch.full(
                (edge_index.size(1),),
                relation_id,
                dtype=torch.long,
                device=edge_index.device,
            )
        )
    return (
        torch.cat(chunks, dim=0),
        torch.cat(edge_chunks, dim=1),
        torch.cat(relation_chunks),
        offsets,
    )


class RGCN(_NativeBackbone):
    def __init__(self, *, num_layers: int, num_bases: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layers = nn.ModuleList(
            RGCNConv(
                self.hidden_dim,
                self.hidden_dim,
                len(self.edge_types),
                num_bases=min(num_bases, len(self.edge_types)),
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(self.hidden_dim) for _ in range(num_layers)
        )

    def forward(self, data) -> dict[str, torch.Tensor]:
        values = self.encoder(data)
        homogeneous, edge_index, edge_type, offsets = _homogeneous_layout(
            data, values, self.node_types, self.edge_types
        )
        for layer, norm in zip(self.layers, self.norms):
            update = layer(homogeneous, edge_index, edge_type)
            homogeneous = norm(homogeneous + self.dropout(F.gelu(update)))
        return {
            node_type: homogeneous[
                offsets[node_type] : offsets[node_type] + values[node_type].size(0)
            ]
            for node_type in self.node_types
        }


def build_model(
    name: str,
    *,
    view_type: str,
    node_types: tuple[str, ...],
    edge_types: tuple[EdgeType, ...],
    hidden_dim: int = 128,
    num_layers: int = 3,
    dropout: float = 0.1,
    feature_transform: str = "signed_log1p",
    heads: int = 4,
    num_bases: int = 4,
) -> nn.Module:
    if hidden_dim < 1 or num_layers < 1:
        raise ValueError("hidden_dim and num_layers must be positive")
    common = dict(
        view_type=view_type,
        node_types=node_types,
        edge_types=edge_types,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        feature_transform=feature_transform,
    )
    normalized = name.lower()
    if normalized == "hgt":
        return HGT(**common, heads=heads)
    if normalized == "han":
        return HAN(**common, heads=heads)
    if normalized == "rgcn":
        return RGCN(**common, num_bases=num_bases)
    raise ValueError(
        f"Unknown model {name!r}; supported: {', '.join(SUPPORTED_MODELS)}"
    )
