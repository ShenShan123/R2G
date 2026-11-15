"""View-aware feature encoders for native B/C/D/E/F graphs.

Each stage graph is already a causal input snapshot, so this module performs
no stage masking.  Its responsibilities are schema validation, finite numeric
preprocessing, and independent node/edge projections for each native view.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from dataset import EdgeType, relation_to_string
from torch import nn


@dataclass(frozen=True)
class ViewContract:
    node_dims: dict[str, int]
    edge_dims: dict[EdgeType, int]


VIEW_CONTRACTS: dict[str, ViewContract] = {
    "B": ViewContract(
        {"gate": 30, "net": 28, "pin": 37, "io_pin": 30},
        {
            ("gate", "has", "pin"): 2,
            ("pin", "connects_to", "net"): 2,
            ("io_pin", "connects_to", "net"): 2,
            ("gate", "congestion_geom", "gate"): 1,
        },
    ),
    "C": ViewContract(
        {"gate": 30, "net": 28, "io_pin": 30},
        {
            ("gate", "pin_incidence", "net"): 41,
            ("io_pin", "connects_to", "net"): 2,
            ("gate", "congestion_geom", "gate"): 1,
        },
    ),
    "D": ViewContract(
        {"gate": 30, "pin": 37, "io_pin": 30},
        {
            ("gate", "has", "pin"): 2,
            ("pin", "net_connection", "pin"): 32,
            ("pin", "net_connection", "io_pin"): 32,
            ("io_pin", "net_connection", "pin"): 32,
            ("io_pin", "net_connection", "io_pin"): 32,
            ("gate", "congestion_geom", "gate"): 1,
        },
    ),
    "E": ViewContract(
        {"pin": 37, "io_pin": 30},
        {
            ("pin", "cell_arc", "pin"): 34,
            ("pin", "net_connection", "pin"): 32,
            ("pin", "net_connection", "io_pin"): 32,
            ("io_pin", "net_connection", "pin"): 32,
            ("io_pin", "net_connection", "io_pin"): 32,
        },
    ),
    "F": ViewContract(
        {"gate": 30, "io_pin": 30},
        {
            ("gate", "net_connection", "gate"): 108,
            ("gate", "net_connection", "io_pin"): 108,
            ("io_pin", "net_connection", "gate"): 108,
            ("io_pin", "net_connection", "io_pin"): 108,
            ("gate", "congestion_geom", "gate"): 1,
        },
    ),
}


def stable_numeric_transform(
    value: torch.Tensor,
    *,
    transform: str = "signed_log1p",
    append_finite_mask: bool = True,
) -> torch.Tensor:
    """Make physical features finite without hiding missing-value semantics."""

    value = value.float()
    finite = torch.isfinite(value)
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    if transform == "signed_log1p":
        value = torch.sign(value) * torch.log1p(value.abs())
    elif transform != "raw":
        raise ValueError("feature_transform must be 'signed_log1p' or 'raw'")
    if append_finite_mask:
        value = torch.cat((value, finite.float()), dim=-1)
    return value


def _module_key(edge_type: EdgeType) -> str:
    return relation_to_string(edge_type).replace("|", "__")


def _forward_contract_relation(edge_type: EdgeType) -> EdgeType:
    source, relation, target = edge_type
    if relation.startswith("rev_"):
        return target, relation[4:], source
    return edge_type


class NativeViewFeatureEncoder(nn.Module):
    """Encode one native view with independent node and relation projections.

    Edge features are projected per relation and mean-aggregated into their
    endpoints before message passing.  This keeps C/D/E/F's rich 41/32/34/108-D
    edge features in the model instead of silently discarding them in HGT/HAN.
    """

    def __init__(
        self,
        *,
        view_type: str,
        node_types: tuple[str, ...] | list[str],
        message_edge_types: tuple[EdgeType, ...] | list[EdgeType],
        hidden_dim: int,
        feature_transform: str = "signed_log1p",
        append_finite_mask: bool = True,
    ) -> None:
        super().__init__()
        self.view_type = view_type.upper()
        if self.view_type not in VIEW_CONTRACTS:
            raise ValueError(f"Unknown native view {view_type!r}")
        self.contract = VIEW_CONTRACTS[self.view_type]
        self.node_types = tuple(node_types)
        self.message_edge_types = tuple(message_edge_types)
        self.hidden_dim = hidden_dim
        self.feature_transform = feature_transform
        self.append_finite_mask = append_finite_mask

        if set(self.node_types) != set(self.contract.node_dims):
            raise ValueError(
                f"{self.view_type} node schema mismatch: expected "
                f"{sorted(self.contract.node_dims)}, got {sorted(self.node_types)}"
            )
        self.node_projections = nn.ModuleDict(
            {node_type: nn.LazyLinear(hidden_dim) for node_type in self.node_types}
        )
        self.node_norms = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types}
        )
        self.edge_projections = nn.ModuleDict()
        for edge_type in self.message_edge_types:
            canonical = _forward_contract_relation(edge_type)
            width = self.contract.edge_dims.get(canonical)
            if width is None:
                raise ValueError(
                    f"{self.view_type} message relation {edge_type} is outside its contract"
                )
            if width:
                self.edge_projections[_module_key(edge_type)] = nn.LazyLinear(
                    hidden_dim
                )
        self.global_projection = nn.LazyLinear(hidden_dim)
        self.node_type_embedding = nn.Embedding(len(self.node_types), hidden_dim)

    def _preprocess(self, value: torch.Tensor) -> torch.Tensor:
        return stable_numeric_transform(
            value,
            transform=self.feature_transform,
            append_finite_mask=self.append_finite_mask,
        )

    def validate_graph(self, data) -> None:
        for node_type, width in self.contract.node_dims.items():
            actual = int(data[node_type].x.size(1))
            if actual != width:
                raise ValueError(
                    f"{self.view_type}.{node_type}.x expected {width}D, got {actual}D"
                )
        for edge_type in self.message_edge_types:
            canonical = _forward_contract_relation(edge_type)
            expected = self.contract.edge_dims[canonical]
            store = data[edge_type]
            actual = int(store.edge_attr.size(1)) if hasattr(store, "edge_attr") else 0
            if actual != expected:
                raise ValueError(
                    f"{self.view_type}.{edge_type}.edge_attr expected {expected}D, "
                    f"got {actual}D"
                )

    def forward(self, data) -> dict[str, torch.Tensor]:
        self.validate_graph(data)
        global_features = data._global_store.get("global_features")
        if global_features is None:
            global_context = None
        else:
            transformed_global = self._preprocess(global_features.reshape(1, -1))
            global_context = self.global_projection(transformed_global).squeeze(0)

        encoded: dict[str, torch.Tensor] = {}
        edge_sum: dict[str, torch.Tensor] = {}
        edge_count: dict[str, torch.Tensor] = {}
        for type_id, node_type in enumerate(self.node_types):
            projected = self.node_projections[node_type](
                self._preprocess(data[node_type].x)
            )
            projected = projected + self.node_type_embedding.weight[type_id]
            if global_context is not None:
                projected = projected + global_context
            encoded[node_type] = projected
            edge_sum[node_type] = projected.new_zeros(projected.shape)
            edge_count[node_type] = projected.new_zeros((projected.size(0), 1))

        for edge_type in self.message_edge_types:
            store = data[edge_type]
            if not hasattr(store, "edge_attr") or store.edge_attr.size(1) == 0:
                continue
            projected = self.edge_projections[_module_key(edge_type)](
                self._preprocess(store.edge_attr)
            )
            source_type, _, target_type = edge_type
            source_index, target_index = store.edge_index
            edge_sum[source_type].index_add_(0, source_index, projected)
            edge_sum[target_type].index_add_(0, target_index, projected)
            ones = projected.new_ones((projected.size(0), 1))
            edge_count[source_type].index_add_(0, source_index, ones)
            edge_count[target_type].index_add_(0, target_index, ones)

        return {
            node_type: self.node_norms[node_type](
                value + edge_sum[node_type] / edge_count[node_type].clamp_min(1.0)
            )
            for node_type, value in encoded.items()
        }
