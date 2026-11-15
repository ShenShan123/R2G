"""输入编码：有限值掩码、signed-log 特征、类型投影与全局特征。"""

from __future__ import annotations

import torch
from dataset import HomogeneousSchema
from torch import nn


def stable_numeric_transform(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    finite = torch.isfinite(value)
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    value = torch.sign(value) * torch.log1p(value.abs())
    return torch.cat((value, finite.float()), dim=-1)


class HomogeneousFeatureEncoder(nn.Module):
    """Type-aware input encoder followed by a homogeneous GNN.

    Node and relation identities originate from PyG's official conversion,
    while native feature schemas keep independent projections.  This avoids
    treating padded columns from different node types as identical semantics.
    """

    def __init__(self, schema: HomogeneousSchema, hidden_dim: int) -> None:
        super().__init__()
        self.schema = schema
        self.hidden_dim = hidden_dim
        self.node_projections = nn.ModuleList(
            nn.Linear(width * 2, hidden_dim) for width in schema.node_widths
        )
        self.node_norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in schema.node_widths
        )
        self.node_type_embedding = nn.Embedding(len(schema.node_widths), hidden_dim)
        self.edge_projections = nn.ModuleList(
            nn.Linear(width * 2, hidden_dim) if width else nn.Identity()
            for width in schema.edge_widths
        )
        self.edge_type_embedding = nn.Embedding(len(schema.edge_widths), hidden_dim)
        self.global_projection = (
            nn.Linear(schema.global_width * 2, hidden_dim)
            if schema.global_width
            else None
        )

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        encoded_x = x.new_zeros((x.size(0), self.hidden_dim))
        global_context = None
        if self.global_projection is not None and hasattr(data, "global_features"):
            global_context = self.global_projection(
                stable_numeric_transform(data.global_features)
            ).squeeze(0)
        for type_id, width in enumerate(self.schema.node_widths):
            mask = data.node_type == type_id
            if not bool(mask.any()):
                continue
            value = self.node_projections[type_id](
                stable_numeric_transform(x[mask, :width])
            )
            value = value + self.node_type_embedding.weight[type_id]
            if global_context is not None:
                value = value + global_context
            encoded_x[mask] = self.node_norms[type_id](value)

        encoded_edge = x.new_zeros((data.edge_index.size(1), self.hidden_dim))
        for relation_id, width in enumerate(self.schema.edge_widths):
            mask = data.edge_type == relation_id
            if not bool(mask.any()):
                continue
            value = self.edge_type_embedding.weight[relation_id].expand(
                int(mask.sum()), -1
            )
            if width:
                value = value + self.edge_projections[relation_id](
                    stable_numeric_transform(data.edge_attr[mask, :width])
                )
            encoded_edge[mask] = value
        return encoded_x, encoded_edge
