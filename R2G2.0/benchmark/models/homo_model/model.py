"""同构 GNN：PyG 卷积、残差归一化，以及节点/边 MLP 预测头。"""

from __future__ import annotations

import torch
from dataset import HomogeneousSchema
from encoder import HomogeneousFeatureEncoder
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv, GINEConv, ResGatedGraphConv


def _mlp(
    input_dim: int, hidden_dim: int, layers: int, output_dim: int
) -> nn.Sequential:
    if layers < 1:
        raise ValueError("MLP layers must be positive")
    modules: list[nn.Module] = []
    current = input_dim
    for _ in range(layers - 1):
        modules.extend((nn.Linear(current, hidden_dim), nn.GELU()))
        current = hidden_dim
    modules.append(nn.Linear(current, output_dim))
    return nn.Sequential(*modules)


class HomogeneousGNN(nn.Module):
    """GAT/GINE/ResGatedGCN operating on one homogeneous PyG graph."""

    def __init__(
        self,
        *,
        schema: HomogeneousSchema,
        model_name: str,
        hidden_dim: int,
        num_layers: int,
        head_layers: int,
        heads: int,
        dropout: float,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.activation_checkpointing = activation_checkpointing
        self.encoder = HomogeneousFeatureEncoder(schema, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if model_name == "gat":
                layer = GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=hidden_dim,
                    add_self_loops=False,
                )
            elif model_name == "gine":
                update = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                layer = GINEConv(update, edge_dim=hidden_dim)
            elif model_name == "resgatedgcn":
                layer = ResGatedGraphConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
            else:
                raise ValueError(f"Unknown homogeneous model {model_name!r}")
            self.layers.append(layer)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.node_head = _mlp(hidden_dim, hidden_dim, head_layers, 1)
        self.edge_head = _mlp(hidden_dim * 4, hidden_dim, head_layers, 1)

    def encode(self, data) -> torch.Tensor:
        value, edge_attr = self.encoder(data)
        for layer, norm in zip(self.layers, self.norms):

            def block(node_value, relation_value, layer=layer, norm=norm):
                update = layer(node_value, data.edge_index, edge_attr=relation_value)
                return norm(node_value + self.dropout(F.gelu(update)))

            if self.training and self.activation_checkpointing:
                value = checkpoint(block, value, edge_attr, use_reentrant=False)
            else:
                value = block(value, edge_attr)
        return value

    def predict(
        self, data, embeddings: torch.Tensor, target_rows: torch.Tensor
    ) -> torch.Tensor:
        if data.target_kind == "node":
            nodes = data.target_index[target_rows]
            return self.node_head(embeddings[nodes]).squeeze(-1)
        if data.target_kind == "edge":
            return self._predict_edges(data, embeddings, target_rows)
        if data.target_kind != "mixed":
            raise ValueError(f"Unknown target kind {data.target_kind!r}")

        kind = data.target_kind_code[target_rows]
        node_positions = torch.nonzero(kind == 0, as_tuple=False).flatten()
        edge_positions = torch.nonzero(kind == 1, as_tuple=False).flatten()
        if node_positions.numel() + edge_positions.numel() != target_rows.numel():
            raise ValueError("Mixed target kind codes must be 0 (node) or 1 (edge)")
        prediction = embeddings.new_zeros(target_rows.numel())
        if node_positions.numel():
            nodes = data.target_node_index[target_rows[node_positions]]
            node_prediction = self.node_head(embeddings[nodes]).squeeze(-1)
            prediction = prediction.index_copy(0, node_positions, node_prediction)
        if edge_positions.numel():
            edge_prediction = self._predict_edges(
                data, embeddings, target_rows[edge_positions]
            )
            prediction = prediction.index_copy(0, edge_positions, edge_prediction)
        return prediction

    def _predict_edges(
        self, data, embeddings: torch.Tensor, target_rows: torch.Tensor
    ) -> torch.Tensor:
        endpoints = data.target_edge_index[:, target_rows]
        source = embeddings[endpoints[0]]
        target = embeddings[endpoints[1]]
        pair = torch.cat(
            (source, target, source * target, (source - target).abs()), dim=-1
        )
        return self.edge_head(pair).squeeze(-1)
