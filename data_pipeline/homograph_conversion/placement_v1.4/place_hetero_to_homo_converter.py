#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Universal heterograph to homograph converter

Universal tool for converting PyTorch Geometric heterogeneous graphs (HeteroData) to homogeneous graphs (Data).

Features:
- Automatically handle different node and edge types
- Unify feature dimensions
- Preserve labels and global features
- Support custom mapping tables or auto-generation

Author: EDA for AI Team
"""

import os
import argparse
import torch
from torch_geometric.data import HeteroData, Data
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class HeteroToHomoConverter:
    """Universal heterograph to homograph converter"""

    def __init__(self, target_feature_dim: int = 10):
        """Initialize converter

        Args:
            target_feature_dim: Target feature dimension, default is 10 dimensions
        """
        self.target_feature_dim = target_feature_dim

        # Built-in mapping table - based on comprehensive analysis of all node and edge types
        self.node_type_to_id = {
            "gate": 0,
            "io_pin": 1,
            "net": 2,
            "pin": 3
        }

        self.edge_type_to_id = {
            "('gate', 'connects_to', 'gate')": 0,
            "('gate', 'connects_to', 'net')": 1,
            "('gate', 'has', 'pin')": 2,
            "('gate', 'has_pin', 'pin')": 2,  # Same as 'has', merged encoding
            "('io_pin', 'connects_to', 'gate')": 3,
            "('io_pin', 'connects_to', 'net')": 4,
            "('io_pin', 'connects_to', 'pin')": 5,
            "('pin', 'connects_to', 'net')": 6,
            "('pin', 'connects_to', 'pin')": 7,
            "('pin', 'gate_connects', 'pin')": 8
        }

    def _create_mapping(self, hetero_data: HeteroData):
        """Automatically create type mappings"""
        # Create node type mapping
        for i, node_type in enumerate(hetero_data.node_types):
            if node_type not in self.node_type_to_id:
                self.node_type_to_id[node_type] = i

        # Create edge type mapping
        for i, edge_type in enumerate(hetero_data.edge_types):
            edge_type_str = str(edge_type)
            if edge_type_str not in self.edge_type_to_id:
                self.edge_type_to_id[edge_type_str] = i

    def _normalize_features(self, features: torch.Tensor, target_dim: int = 10) -> torch.Tensor:
        """Unify feature dimensions to specified dimension (default 10 dimensions)

        Args:
            features: Input feature tensor
            target_dim: Target feature dimension, default is 10 dimensions

        Returns:
            torch.Tensor: Normalized feature tensor
        """
        current_dim = features.shape[1]

        if current_dim == target_dim:
            return features
        elif current_dim < target_dim:
            # Zero padding to target dimension
            padding = torch.zeros((features.shape[0], target_dim - current_dim), dtype=features.dtype)
            return torch.cat([features, padding], dim=1)
        else:
            # Truncate to target dimension
            return features[:, :target_dim]

    def _extract_labels(self, data, label_attrs: List[str] = None) -> Optional[torch.Tensor]:
        """Extract labels"""
        if label_attrs is None:
            label_attrs = ['y', 'label', 'target', 'node_label', 'edge_label']

        for attr in label_attrs:
            if hasattr(data, attr) and getattr(data, attr) is not None:
                return getattr(data, attr)
        return None

    def convert(self, hetero_data: HeteroData) -> Data:
        """Convert heterograph to homograph

        Args:
            hetero_data: Input heterograph data

        Returns:
            Data: Converted homograph data
        """
        # If no mapping table, automatically create
        if not self.node_type_to_id or not self.edge_type_to_id:
            self._create_mapping(hetero_data)

        # Convert nodes
        node_features, node_labels, node_mapping = self._convert_nodes(hetero_data)

        # Convert edges
        edge_index, edge_features, edge_labels = self._convert_edges(hetero_data, node_mapping)

        # Create homograph
        homo_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features
        )

        # Add labels
        if node_labels is not None:
            homo_data.y = node_labels
        if edge_labels is not None:
            homo_data.edge_label = edge_labels

        # Preserve global attributes
        self._preserve_global_attributes(hetero_data, homo_data)

        # Note: No longer save mapping information to .pt file
        # Mapping table information has been encoded into the last dimension of node and edge features

        return homo_data

    def _convert_nodes(self, hetero_data: HeteroData) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """Convert node data"""
        all_node_features = []
        all_node_labels = []
        node_mapping = {}

        current_idx = 0
        has_labels = False

        for node_type in hetero_data.node_types:
            node_data = hetero_data[node_type]
            num_nodes = node_data.num_nodes

            # Get node type ID
            node_type_id = self.node_type_to_id.get(node_type, 0)

            # Process node features
            if hasattr(node_data, 'x') and node_data.x is not None:
                features = node_data.x.float()
            else:
                features = torch.zeros((num_nodes, 1), dtype=torch.float)

            # Unify feature dimensions (reserve one dimension for type encoding)
            feature_dim = self.target_feature_dim - 1
            normalized_features = self._normalize_features(features, feature_dim)

            # Add node type encoding
            type_column = torch.full((num_nodes, 1), node_type_id, dtype=torch.float)
            final_features = torch.cat([normalized_features, type_column], dim=1)
            all_node_features.append(final_features)

            # Process node labels
            labels = self._extract_labels(node_data)
            if labels is not None:
                all_node_labels.append(labels)
                has_labels = True
            else:
                all_node_labels.append(torch.full((num_nodes,), -1, dtype=torch.long))

            # Build node mapping
            for i in range(num_nodes):
                node_mapping[(node_type, i)] = current_idx + i

            current_idx += num_nodes

        # Merge all node data
        node_features = torch.cat(all_node_features, dim=0)
        node_labels = torch.cat(all_node_labels, dim=0) if has_labels else None

        return node_features, node_labels, node_mapping

    def _convert_edges(self, hetero_data: HeteroData, node_mapping: Dict) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Convert edge data"""
        all_edge_indices = []
        all_edge_features = []
        all_edge_labels = []
        has_labels = False

        for edge_type in hetero_data.edge_types:
            src_type, relation, dst_type = edge_type
            edge_data = hetero_data[edge_type]

            if not hasattr(edge_data, 'edge_index') or edge_data.edge_index is None:
                continue

            # Get edge type ID
            edge_type_str = str(edge_type)
            edge_type_id = self.edge_type_to_id.get(edge_type_str, 0)

            edge_index = edge_data.edge_index
            num_edges = edge_index.shape[1]

            # Convert edge indices
            src_indices = edge_index[0]
            dst_indices = edge_index[1]

            new_src_indices = []
            new_dst_indices = []
            valid_indices = []

            for i in range(num_edges):
                src_key = (src_type, src_indices[i].item())
                dst_key = (dst_type, dst_indices[i].item())

                if src_key in node_mapping and dst_key in node_mapping:
                    new_src_indices.append(node_mapping[src_key])
                    new_dst_indices.append(node_mapping[dst_key])
                    valid_indices.append(i)

            if not new_src_indices:
                continue

            new_edge_index = torch.stack([
                torch.tensor(new_src_indices, dtype=torch.long),
                torch.tensor(new_dst_indices, dtype=torch.long)
            ])
            all_edge_indices.append(new_edge_index)

            # Process edge features
            if hasattr(edge_data, 'edge_attr') and edge_data.edge_attr is not None:
                features = edge_data.edge_attr.float()[valid_indices]
            else:
                features = torch.zeros((len(new_src_indices), 1), dtype=torch.float)

            # Unify edge feature dimensions
            feature_dim = self.target_feature_dim - 1
            normalized_features = self._normalize_features(features, feature_dim)

            # Add edge type encoding
            type_column = torch.full((len(new_src_indices), 1), edge_type_id, dtype=torch.float)
            final_features = torch.cat([normalized_features, type_column], dim=1)
            all_edge_features.append(final_features)

            # Process edge labels
            labels = self._extract_labels(edge_data)
            if labels is not None:
                all_edge_labels.append(labels[valid_indices])
                has_labels = True
            else:
                all_edge_labels.append(torch.full((len(new_src_indices),), -1, dtype=torch.long))

        # Merge all edge data
        if all_edge_indices:
            edge_index = torch.cat(all_edge_indices, dim=1)
            edge_features = torch.cat(all_edge_features, dim=0)
            edge_labels = torch.cat(all_edge_labels, dim=0) if has_labels else None
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_features = torch.empty((0, self.target_feature_dim), dtype=torch.float)
            edge_labels = None

        return edge_index, edge_features, edge_labels

    def _preserve_global_attributes(self, hetero_data: HeteroData, homo_data: Data):
        """Preserve global attributes"""
        # Preserve global features
        if hasattr(hetero_data, 'global_features'):
            homo_data.global_features = hetero_data.global_features

        # Preserve global labels (avoid conflict with node labels)
        if hasattr(hetero_data, 'y') and hetero_data.y is not None:
            homo_data.global_y = hetero_data.y

        # Preserve other global attributes
        for attr in ['die_coordinates', 'global_label', 'metadata']:
            if hasattr(hetero_data, attr):
                setattr(homo_data, attr, getattr(hetero_data, attr))


def convert_file(input_file: str, output_file: str, converter: HeteroToHomoConverter):
    """Convert a single file"""
    try:
        # Load heterograph
        hetero_data = torch.load(input_file, weights_only=False)

        # Convert to homograph
        homo_data = converter.convert(hetero_data)

        # Save result
        torch.save(homo_data, output_file)

        print(f"Conversion completed: {input_file} -> {output_file}")
        print(f"  Node count: {homo_data.x.shape[0]}")
        print(f"  Edge count: {homo_data.edge_index.shape[1]}")
        print(f"  Node feature dimension: {homo_data.x.shape[1]}")
        print(f"  Edge feature dimension: {homo_data.edge_attr.shape[1]}")

        return True

    except Exception as e:
        print(f"Conversion failed: {input_file} - {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Universal heterograph to homograph converter")
    parser.add_argument("input", help="Input heterograph file path or directory path")
    parser.add_argument("-o", "--output", help="Output homograph file path")
    parser.add_argument("--feature-dim", type=int, default=10, help="Target feature dimension")

    args = parser.parse_args()

    # Initialize converter
    converter = HeteroToHomoConverter(target_feature_dim=args.feature_dim)

    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single file conversion
        # Generate output file name
        if not args.output:
            input_path = Path(args.input)
            args.output = str(input_path.with_name(
                input_path.stem.replace("_heterograph", "_homograph") + ".pt"
            ))

        # Execute conversion
        success = convert_file(args.input, args.output, converter)
        return 0 if success else 1

    elif os.path.isdir(args.input):
        # Batch convert all .pt files in directory
        success_count = 0
        total_count = 0

        for filename in os.listdir(args.input):
            if filename.endswith('.pt'):
                input_file = os.path.join(args.input, filename)
                output_file = filename.replace('heterograph', 'homograph')

                try:
                    success = convert_file(input_file, output_file, converter)
                    if success:
                        success_count += 1
                except Exception as e:
                    print(f"Conversion failed: {input_file} - {e}")

                total_count += 1

        print(f"Batch conversion completed: {success_count}/{total_count} files successfully converted")
        return 0 if success_count == total_count else 1
    else:
        print(f"Error: Input path does not exist: {args.input}")
        return 1


if __name__ == "__main__":
    exit(main())