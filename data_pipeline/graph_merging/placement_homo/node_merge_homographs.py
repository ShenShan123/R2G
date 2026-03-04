#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Homogeneous graph merging script

This script merges multiple homogeneous graph files into one large graph, referencing the design method of the heterogeneous graph merging script:
- Generate graph_id for all nodes of each subgraph as a distinguishing marker for different graphs
- Preserve all node and edge features and labels
- Merge all nodes and edges, maintaining node index continuity
- Ensure data integrity and correctness

Functionality:
- Load all homogeneous graph files in the specified directory
- Add graph_id feature to all nodes of each graph
- Merge all nodes and edges, maintaining node index continuity
- Preserve node and edge feature information
- Save the merged large graph

Author: EDA for AI Team
Date: 2024

Usage:
    python merge_homographs.py
"""

import os
import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional
import glob
from tqdm import tqdm

def load_homograph_files(input_dir: str) -> List[Tuple[str, Data]]:
    """Load all homogeneous graph files in the specified directory

    Args:
        input_dir: Input directory path

    Returns:
        List[Tuple[str, Data]]: List of (filename, homogeneous graph data)
    """
    print(f"🔍 Scanning directory: {input_dir}")

    # Find all .pt files
    pattern = os.path.join(input_dir, "*.pt")
    pt_files = glob.glob(pattern)

    # Filter homogeneous graph files (ending with _homograph.pt)
    homograph_files = [f for f in pt_files if f.endswith('_homograph.pt')]

    print(f"📁 Found {len(homograph_files)} homogeneous graph files")

    loaded_graphs = []
    for file_path in sorted(homograph_files):
        try:
            print(f"📖 Loading: {os.path.basename(file_path)}")
            data = torch.load(file_path, weights_only=False)

            # Extract design name (remove _homograph.pt suffix)
            design_name = os.path.basename(file_path).replace('_homograph.pt', '')
            loaded_graphs.append((design_name, data))

        except Exception as e:
            print(f"⚠️  Warning: Unable to load {file_path}: {e}")
            continue

    print(f"✅ Successfully loaded {len(loaded_graphs)} homogeneous graphs")
    return loaded_graphs

def add_graph_id_to_nodes(data: Data, graph_id: int) -> Data:
    """Add subgraph ID graph_id feature to all nodes of the homogeneous graph (placed in the 9th dimension)

    Args:
        data: Original homogeneous graph data
        graph_id: Subgraph ID (0-29)

    Returns:
        Data: Homogeneous graph data with the 9th dimension modified to graph_id
    """
    print(f"🏷️  Setting 9th dimension to graph_id for subgraph {graph_id}...")

    # Copy original data
    new_data = Data()

    # Handle node features, set the 9th dimension (index 8) to graph_id
    if hasattr(data, 'x') and data.x is not None:
        original_features = data.x.clone()
        num_nodes = original_features.shape[0]
        feature_dim = original_features.shape[1]

        # Ensure feature dimension is at least 10
        if feature_dim < 10:
            # If dimension is less than 10, pad with zeros to 10
            padding = torch.zeros((num_nodes, 10 - feature_dim), dtype=torch.float32)
            original_features = torch.cat([original_features, padding], dim=1)
            print(f"  📏 Feature dimension expanded from {feature_dim} to 10")

        # Set the 9th dimension (index 8) to graph_id
        original_features[:, 8] = graph_id
        new_data.x = original_features

        print(f"  📊 Set 9th dimension to subgraph ID {graph_id} for {num_nodes} nodes")
        print(f"  📏 Node feature dimension maintained: {new_data.x.shape[1]}")

    # Copy edge index
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        new_data.edge_index = data.edge_index.clone()
        print(f"  🔗 Copied edge index: {data.edge_index.shape}")

    # Copy edge features
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        new_data.edge_attr = data.edge_attr.clone()
        print(f"  🔗 Copied edge features: {data.edge_attr.shape}")

    # Copy node labels (if they exist)
    if hasattr(data, 'y') and data.y is not None:
        new_data.y = data.y.clone()
        print(f"  🏷️  Copied node labels: {data.y.shape}")

    # Preserve global features and labels
    global_attrs = ['global_features', 'global_y', 'die_coordinates']
    for attr_name in global_attrs:
        if hasattr(data, attr_name):
            attr_value = getattr(data, attr_name)
            if torch.is_tensor(attr_value):
                setattr(new_data, attr_name, attr_value.clone())
                print(f"  🌐 Preserved global feature {attr_name}: {attr_value.shape}")

    # Copy other attributes (excluding processed and non-copiable ones)
    skip_attrs = ['x', 'edge_index', 'edge_attr', 'y'] + global_attrs
    for attr_name in dir(data):
        if not attr_name.startswith('_') and attr_name not in skip_attrs:
            try:
                attr_value = getattr(data, attr_name)
                if torch.is_tensor(attr_value):
                    setattr(new_data, attr_name, attr_value.clone())
                    print(f"  📋 Copied attribute {attr_name}: {attr_value.shape}")
                elif not callable(attr_value):
                    setattr(new_data, attr_name, attr_value)
                    print(f"  📋 Copied attribute {attr_name}: {type(attr_value)}")
            except:
                pass  # Ignore attributes that cannot be copied

    return new_data

def merge_homographs(graph_list: List[Tuple[str, Data]]) -> Data:
    """Merge multiple homogeneous graphs into one large graph, preserving global features and labels

    Args:
        graph_list: List of (design name, homogeneous graph data)

    Returns:
        Data: Merged large graph with global features and labels
    """
    print(f"🔗 Starting to merge {len(graph_list)} homogeneous graphs...")

    if not graph_list:
        raise ValueError("No homogeneous graphs available for merging")

    # Add subgraph ID graph_id to each graph and collect
    processed_graphs = []
    design_names = []

    for i, (design_name, data) in enumerate(tqdm(graph_list, desc="Processing subgraphs")):
        print(f"🏷️  Processing graph {i+1}/{len(graph_list)}: {design_name}")
        processed_data = add_graph_id_to_nodes(data, i)
        processed_graphs.append(processed_data)
        design_names.append(design_name)

    # Create merged homogeneous graph
    merged_graph = Data()

    # Record node offsets for each graph
    node_offsets = []
    current_offset = 0
    for data in processed_graphs:
        node_offsets.append(current_offset)
        if hasattr(data, 'x') and data.x is not None:
            current_offset += data.x.shape[0]

    print(f"📊 Node offsets: {node_offsets}")

    # Merge node features
    print("\n🔗 Merging node features...")
    node_features = []
    node_labels = []

    for data in processed_graphs:
        if hasattr(data, 'x') and data.x is not None:
            node_features.append(data.x)

        if hasattr(data, 'y') and data.y is not None:
            node_labels.append(data.y)

    if node_features:
        merged_graph.x = torch.cat(node_features, dim=0)
        print(f"  📊 Merged node features: {merged_graph.x.shape[0]} nodes, {merged_graph.x.shape[1]} feature dimensions")

    if node_labels:
        merged_graph.y = torch.cat(node_labels, dim=0)
        print(f"  🏷️  Merged node labels: {merged_graph.y.shape}")

    # Merge edge indices and edge features
    print("\n🔗 Merging edge indices and edge features...")
    edge_indices = []
    edge_attrs = []

    for i, data in enumerate(processed_graphs):
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            # Get original edge indices
            edge_index = data.edge_index.clone()

            # Add node offsets
            edge_index += node_offsets[i]

            edge_indices.append(edge_index)

            print(f"  🔗 Graph {i}: {edge_index.shape[1]} edges, offset: {node_offsets[i]}")

        # Handle edge features
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attrs.append(data.edge_attr)

    if edge_indices:
        merged_graph.edge_index = torch.cat(edge_indices, dim=1)
        print(f"  📊 Merged edge indices: {merged_graph.edge_index.shape}")

    if edge_attrs:
        merged_graph.edge_attr = torch.cat(edge_attrs, dim=0)
        print(f"  📊 Merged edge features: {merged_graph.edge_attr.shape}")

    # Merge global features and labels
    print("\n🌐 Merging global features and labels...")
    global_features_list = []
    global_y_list = []
    die_coordinates_list = []

    for i, data in enumerate(processed_graphs):
        # Collect global features
        if hasattr(data, 'global_features') and data.global_features is not None:
            # Add subgraph ID to global features for each subgraph
            global_feat = data.global_features.clone()
            # Add subgraph ID after global features (in the last dimension)
            global_feat_with_id = torch.cat([global_feat, torch.tensor([i], dtype=torch.float32)])
            global_features_list.append(global_feat_with_id)
            print(f"  🌐 Subgraph {i} global features: {global_feat.shape} -> {global_feat_with_id.shape}")

        # Collect global labels
        if hasattr(data, 'global_y') and data.global_y is not None:
            # Add subgraph ID after global labels (in the last dimension)
            global_y_with_id = torch.cat([data.global_y.clone(), torch.tensor([i], dtype=torch.float32)])
            global_y_list.append(global_y_with_id)
            print(f"  🏷️  Subgraph {i} global labels: {data.global_y.shape} -> {global_y_with_id.shape}")

        # Collect die coordinates
        if hasattr(data, 'die_coordinates') and data.die_coordinates is not None:
            die_coordinates_list.append(data.die_coordinates.clone())
            print(f"  📍 Subgraph {i} die coordinates: {data.die_coordinates.shape}")

    # Merge global features
    if global_features_list:
        merged_graph.global_features = torch.stack(global_features_list, dim=0)
        print(f"  📊 Merged global features: {merged_graph.global_features.shape}")

    if global_y_list:
        merged_graph.global_y = torch.stack(global_y_list, dim=0)
        print(f"  📊 Merged global labels: {merged_graph.global_y.shape}")

    if die_coordinates_list:
        merged_graph.die_coordinates = torch.stack(die_coordinates_list, dim=0)
        print(f"  📊 Merged die coordinates: {merged_graph.die_coordinates.shape}")

    # Add metadata
    merged_graph.design_names = design_names
    merged_graph.num_subgraphs = len(processed_graphs)
    merged_graph.name = f"merged_homograph_{len(processed_graphs)}_designs"

    print(f"\n✅ Merge complete!")
    return merged_graph

def save_merged_graph(merged_graph: Data, output_path: str):
    """Save the merged homogeneous graph

    Args:
        merged_graph: Merged homogeneous graph
        output_path: Output file path
    """
    print(f"💾 Saving merged graph to: {output_path}")
    torch.save(merged_graph, output_path, pickle_protocol=4)

    # Display final statistics
    print(f"\n📈 Merged graph statistics:")

    if hasattr(merged_graph, 'x'):
        total_nodes = merged_graph.x.shape[0]
        feature_dim = merged_graph.x.shape[1]
        print(f"  - Total nodes: {total_nodes:,}")
        print(f"  - Node feature dimension: {feature_dim} (9th dimension is graph_id)")

        # Verify graph_id range (9th dimension, index 8)
        graph_ids = merged_graph.x[:, 8]  # 9th dimension is graph_id
        min_id = int(graph_ids.min().item())
        max_id = int(graph_ids.max().item())
        print(f"  - graph_id range: {min_id} - {max_id}")

    if hasattr(merged_graph, 'edge_index'):
        total_edges = merged_graph.edge_index.shape[1]
        print(f"  - Total edges: {total_edges:,}")

    if hasattr(merged_graph, 'edge_attr'):
        edge_feature_dim = merged_graph.edge_attr.shape[1]
        print(f"  - Edge feature dimension: {edge_feature_dim}")

    # Display global feature information
    if hasattr(merged_graph, 'global_features'):
        print(f"  - Global features: {merged_graph.global_features.shape}")

    if hasattr(merged_graph, 'global_y'):
        print(f"  - Global labels: {merged_graph.global_y.shape}")

    if hasattr(merged_graph, 'die_coordinates'):
        print(f"  - Die coordinates: {merged_graph.die_coordinates.shape}")

    print(f"  - Number of subgraphs: {merged_graph.num_subgraphs}")

    # Display node count distribution for each subgraph
    if hasattr(merged_graph, 'x'):
        graph_ids = merged_graph.x[:, 8]  # 9th dimension is graph_id
        for i in range(merged_graph.num_subgraphs):
            count = (graph_ids == i).sum().item()
            design_name = merged_graph.design_names[i] if hasattr(merged_graph, 'design_names') else f"graph_{i}"
            print(f"    * {design_name}: {count:,} nodes")

def merge_directory(input_dir: str, output_path: str):
    """Merge all homogeneous graphs in the specified directory

    Args:
        input_dir: Input directory path
        output_path: Output file path
    """
    print(f"🚀 Starting to merge directory: {input_dir}")
    print("=" * 60)

    try:
        # 1. Load all homogeneous graph files
        graph_list = load_homograph_files(input_dir)

        if not graph_list:
            print("❌ Error: No loadable homogeneous graph files found")
            return False

        # 2. Merge homogeneous graphs
        merged_graph = merge_homographs(graph_list)

        # 3. Save merge results
        save_merged_graph(merged_graph, output_path)

        print(f"\n🎉 Directory {os.path.basename(input_dir)} merge complete!")
        print(f"📁 Output file: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Error during merging: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function - batch process B and C directories"""
    print("🚀 Homogeneous graph merging script started - batch processing version")
    print("=" * 80)

    # Configure paths - batch process B and C directories
    base_dir = "/Users/david/Desktop/benchmark_dataset/place_homo_v1.4/to_homogeneous"
    directories = ['B', 'C']  # Integrate B and C directories

    success_count = 0

    for dir_name in directories:
        input_dir = os.path.join(base_dir, dir_name)
        output_path = os.path.join(base_dir, f"place_{dir_name}_homograph.pt")

        print(f"\n{'='*20} Processing directory {dir_name} {'='*20}")

        if merge_directory(input_dir, output_path):
            success_count += 1
        else:
            print(f"❌ Directory {dir_name} merge failed")

    print(f"\n{'='*80}")
    print(f"🎉 Merge task complete! Success: {success_count}/{len(directories)}")
    print(f"🔧 Feature capabilities:")
    print(f"  - ✅ graph_id generation, placed in 9th dimension (node features)")
    print(f"  - ✅ Node features preserved")
    print(f"  - ✅ Edge features preserved")
    print(f"  - ✅ Global features and labels preserved")

    return 0 if success_count == len(directories) else 1

if __name__ == "__main__":
    exit(main())