#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heterogeneous graph merging script - improved version

This script merges multiple heterogeneous graph files into one large graph, referencing the design method of advanced_graph_integration1.py:
- Generate graph_id for all nodes of each subgraph as a distinguishing marker for different graphs
- Preserve all global features (global_features, die_coordinates, etc.)
- Preserve all node and edge features and labels
- Ensure data integrity and correctness

Functionality:
- Load all heterogeneous graph files in the specified directory
- Add graph_id feature to all node types for each graph
- Merge all nodes and edges, maintaining node index continuity
- Preserve global features and label information
- Save the merged large graph

Author: EDA for AI Team
Date: 2024

Usage:
    python merge_heterographs.py
"""

import os
import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import List, Dict, Tuple, Optional
import glob
from tqdm import tqdm

def load_heterograph_files(input_dir: str) -> List[Tuple[str, HeteroData]]:
    """Load all heterogeneous graph files in the specified directory

    Args:
        input_dir: Input directory path

    Returns:
        List[Tuple[str, HeteroData]]: List of (filename, heterogeneous graph data)
    """
    print(f"🔍 Scanning directory: {input_dir}")

    # Find all .pt files
    pattern = os.path.join(input_dir, "*.pt")
    pt_files = glob.glob(pattern)

    # Filter out non-heterogeneous graph files (such as report files)
    heterograph_files = [f for f in pt_files if f.endswith('_c_heterograph.pt')]

    print(f"📁 Found {len(heterograph_files)} RC heterogeneous graph files")

    loaded_graphs = []
    for file_path in sorted(heterograph_files):
        try:
            print(f"📖 Loading: {os.path.basename(file_path)}")
            data = torch.load(file_path, weights_only=False)

            # Extract design name (remove _c_heterograph.pt suffix)
            design_name = os.path.basename(file_path).replace('_c_heterograph.pt', '')
            loaded_graphs.append((design_name, data))

        except Exception as e:
            print(f"⚠️  Warning: Unable to load {file_path}: {e}")
            continue

    print(f"✅ Successfully loaded {len(loaded_graphs)} heterogeneous graphs")
    return loaded_graphs

def add_graph_id_to_all_nodes(data: HeteroData, graph_id: int) -> HeteroData:
    """Add subgraph ID graph_id feature to all node types of the heterogeneous graph

    Args:
        data: Original heterogeneous graph data
        graph_id: Subgraph ID (0-29)

    Returns:
        HeteroData: Heterogeneous graph data with graph_id added
    """
    print(f"🏷️  Adding graph_id to subgraph {graph_id}...")

    # Copy original data
    new_data = HeteroData()

    # Copy data for all node types and add subgraph ID graph_id
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x') and data[node_type].x is not None:
            original_features = data[node_type].x
            num_nodes = original_features.shape[0]

            # Create subgraph ID graph_id feature column (all nodes have the same subgraph ID)
            graph_id_feature = torch.full((num_nodes, 1), graph_id, dtype=torch.float32)

            # Add graph_id feature to the end of original features
            new_features = torch.cat([original_features, graph_id_feature], dim=1)
            new_data[node_type].x = new_features

            print(f"  📊 Added subgraph ID {graph_id} to {num_nodes} {node_type} nodes")

        # Copy node labels (if they exist)
        if hasattr(data[node_type], 'y') and data[node_type].y is not None:
            new_data[node_type].y = data[node_type].y.clone()
            print(f"  🏷️  Copied {node_type} node labels: {data[node_type].y.shape}")

    # Copy data for all edge types
    for edge_type in data.edge_types:
        new_data[edge_type].edge_index = data[edge_type].edge_index.clone()
        if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
            new_data[edge_type].edge_attr = data[edge_type].edge_attr.clone()

    # Copy global features (if they exist)
    if hasattr(data, 'global_features') and data.global_features is not None:
        new_data.global_features = data.global_features.clone()
        print(f"  🌐 Copied global features: {data.global_features.shape}")

    # Copy graph-level labels (if they exist)
    if hasattr(data, 'y') and data.y is not None:
        new_data.y = data.y.clone()
        print(f"  🎯 Copied graph-level labels: {data.y.shape}")

    # Copy die coordinates (if they exist)
    if hasattr(data, 'die_coordinates') and data.die_coordinates is not None:
        new_data.die_coordinates = data.die_coordinates.clone()
        print(f"  📍 Copied die coordinates: {data.die_coordinates.shape}")

    return new_data

def merge_heterographs_advanced(graph_list: List[Tuple[str, HeteroData]]) -> HeteroData:
    """Merge multiple heterogeneous graphs into one large graph (improved version)

    Reference the design method of advanced_graph_integration1.py

    Args:
        graph_list: List of (design name, heterogeneous graph data)

    Returns:
        HeteroData: Merged large graph
    """
    print(f"🔗 Starting to merge {len(graph_list)} heterogeneous graphs...")

    if not graph_list:
        raise ValueError("No heterogeneous graphs available for merging")

    # Add subgraph ID graph_id to each graph and collect
    processed_graphs = []
    design_names = []

    for i, (design_name, data) in enumerate(tqdm(graph_list, desc="Processing subgraphs")):
        print(f"🏷️  Processing graph {i+1}/{len(graph_list)}: {design_name}")
        processed_data = add_graph_id_to_all_nodes(data, i)
        processed_graphs.append(processed_data)
        design_names.append(design_name)

    # Create merged heterogeneous graph
    merged_graph = HeteroData()

    # Get all node types and edge types
    all_node_types = set()
    all_edge_types = set()
    for data in processed_graphs:
        all_node_types.update(data.node_types)
        all_edge_types.update(data.edge_types)

    print(f"📋 Node types: {sorted(all_node_types)}")
    print(f"📋 Edge types: {sorted(all_edge_types)}")

    # Record offsets for each node type in each graph
    node_offsets = {}
    for node_type in all_node_types:
        node_offsets[node_type] = []
        current_offset = 0
        for data in processed_graphs:
            node_offsets[node_type].append(current_offset)
            if node_type in data.node_types and hasattr(data[node_type], 'x') and data[node_type].x is not None:
                current_offset += data[node_type].x.shape[0]
    
    # Merge node features
    print("\n🔗 Merging node features...")
    for node_type in all_node_types:
        node_features = []
        node_labels = []

        for data in processed_graphs:
            if node_type in data.node_types:
                if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                    node_features.append(data[node_type].x)

                if hasattr(data[node_type], 'y') and data[node_type].y is not None:
                    node_labels.append(data[node_type].y)

        if node_features:
            merged_graph[node_type].x = torch.cat(node_features, dim=0)
            print(f"  📊 {node_type}: {merged_graph[node_type].x.shape[0]} nodes, {merged_graph[node_type].x.shape[1]} feature dimensions")

        if node_labels:
            merged_graph[node_type].y = torch.cat(node_labels, dim=0)
            print(f"  🏷️  {node_type}: {merged_graph[node_type].y.shape[0]} node labels")

    # Merge edge indices
    print("\n🔗 Merging edge indices...")
    for edge_type in all_edge_types:
        edge_indices = []
        edge_attrs = []

        src_type, rel_type, dst_type = edge_type

        for i, data in enumerate(processed_graphs):
            if edge_type in data.edge_types:
                # Get original edge indices
                edge_index = data[edge_type].edge_index.clone()

                # Add offsets
                edge_index[0] += node_offsets[src_type][i]  # Source node offset
                edge_index[1] += node_offsets[dst_type][i]  # Target node offset

                edge_indices.append(edge_index)

                # Handle edge attributes (if they exist)
                if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                    edge_attrs.append(data[edge_type].edge_attr)

        if edge_indices:
            merged_graph[edge_type].edge_index = torch.cat(edge_indices, dim=1)
            print(f"  🔗 {edge_type}: {merged_graph[edge_type].edge_index.shape[1]} edges")

            # Merge edge attributes
            if edge_attrs:
                merged_graph[edge_type].edge_attr = torch.cat(edge_attrs, dim=0)
                print(f"    Edge features: {merged_graph[edge_type].edge_attr.shape}")
    
    # Merge global features
    print("\n🌐 Merging global features...")
    global_features_list = []
    graph_labels_list = []
    die_coordinates_list = []

    for i, data in enumerate(processed_graphs):
        # Collect global features
        if hasattr(data, 'global_features') and data.global_features is not None:
            global_features_list.append(data.global_features)
        else:
            # If no global features, create default zero vector
            default_global_features = torch.zeros(5)  # Based on observed dimensions
            global_features_list.append(default_global_features)

        # Collect graph-level labels
        if hasattr(data, 'y') and data.y is not None:
            graph_labels_list.append(data.y)
        else:
            # If no graph-level labels, create default label
            default_label = torch.zeros(3)  # Based on observed dimensions
            graph_labels_list.append(default_label)

        # Collect die coordinates
        if hasattr(data, 'die_coordinates') and data.die_coordinates is not None:
            die_coordinates_list.append(data.die_coordinates)
        else:
            # If no die coordinates, create default coordinates
            default_coordinates = torch.zeros(2, 2)
            die_coordinates_list.append(default_coordinates)

    # Stack global features
    if global_features_list:
        merged_graph.global_features = torch.stack(global_features_list, dim=0)
        print(f"  ✅ Global features: {merged_graph.global_features.shape}")

    # Stack graph-level labels
    if graph_labels_list:
        merged_graph.y = torch.stack(graph_labels_list, dim=0)
        print(f"  ✅ Graph-level labels: {merged_graph.y.shape}")

    # Stack die coordinates
    if die_coordinates_list:
        merged_graph.die_coordinates = torch.stack(die_coordinates_list, dim=0)
        print(f"  ✅ Die coordinates: {merged_graph.die_coordinates.shape}")

    # Add metadata
    merged_graph.design_names = design_names
    merged_graph.num_subgraphs = len(processed_graphs)
    merged_graph.name = f"merged_heterograph_{len(processed_graphs)}_designs"

    print(f"\n✅ Merge complete!")
    return merged_graph

def save_merged_graph(merged_graph: HeteroData, output_path: str):
    """Save the merged heterogeneous graph

    Args:
        merged_graph: Merged heterogeneous graph
        output_path: Output file path
    """
    print(f"💾 Saving merged graph to: {output_path}")
    torch.save(merged_graph, output_path, pickle_protocol=4)

    # Display final statistics
    print(f"\n📈 Merged graph statistics:")
    print(f"  - Node types: {len(merged_graph.node_types)}")
    print(f"  - Edge types: {len(merged_graph.edge_types)}")

    total_nodes = sum(merged_graph[nt].x.shape[0] for nt in merged_graph.node_types if hasattr(merged_graph[nt], 'x'))
    total_edges = sum(merged_graph[et].edge_index.shape[1] for et in merged_graph.edge_types)
    print(f"  - Total nodes: {total_nodes:,}")
    print(f"  - Total edges: {total_edges:,}")
    print(f"  - Number of subgraphs: {merged_graph.num_subgraphs}")

    # Display count of each node type
    for node_type in sorted(merged_graph.node_types):
        if hasattr(merged_graph[node_type], 'x'):
            count = merged_graph[node_type].x.shape[0]
            feature_dim = merged_graph[node_type].x.shape[1]
            print(f"    * {node_type}: {count:,} nodes, {feature_dim} feature dimensions (including graph_id)")

            # Verify graph_id range
            graph_ids = merged_graph[node_type].x[:, -1]  # Last column is graph_id
            min_id = int(graph_ids.min().item())
            max_id = int(graph_ids.max().item())
            print(f"      (graph_id range: {min_id} - {max_id})")

    # Display global feature information
    if hasattr(merged_graph, 'global_features'):
        print(f"  - Global features: {merged_graph.global_features.shape}")
    if hasattr(merged_graph, 'y'):
        print(f"  - Graph-level labels: {merged_graph.y.shape}")
    if hasattr(merged_graph, 'die_coordinates'):
        print(f"  - Die coordinates: {merged_graph.die_coordinates.shape}")

def main():
    """Main function"""
    print("🚀 Heterogeneous graph merging script started (improved version)")
    print("=" * 60)

    # Configure paths
    input_dir = "/Users/david/Desktop/route1.3/RC"
    output_path = "/Users/david/Desktop/route1.3/RC_merged_heterograph.pt"

    try:
        # 1. Load all heterogeneous graph files
        graph_list = load_heterograph_files(input_dir)

        if not graph_list:
            print("❌ Error: No loadable heterogeneous graph files found")
            return 1

        # 2. Merge heterogeneous graphs
        merged_graph = merge_heterographs_advanced(graph_list)

        # 3. Save merge results
        save_merged_graph(merged_graph, output_path)

        print("\n🎉 Heterogeneous graph merging complete!")
        print(f"📁 Output file: {output_path}")
        print(f"🔧 Improvements: graph_id generation, global features preservation, node labels preservation")
        return 0

    except Exception as e:
        print(f"❌ Error during merging: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())