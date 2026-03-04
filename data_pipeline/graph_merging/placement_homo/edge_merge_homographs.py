#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[CN][CN][CN][CN][CN][CN][CN] - global label integrated version

[CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN],[CN][CN][CN][CN][CN] labelinformation:
- [CN][CN][CN]graph's[CN][CN]nodegenerategraph_idas identification[CN][CN] for different graphs
- preserve allnode[CN]edge[CN]featureand label([CN][CN][CN]edge labely[CN]global labelglobal_y)
- [CN][CN][CN][CN]node[CN]edge,keepnode[CN][CN][CN][CN][CN][CN]
- keep[CN][CN][CN][CN][CN][CN]:x(nodefeature),edge_attr(edgefeature)
- ensure data integrity and correctness

[CN][CN][CN][CN]:
- [CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN]
- [CN][CN]graph's[CN][CN]node addgraph_idfeature
- [CN][CN][CN][CN]node[CN]edge,keepnode[CN][CN][CN][CN][CN][CN]
- [CN][CN]node[CN]edge[CN]featureinformation,[CN][CN][CN]edge labely[CN]global labelglobal_y
- save[CN][CN][CN][CN][CN][CN]

Author: EDA for AI Team
date: 2024

[CN][CN][CN][CN]:
    python homograph_merge_with_edge_labels.py
"""

import os
import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional
import glob
from tqdm import tqdm

def load_homograph_files(input_dir: str) -> List[Tuple[str, Data]]:
    """[CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN]
    
    Args:
        input_dir:  input[CN][CN][CN][CN]
        
    Returns:
        List[Tuple[str, Data]]: (filename, homograph data) [CN] list
    """
    print(f"[SEARCH] [CN][CN][CN][CN]: {input_dir}")
    
    # [CN][CN][CN][CN].pt[CN][CN]
    pattern = os.path.join(input_dir, "*.pt")
    pt_files = glob.glob(pattern)
    
    # [CN][CN][CN][CN][CN][CN][CN]([CN]_homograph.pt[CN][CN])
    homograph_files = [f for f in pt_files if f.endswith('_homograph.pt')]
    
    print(f"[FILE] [CN][CN] {len(homograph_files)} [CN][CN][CN][CN][CN]")
    
    loaded_graphs = []
    for i, file_path in enumerate(sorted(homograph_files)):
        try:
            print(f"📖 [CN][CN]: {os.path.basename(file_path)}")
            if i == 0:
                print(f"\n[SEARCH] [CN][CN][CN][CN]: {os.path.basename(file_path)}")
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            
            #  extract[CN][CN][CN][CN]([CN][CN]_homograph.pt[CN][CN])
            design_name = os.path.basename(file_path).replace('_homograph.pt', '')
            loaded_graphs.append((design_name, data))
            
        except Exception as e:
            print(f"[!]️  warning: [CN][CN][CN][CN] {file_path}: {e}")
            continue
    
    print(f"[OK] success[CN][CN] {len(loaded_graphs)} [CN][CN][CN]")
    return loaded_graphs

def add_graph_id_to_nodes(data: Data, graph_id: int) -> Data:
    """[CN][CN][CN]graph's[CN][CN]node addsubgraph IDgraph_idfeature([CN][CN][CN][CN][CN])
    
    Args:
        data: [CN][CN]homograph data
        graph_id: subgraph ID(0-29)
        
    Returns:
        Data: [CN][CN][CN]9th dimension isgraph_id[CN]homograph data,keep[CN][CN][CN][CN][CN][CN]
    """
    print(f"🏷️  for subgraph {graph_id} [CN][CN]9th dimension isgraph_id...")
    
    # [CN][CN][CN][CN] data
    new_data = Data()
    
    # Process node features,[CN][CN][CN][CN]([CN][CN]8)[CN][CN][CN]graph_id
    if hasattr(data, 'x') and data.x is not None:
        original_features = data.x.clone()
        num_nodes = original_features.shape[0]
        feature_dim = original_features.shape[1]
        
        # ensurefeature dimension[CN][CN][CN]10[CN]
        if feature_dim < 10:
            # if dimension insufficient10,[CN][CN][CN][CN][CN]10[CN]
            padding = torch.zeros((num_nodes, 10 - feature_dim), dtype=torch.float32)
            original_features = torch.cat([original_features, padding], dim=1)
            print(f"  📏 feature dimension[CN] {feature_dim} [CN][CN][CN] 10")
        
        # [CN][CN][CN][CN]([CN][CN]8)[CN][CN][CN]graph_id
        original_features[:, 8] = graph_id
        new_data.x = original_features  # keep[CN][CN][CN][CN]:x
        
        print(f"  [CHART] [CN] {num_nodes} node[CN][CN][CN][CN][CN]for subgraph[CN][CN]: {graph_id}")
        print(f"  📏 nodefeature dimensionkeep: {new_data.x.shape[1]} (keep[CN][CN][CN][CN]x)")
    
    # [CN][CN]edge[CN][CN]
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        new_data.edge_index = data.edge_index.clone()
        print(f"  🔗 [CN][CN]edge[CN][CN]: {data.edge_index.shape}")
    
    # [CN][CN]edgefeature
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        new_data.edge_attr = data.edge_attr.clone()  # keep[CN][CN][CN][CN]:edge_attr
        print(f"  🔗 [CN][CN]edgefeature: {data.edge_attr.shape} (keep[CN][CN][CN][CN]edge_attr)")
    
    # [CN][CN]edge label - [CN][CN][CN][CN]！
    if hasattr(data, 'edge_label') and data.edge_label is not None:
        new_data.y = data.edge_label.clone()
        print(f"  🏷️  [CN][CN]edge label(edge_label->y): {data.edge_label.shape}")
    elif hasattr(data, 'y') and data.y is not None:
        new_data.y = data.y.clone()
        print(f"  🏷️  [CN][CN]edge label(y): {data.y.shape}")
    
    # [CN][CN]global label -  new[CN][CN]！
    if hasattr(data, 'global_y') and data.global_y is not None:
        new_data.global_y = data.global_y.clone()
        print(f"  [GLOBE] [CN][CN]global label: {data.global_y.shape}")
    
    # [CN][CN]node label([CN][CN][CN][CN][CN][CN][CN]edge label)
    if hasattr(data, 'node_y') and data.node_y is not None:
        new_data.node_y = data.node_y.clone()
        print(f"  🏷️  [CN][CN]node label: {data.node_y.shape}")
    
    # [CN][CN]globalfeature([CN][CN][CN][CN])
    if hasattr(data, 'global_features') and data.global_features is not None:
        new_data.global_features = data.global_features.clone()
        print(f"  [GLOBE] [CN][CN]globalfeature: {data.global_features.shape}")
    
    # copy die coordinates([CN][CN][CN][CN])
    if hasattr(data, 'die_coordinates') and data.die_coordinates is not None:
        new_data.die_coordinates = data.die_coordinates.clone()
        print(f"  📍 copy die coordinates: {data.die_coordinates.shape}")
    
    return new_data

def merge_homographs_with_edge_labels(graph_list: List[Tuple[str, Data]]) -> Data:
    """[CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN],[CN][CN]edge label[CN]global label
    
    Args:
        graph_list: ([CN][CN][CN][CN], homograph data) [CN] list
        
    Returns:
        Data: [CN][CN][CN][CN][CN][CN],keep[CN][CN][CN][CN][CN][CN]
    """
    print(f"🔗 [CN][CN][CN][CN] {len(graph_list)} [CN][CN][CN]...")
    
    if not graph_list:
        raise ValueError("no homographs to merge")
    
    # [CN][CN][CN] addsubgraph IDgraph_idand collect
    processed_graphs = []
    design_names = []
    
    for i, (design_name, data) in enumerate(tqdm(graph_list, desc="process subgraph")):
        print(f"🏷️   process[CN] {i+1}/{len(graph_list)}: {design_name}")
        processed_data = add_graph_id_to_nodes(data, i)
        processed_graphs.append(processed_data)
        design_names.append(design_name)
    
    #  create[CN][CN][CN][CN][CN][CN][CN]
    merged_graph = Data()
    
    # [CN][CN][CN]graph'snode[CN][CN][CN]
    node_offsets = []
    current_offset = 0
    for data in processed_graphs:
        node_offsets.append(current_offset)
        if hasattr(data, 'x') and data.x is not None:
            current_offset += data.x.shape[0]
    
    print(f"[CHART] node[CN][CN][CN]: {node_offsets}")
    
    # [CN][CN]nodefeature
    print("\n🔗 [CN][CN]nodefeature...")
    node_features = []
    node_labels = []
    
    for data in processed_graphs:
        if hasattr(data, 'x') and data.x is not None:
            node_features.append(data.x)
        
        if hasattr(data, 'node_y') and data.node_y is not None:
            node_labels.append(data.node_y)
    
    if node_features:
        merged_graph.x = torch.cat(node_features, dim=0)
        print(f"  [CHART] [CN][CN]nodefeature: {merged_graph.x.shape[0]} node, {merged_graph.x.shape[1]} [CN]feature ( containsgraph_id)")
    
    if node_labels:
        merged_graph.node_y = torch.cat(node_labels, dim=0)
        print(f"  🏷️  [CN][CN]node label: {merged_graph.node_y.shape}")
    
    # [CN][CN]edge[CN][CN],edgefeature[CN]edge label
    print("\n🔗 [CN][CN]edge[CN][CN],edgefeature[CN]edge label...")
    edge_indices = []
    edge_attrs = []  # edgefeature
    edge_labels = []
    
    for i, data in enumerate(processed_graphs):
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            # get originaledge[CN][CN]
            edge_index = data.edge_index.clone()
            
            #  addnode[CN][CN][CN]
            edge_index += node_offsets[i]
            
            edge_indices.append(edge_index)
            
            print(f"  🔗 [CN] {i}: {edge_index.shape[1]} entriesedge, [CN][CN][CN]: {node_offsets[i]}")
        
        # Process edge features
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attrs.append(data.edge_attr)
        
        # Process edge labels - [CN][CN]part！
        if hasattr(data, 'y') and data.y is not None:
            edge_labels.append(data.y)
            if i == 0:  # [CN][CN][CN][CN][CN][CN][CN][CN][CN][CN]information
                print(f"    📋 [CN]{i}: yshape: {data.y.shape}")
        elif hasattr(data, 'edge_label') and data.edge_label is not None:
            edge_labels.append(data.edge_label)
            if i == 0:  # [CN][CN][CN][CN][CN][CN][CN][CN][CN][CN]information
                print(f"    📋 [CN]{i}: edge_labelshape: {data.edge_label.shape}")
    
    if edge_indices:
        merged_graph.edge_index = torch.cat(edge_indices, dim=1)
        print(f"  [CHART] [CN][CN]edge[CN][CN]: {merged_graph.edge_index.shape}")
    
    if edge_attrs:
        merged_graph.edge_attr = torch.cat(edge_attrs, dim=0)  # edgefeature
        print(f"  [CHART] [CN][CN]edgefeature(edge_attr): {merged_graph.edge_attr.shape}")
    
    # [CN][CN]edge label - [CN][CN]！
    if edge_labels:
        merged_graph.y = torch.cat(edge_labels, dim=0)
        print(f"  [OK] [CN][CN]edge label: {merged_graph.y.shape}")
        print(f"    [TARGET] edge label[CN][CN][CN][CN][CN]！")
    
    # [CN][CN]globalfeature[CN]global label
    print("\n[GLOBE] [CN][CN]globalfeature[CN]global label...")
    global_features_list = []
    global_labels_list = []  #  new:global label list
    die_coordinates_list = []
    
    for i, data in enumerate(processed_graphs):
        # [CN][CN]globalfeature
        if hasattr(data, 'global_features') and data.global_features is not None:
            # [CN][CN][CN]graph'sglobalfeature addsubgraph ID
            global_feat = data.global_features.clone()
            # [CN]globalfeature[CN] addsubgraph ID([CN][CN][CN][CN][CN][CN])
            global_feat_with_id = torch.cat([global_feat, torch.tensor([i], dtype=torch.float32)])
            global_features_list.append(global_feat_with_id)
            print(f"  [GLOBE] [CN][CN] {i} globalfeature: {global_feat.shape} -> {global_feat_with_id.shape}")
        else:
            # [CN][CN][CN][CN]globalfeature, create default[CN][CN][CN][CN][CN] addgraph_id
            default_global_features = torch.zeros(5)  # [CN][CN][CN][CN][CN][CN] dimension
            default_global_features_with_id = torch.cat([default_global_features, torch.tensor([i], dtype=torch.float32)])
            global_features_list.append(default_global_features_with_id)
            print(f"  [GLOBE] [CN][CN] {i}  defaultglobalfeature: {default_global_features.shape} -> {default_global_features_with_id.shape}")
        
        # [CN][CN]global label -  new[CN][CN]！
        if hasattr(data, 'global_y') and data.global_y is not None:
            # [CN]global label[CN] addsubgraph ID([CN][CN][CN][CN][CN][CN])
            global_y_with_id = torch.cat([data.global_y.clone(), torch.tensor([i], dtype=torch.float32)])
            global_labels_list.append(global_y_with_id)
            print(f"  🏷️  [CN][CN] {i} global label: {data.global_y.shape} -> {global_y_with_id.shape}")
        else:
            # [CN][CN][CN][CN]global label, create default label[CN] addgraph_id
            default_global_label = torch.tensor([0.0])  #  defaultglobal label(changed to1[CN]tensor)
            default_global_label_with_id = torch.cat([default_global_label, torch.tensor([i], dtype=torch.float32)])
            global_labels_list.append(default_global_label_with_id)
            print(f"  🏷️  [CN][CN] {i}  defaultglobal label: {default_global_label.shape} -> {default_global_label_with_id.shape}")
        
        # [CN][CN][CN][CN][CN][CN]
        if hasattr(data, 'die_coordinates') and data.die_coordinates is not None:
            die_coordinates_list.append(data.die_coordinates)
        else:
            # [CN][CN][CN][CN][CN][CN][CN][CN], create default[CN][CN]
            default_coordinates = torch.zeros(2, 2)
            die_coordinates_list.append(default_coordinates)
    
    # [CN][CN]globalfeature
    if global_features_list:
        merged_graph.global_features = torch.stack(global_features_list, dim=0)
        print(f"  [OK] globalfeature: {merged_graph.global_features.shape} ( containsgraph_id)")
    
    # [CN][CN]global label -  new[CN][CN]！
    if global_labels_list:
        merged_graph.global_y = torch.stack(global_labels_list, dim=0)
        print(f"  [OK] global label: {merged_graph.global_y.shape} ( containsgraph_id)")
        print(f"    [TARGET] global label[CN][CN][CN][CN][CN]！")
    
    # [CN][CN][CN][CN][CN][CN]
    if die_coordinates_list:
        merged_graph.die_coordinates = torch.stack(die_coordinates_list, dim=0)
        print(f"  [OK] [CN][CN][CN][CN]: {merged_graph.die_coordinates.shape}")
    
    #  add[CN]information
    merged_graph.design_names = design_names
    merged_graph.num_subgraphs = len(processed_graphs)
    merged_graph.name = f"merged_homograph_{len(processed_graphs)}_designs_edge_task"
    
    print(f"\n[OK] [CN][CN]completed！")
    return merged_graph

def save_merged_graph(merged_graph: Data, output_path: str):
    """save[CN][CN][CN][CN][CN][CN][CN]
    
    Args:
        merged_graph: [CN][CN][CN][CN][CN][CN][CN]
        output_path:  output[CN][CN][CN][CN]
    """
    print(f"[SAVE] save[CN][CN][CN][CN]: {output_path}")
    torch.save(merged_graph, output_path, pickle_protocol=4)
    
    # [CN][CN][CN][CN]statisticsinformation
    print(f"\n📈 [CN][CN][CN]statisticsinformation:")
    
    if hasattr(merged_graph, 'x'):
        total_nodes = merged_graph.x.shape[0]
        feature_dim = merged_graph.x.shape[1]
        print(f"  - [CN]node[CN]: {total_nodes:,}")
        print(f"  - nodefeature dimension(x): {feature_dim} ( containsgraph_id)")
        
        # [CN][CN]graph_id[CN][CN]([CN][CN][CN],[CN][CN]8)
        graph_ids = merged_graph.x[:, 8]  # 9th dimension isgraph_id
        min_id = int(graph_ids.min().item())
        max_id = int(graph_ids.max().item())
        print(f"  - graph_id [CN][CN]: {min_id} - {max_id}")
    
    if hasattr(merged_graph, 'edge_index'):
        total_edges = merged_graph.edge_index.shape[1]
        print(f"  - [CN]edge[CN]: {total_edges:,}")
    
    if hasattr(merged_graph, 'edge_attr'):
        edge_feature_dim = merged_graph.edge_attr.shape[1]
        print(f"  - edgefeature dimension(edge_attr): {edge_feature_dim}")
    
    # [CN][CN]edge labelinformation - [CN][CN]！
    if hasattr(merged_graph, 'y'):
        print(f"  - [OK] edge label(y): {merged_graph.y.shape}")
        print(f"    [TARGET] edge label[CN][CN][CN][CN][CN]！")
        
        # [CN][CN]edge label[CN]statisticsinformation
        unique_labels = torch.unique(merged_graph.y)
        print(f"    [CHART] [CN][CN]edge label[CN][CN]: {len(unique_labels)}")
        print(f"    [CHART] edge label[CN][CN]: {merged_graph.y.min().item():.1f} - {merged_graph.y.max().item():.1f}")
    else:
        print(f"  - [X] edge label: [CN]")
    
    # [CN][CN]global labelinformation -  new！
    if hasattr(merged_graph, 'global_y'):
        print(f"  - [OK] global label(global_y): {merged_graph.global_y.shape} ( containsgraph_id)")
        print(f"    [TARGET] global label[CN][CN][CN][CN][CN]！")
        
        # [CN][CN]global label[CN]statisticsinformation
        unique_global_labels = torch.unique(merged_graph.global_y)
        print(f"    [CHART] [CN][CN]global label[CN][CN]: {len(unique_global_labels)}")
        print(f"    [CHART] global label[CN][CN]: {merged_graph.global_y.min().item():.1f} - {merged_graph.global_y.max().item():.1f}")
    else:
        print(f"  - [X] global label: [CN]")
    
    # [CN][CN]globalfeatureinformation
    if hasattr(merged_graph, 'global_features'):
        print(f"  - globalfeature: {merged_graph.global_features.shape} ( containsgraph_id)")
    
    if hasattr(merged_graph, 'die_coordinates'):
        print(f"  - [CN][CN][CN][CN]: {merged_graph.die_coordinates.shape}")
    
    print(f"  - [CN][CN][CN][CN]: {merged_graph.num_subgraphs}")
    
    # [CN][CN][CN][CN]graph'snode[CN][CN][CN][CN]
    if hasattr(merged_graph, 'x'):
        graph_ids = merged_graph.x[:, 8]  # 9th dimension isgraph_id
        for i in range(merged_graph.num_subgraphs):
            count = (graph_ids == i).sum().item()
            design_name = merged_graph.design_names[i] if hasattr(merged_graph, 'design_names') else f"graph_{i}"
            print(f"    * {design_name}: {count:,} node")
    
    # [CN][CN][CN][CN]information
    print(f"\n🔧  data[CN][CN]information:")
    print(f"  - nodefeature: x (keep[CN][CN][CN][CN])")
    print(f"  - edgefeature: edge_attr (keep[CN][CN][CN][CN])")
    print(f"  - edge label: y (keep[CN][CN])")
    print(f"  - global label: global_y ( new[CN][CN])")

def merge_directory(input_dir: str, output_path: str):
    """[CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN]
    
    Args:
        input_dir:  input[CN][CN][CN][CN]
        output_path:  output[CN][CN][CN][CN]
    """
    print(f"[START] [CN][CN][CN][CN][CN][CN]: {input_dir}")
    print("=" * 60)
    
    try:
        # 1. [CN][CN][CN][CN][CN][CN][CN][CN][CN]
        graph_list = load_homograph_files(input_dir)
        
        if not graph_list:
            print("[X] error: [CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN]")
            return False
        
        # 2. [CN][CN][CN][CN][CN]
        merged_graph = merge_homographs_with_edge_labels(graph_list)
        
        # 3. save[CN][CN][CN][CN]
        save_merged_graph(merged_graph, output_path)
        
        print(f"\n🎉 [CN][CN] {os.path.basename(input_dir)} [CN][CN]completed！")
        print(f"[FILE]  output[CN][CN]: {output_path}")
        return True
        
    except Exception as e:
        print(f"[X] [CN][CN][CN][CN][CN][CN][CN]error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function - [CN][CN]D,E[CN]F[CN][CN][CN]global label[CN][CN]"""
    print("[START] homograph merge script started - global label integrated version")
    print("=" * 80)
    
    #  configuration[CN][CN] - [CN][CN]D,E[CN]F[CN][CN]
    base_dir = "/Users/david/Desktop/benchmark_dataset/place_homo_v1.4/to_homogeneous"
    directories = ['D', 'E', 'F']  # [CN][CN]D,E[CN]F[CN][CN]
    
    success_count = 0
    
    for dir_name in directories:
        input_dir = os.path.join(base_dir, dir_name)
        output_path = os.path.join(base_dir, f"place_{dir_name}_homograph.pt")
        
        print(f"\n{'='*20}  process[CN][CN] {dir_name} {'='*20}")
        
        if merge_directory(input_dir, output_path):
            success_count += 1
        else:
            print(f"[X] [CN][CN] {dir_name} [CN][CN]failed")
    
    print(f"\n{'='*80}")
    print(f"🎉 [CN][CN][CN][CN]completed！success: {success_count}/{len(directories)}")
    print(f"🔧 [CN][CN][CN][CN]:")
    print(f"  - [OK] global label(global_y)[CN][CN],last dimension containsgraph_id")
    print(f"  - [OK] globalfeature(global_features)[CN][CN],last dimension containsgraph_id")
    print(f"  - [OK] keep[CN][CN][CN][CN][CN][CN]: x(nodefeature), edge_attr(edgefeature)")
    print(f"  - [OK] graph_id[CN][CN][CN][CN][CN](nodefeature)")
    print(f"  - [OK] edge label[CN]global label[CN][CN][CN][CN]")
    
    return 0 if success_count == len(directories) else 1

if __name__ == "__main__":
    exit(main())