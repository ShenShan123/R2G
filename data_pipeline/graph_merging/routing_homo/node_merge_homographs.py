#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[CN][CN][CN][CN][CN][CN][CN]

[CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN],referenceheterograph merge script[CN][CN][CN][CN][CN]:
- [CN][CN][CN]graph's[CN][CN]nodegenerategraph_idas identification[CN][CN] for different graphs
- preserve allnode[CN]edge[CN]featureand label
- [CN][CN][CN][CN]node[CN]edge,keepnode[CN][CN][CN][CN][CN][CN]
- ensure data integrity and correctness

[CN][CN][CN][CN]:
- [CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN]
- [CN][CN]graph's[CN][CN]node addgraph_idfeature
- [CN][CN][CN][CN]node[CN]edge,keepnode[CN][CN][CN][CN][CN][CN]
- [CN][CN]node[CN]edge[CN]featureinformation
- save[CN][CN][CN][CN][CN][CN]

Author: EDA for AI Team
date: 2024

[CN][CN][CN][CN]:
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
    for file_path in sorted(homograph_files):
        try:
            print(f"📖 [CN][CN]: {os.path.basename(file_path)}")
            data = torch.load(file_path, weights_only=False)
            
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
        Data: [CN][CN][CN]9th dimension isgraph_id[CN]homograph data
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
        new_data.x = original_features
        
        print(f"  [CHART] [CN] {num_nodes} node[CN][CN][CN][CN][CN]for subgraph[CN][CN]: {graph_id}")
        print(f"  📏 nodefeature dimensionkeep: {new_data.x.shape[1]}")
    
    # [CN][CN]edge[CN][CN]
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        new_data.edge_index = data.edge_index.clone()
        print(f"  🔗 [CN][CN]edge[CN][CN]: {data.edge_index.shape}")
    
    # [CN][CN]edgefeature
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        new_data.edge_attr = data.edge_attr.clone()
        print(f"  🔗 [CN][CN]edgefeature: {data.edge_attr.shape}")
    
    # [CN][CN]node label([CN][CN][CN][CN])
    if hasattr(data, 'y') and data.y is not None:
        new_data.y = data.y.clone()
        print(f"  🏷️  [CN][CN]node label: {data.y.shape}")
    
    # Preserve global features[CN] label
    global_attrs = ['global_features', 'global_y', 'die_coordinates']
    for attr_name in global_attrs:
        if hasattr(data, attr_name):
            attr_value = getattr(data, attr_name)
            if torch.is_tensor(attr_value):
                setattr(new_data, attr_name, attr_value.clone())
                print(f"  [GLOBE] Preserve global features {attr_name}: {attr_value.shape}")
    
    # [CN][CN][CN][CN][CN][CN]([CN][CN][CN] process[CN][CN][CN][CN][CN][CN][CN])
    skip_attrs = ['x', 'edge_index', 'edge_attr', 'y'] + global_attrs
    for attr_name in dir(data):
        if not attr_name.startswith('_') and attr_name not in skip_attrs:
            try:
                attr_value = getattr(data, attr_name)
                if torch.is_tensor(attr_value):
                    setattr(new_data, attr_name, attr_value.clone())
                    print(f"  📋 [CN][CN][CN][CN] {attr_name}: {attr_value.shape}")
                elif not callable(attr_value):
                    setattr(new_data, attr_name, attr_value)
                    print(f"  📋 [CN][CN][CN][CN] {attr_name}: {type(attr_value)}")
            except:
                pass  # [CN][CN][CN][CN][CN][CN][CN][CN][CN]
    
    return new_data

def merge_homographs(graph_list: List[Tuple[str, Data]]) -> Data:
    """[CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN],Preserve global features[CN] label
    
    Args:
        graph_list: ([CN][CN][CN][CN], homograph data) [CN] list
        
    Returns:
        Data: [CN][CN][CN][CN][CN][CN], containsglobalfeature[CN] label
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
        
        if hasattr(data, 'y') and data.y is not None:
            node_labels.append(data.y)
    
    if node_features:
        merged_graph.x = torch.cat(node_features, dim=0)
        print(f"  [CHART] [CN][CN]nodefeature: {merged_graph.x.shape[0]} node, {merged_graph.x.shape[1]} [CN]feature")
    
    if node_labels:
        merged_graph.y = torch.cat(node_labels, dim=0)
        print(f"  🏷️  [CN][CN]node label: {merged_graph.y.shape}")
    
    # [CN][CN]edge[CN][CN][CN]edgefeature
    print("\n🔗 [CN][CN]edge[CN][CN][CN]edgefeature...")
    edge_indices = []
    edge_attrs = []
    
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
    
    if edge_indices:
        merged_graph.edge_index = torch.cat(edge_indices, dim=1)
        print(f"  [CHART] [CN][CN]edge[CN][CN]: {merged_graph.edge_index.shape}")
    
    if edge_attrs:
        merged_graph.edge_attr = torch.cat(edge_attrs, dim=0)
        print(f"  [CHART] [CN][CN]edgefeature: {merged_graph.edge_attr.shape}")
    
    # [CN][CN]globalfeature[CN] label
    print("\n[GLOBE] [CN][CN]globalfeature[CN] label...")
    global_features_list = []
    global_y_list = []
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
        
        # [CN][CN]global label
        if hasattr(data, 'global_y') and data.global_y is not None:
            # [CN]global label[CN] addsubgraph ID([CN][CN][CN][CN][CN][CN])
            global_y_with_id = torch.cat([data.global_y.clone(), torch.tensor([i], dtype=torch.float32)])
            global_y_list.append(global_y_with_id)
            print(f"  🏷️  [CN][CN] {i} global label: {data.global_y.shape} -> {global_y_with_id.shape}")
        
        # [CN][CN][CN][CN][CN][CN]
        if hasattr(data, 'die_coordinates') and data.die_coordinates is not None:
            die_coordinates_list.append(data.die_coordinates.clone())
            print(f"  📍 [CN][CN] {i} [CN][CN][CN][CN]: {data.die_coordinates.shape}")
    
    # [CN][CN]globalfeature
    if global_features_list:
        merged_graph.global_features = torch.stack(global_features_list, dim=0)
        print(f"  [CHART] [CN][CN]globalfeature: {merged_graph.global_features.shape}")
    
    if global_y_list:
        merged_graph.global_y = torch.stack(global_y_list, dim=0)
        print(f"  [CHART] [CN][CN]global label: {merged_graph.global_y.shape}")
    
    if die_coordinates_list:
        merged_graph.die_coordinates = torch.stack(die_coordinates_list, dim=0)
        print(f"  [CHART] merge die coordinates: {merged_graph.die_coordinates.shape}")
    
    #  add[CN]information
    merged_graph.design_names = design_names
    merged_graph.num_subgraphs = len(processed_graphs)
    merged_graph.name = f"merged_homograph_{len(processed_graphs)}_designs"
    
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
        print(f"  - nodefeature dimension: {feature_dim} (9th dimension isgraph_id)")
        
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
        print(f"  - edgefeature dimension: {edge_feature_dim}")
    
    # [CN][CN]globalfeatureinformation
    if hasattr(merged_graph, 'global_features'):
        print(f"  - globalfeature: {merged_graph.global_features.shape}")
    
    if hasattr(merged_graph, 'global_y'):
        print(f"  - global label: {merged_graph.global_y.shape}")
    
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
        merged_graph = merge_homographs(graph_list)
        
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
    """Main function -  processB[CN]C[CN][CN]"""
    print("[START] homograph merge script started -  processB[CN]C[CN][CN]")
    print("=" * 80)
    
    #  configuration[CN][CN] -  processB[CN]C[CN][CN]
    base_dir = "/Users/david/Desktop/route1.3/to_homogeneous"
    output_dir = "/Users/david/Desktop/route1.3/route_homo_v1.3"
    directories = ['B', 'C']  #  processB[CN]C[CN][CN]
    
    success_count = 0
    
    for dir_name in directories:
        input_dir = os.path.join(base_dir, dir_name)
        output_path = os.path.join(output_dir, f"route_{dir_name}_homograph.pt")
        
        print(f"\n{'='*20}  process[CN][CN] {dir_name} {'='*20}")
        
        if merge_directory(input_dir, output_path):
            success_count += 1
        else:
            print(f"[X] [CN][CN] {dir_name} [CN][CN]failed")
    
    print(f"\n{'='*80}")
    print(f"🎉 [CN][CN][CN][CN]completed！success: {success_count}/{len(directories)}")
    print(f"🔧 [CN][CN]: graph_idgenerate,nodefeature[CN][CN],edgefeature[CN][CN]")
    
    return 0 if success_count == len(directories) else 1

if __name__ == "__main__":
    exit(main())