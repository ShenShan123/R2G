import torch
import os
from pathlib import Path
from datetime import datetime
import sys

# Node type mapping table - consistent with hetero_to_homo_converter.py
NODE_TYPE_TO_ID = {
    "gate": 0,
    "io_pin": 1,
    "net": 2,
    "pin": 3
}

# Edge type mapping table - consistent with hetero_to_homo_converter.py
EDGE_TYPE_TO_ID = {
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

class LogWriter:
    """Log writer that outputs to both console and file"""
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.console = sys.stdout
    
    def write(self, message):
        self.console.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure real-time write
    
    def flush(self):
        self.console.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

def analyze_heterograph(hetero_data, graph_name):
    """
    Analyze the node types and edge types of a single heterograph

    Args:
        hetero_data: Loaded heterograph data (HeteroData)
        graph_name: Name of the graph
    """
    print(f"\n{'='*70}")
    print(f"Analyzing heterograph: {graph_name}")
    print(f"{'='*70}")

    try:
        # Get node types and edge types
        node_types = hetero_data.node_types
        edge_types = hetero_data.edge_types

        print(f"Heterograph basic information:")
        print(f"  Number of node types: {len(node_types)}")
        print(f"  Number of edge types: {len(edge_types)}")
        print(f"  Node types: {node_types}")
        print(f"  Edge types: {edge_types}")

        # Count total nodes and edges
        total_nodes = sum(hetero_data[node_type].num_nodes for node_type in node_types)
        total_edges = sum(hetero_data[edge_type].num_edges for edge_type in edge_types)

        print(f"  Total nodes: {total_nodes}")
        print(f"  Total edges: {total_edges}")

        # Analyze node type statistics - display all defined node types
        print(f"\nNode type count statistics:")
        print(f"{'-'*40}")

        # Display all node types in the order defined in the mapping table
        for node_type, type_id in sorted(NODE_TYPE_TO_ID.items(), key=lambda x: x[1]):
            if node_type in node_types:
                num_nodes = hetero_data[node_type].num_nodes
                percentage = (num_nodes / total_nodes) * 100 if total_nodes > 0 else 0

                print(f"  {node_type:>8} (ID={type_id}): {num_nodes:>8} ({percentage:>5.1f}%)")

                # Display node feature information
                if hasattr(hetero_data[node_type], 'x') and hetero_data[node_type].x is not None:
                    feature_shape = hetero_data[node_type].x.shape
                    print(f"    └─ Feature dimension: {feature_shape}")
                else:
                    print(f"    └─ No node features")
            else:
                # Display non-existent node types as 0
                print(f"  {node_type:>8} (ID={type_id}): {0:>8} ({0:>5.1f}%)")

        print(f"  {'Total':>8}: {total_nodes:>8} (100.0%)")

        # Analyze edge type statistics - display all defined edge types
        print(f"\nEdge type count statistics:")
        print(f"{'-'*50}")

        # Create unique edge type mapping (avoid duplicate IDs)
        unique_edge_types = {}
        for edge_type_str, type_id in EDGE_TYPE_TO_ID.items():
            if type_id not in unique_edge_types:
                unique_edge_types[type_id] = edge_type_str

        # Display all edge types in ID order
        for type_id in sorted(unique_edge_types.keys()):
            edge_type_str = unique_edge_types[type_id]

            # Parse edge type string
            try:
                # Remove outer brackets and quotes, then split
                edge_type_clean = edge_type_str.strip("()").replace("'", "")
                parts = [part.strip() for part in edge_type_clean.split(',')]
                if len(parts) == 3:
                    src, rel, dst = parts
                    edge_tuple = (src, rel, dst)

                    # Use edge type names consistent with internal heterograph
                    edge_display = f"({src}, {rel}, {dst})"

                    # Check if corresponding edge type exists (including merged has and has_pin cases)
                    found_edge = None
                    for et in edge_types:
                        et_str = str(et)
                        if EDGE_TYPE_TO_ID.get(et_str) == type_id:
                            found_edge = et
                            break

                    if found_edge:
                        num_edges = hetero_data[found_edge].num_edges
                        percentage = (num_edges / total_edges) * 100 if total_edges > 0 else 0

                        print(f"  {edge_display:>30} (ID={type_id}): {num_edges:>8} ({percentage:>5.1f}%)")

                        # Display edge feature information (if any)
                        if hasattr(hetero_data[found_edge], 'edge_attr') and hetero_data[found_edge].edge_attr is not None:
                            edge_attr_shape = hetero_data[found_edge].edge_attr.shape
                            print(f"    └─ Edge feature dimension: {edge_attr_shape}")
                        else:
                            print(f"    └─ No edge features")
                    else:
                        # Display non-existent edge types as 0
                        print(f"  {edge_display:>30} (ID={type_id}): {0:>8} ({0:>5.1f}%)")

            except Exception as e:
                print(f"  Error parsing edge type {edge_type_str}: {e}")

        print(f"  {'Total':>30}: {total_edges:>8} (100.0%)")

        # Check global feature information
        print(f"\nGlobal feature information:")
        print(f"{'-'*30}")

        has_global_features = False

        if hasattr(hetero_data, 'global_features') and hetero_data.global_features is not None:
            global_shape = hetero_data.global_features.shape
            print(f"  Global features:")
            print(f"    Count: {global_shape[0]}")
            print(f"    Dimension: {global_shape[1] if len(global_shape) > 1 else 1}")
            print(f"    Shape: {global_shape}")
            has_global_features = True

        if hasattr(hetero_data, 'global_y') and hetero_data.global_y is not None:
            global_y_shape = hetero_data.global_y.shape
            print(f"  Global labels:")
            print(f"    Count: {global_y_shape[0]}")
            print(f"    Dimension: {global_y_shape[1] if len(global_y_shape) > 1 else 1}")
            print(f"    Shape: {global_y_shape}")
            has_global_features = True

        if not has_global_features:
            print(f"  No global features")

        # Display unmapped types (if any)
        print(f"\nType mapping check:")
        print(f"{'-'*30}")

        # Check unmapped node types
        unmapped_nodes = [nt for nt in node_types if nt not in NODE_TYPE_TO_ID]
        if unmapped_nodes:
            print(f"  Unmapped node types: {unmapped_nodes}")
        else:
            print(f"  All node types are mapped ✓")

        # Check unmapped edge types
        unmapped_edges = [str(et) for et in edge_types if str(et) not in EDGE_TYPE_TO_ID]
        if unmapped_edges:
            print(f"  Unmapped edge types: {unmapped_edges}")
        else:
            print(f"  All edge types are mapped ✓")

    except Exception as e:
        print(f"Error analyzing heterograph {graph_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function: analyze all large heterographs
    """
    # Create log directory
    log_dir = Path("/Users/david/Desktop/route1.3/check_log")
    log_dir.mkdir(exist_ok=True)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M")
    log_filename = f"{timestamp}_route_heterographs.txt"
    log_path = log_dir / log_filename

    # Create log writer
    log_writer = LogWriter(log_path)

    # Redirect standard output to log writer
    original_stdout = sys.stdout
    sys.stdout = log_writer

    try:
        # Set base path
        base_dir = Path("/Users/david/Desktop/route1.3/route_v1.3")

        # List of heterograph files to analyze
        graph_files = [
            "RB_merged_heterograph.pt",
            "RC_merged_heterograph.pt",
            "RD_merged_heterograph.pt",
            "RE_merged_heterograph.pt",
            "RF_merged_heterograph.pt"
        ]

        print("Starting analysis of all large heterographs...")
        print(f"Base directory: {base_dir}")
        print(f"Log file: {log_path}")
        print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nNode type mapping table:")
        for node_type, type_id in NODE_TYPE_TO_ID.items():
            print(f"  {node_type} -> {type_id}")

        print(f"\nEdge type mapping table:")
        for edge_type, type_id in EDGE_TYPE_TO_ID.items():
            print(f"  {edge_type} -> {type_id}")

        # Analyze each heterograph file
        for graph_file in graph_files:
            graph_path = base_dir / graph_file

            if graph_path.exists():
                try:
                    # Load heterograph data
                    print(f"\nLoading: {graph_file}")
                    hetero_data = torch.load(graph_path, map_location='cpu', weights_only=False)

                    # Analyze heterograph type statistics
                    analyze_heterograph(hetero_data, graph_file)

                except Exception as e:
                    print(f"Error loading file {graph_file}: {e}")
            else:
                print(f"\nWarning: File {graph_file} does not exist, skipping analysis")

        print(f"\n{'='*70}")
        print("All heterograph analysis completed!")
        print(f"{'='*70}")
        print(f"Analysis results saved to: {log_path}")

    finally:
        # Restore standard output
        sys.stdout = original_stdout
        log_writer.close()
        print(f"Analysis completed, log saved to: {log_path}")

if __name__ == "__main__":
    main()