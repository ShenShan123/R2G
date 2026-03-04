
import torch
import os
from pathlib import Path
from datetime import datetime
import sys

# Node type constant definitions - consistent with hetero_to_homo_converter.py
NODE_TYPES = {
    'gate': 0,
    'io_pin': 1,
    'net': 2,
    'pin': 3
}

# Edge type constant definitions - consistent with hetero_to_homo_converter.py
EDGE_TYPES = {
    'gate_gate': 0,           # (gate, connects_to, gate)
    'gate_net': 1,            # (gate, connects_to, net)
    'gate_pin': 2,            # (gate, has, pin)
    'io_pin_gate': 3,         # (io_pin, connects_to, gate)
    'io_pin_net': 4,          # (io_pin, connects_to, net)
    'io_pin_pin': 5,          # (io_pin, connects_to, pin)
    'pin_net': 6,             # (pin, connects_to, net)
    'pin_pin': 7,             # (pin, connects_to, pin)
    'pin_gate_pin': 8         # (pin, gate_connects, pin)
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

def analyze_graph_types(graph_data, graph_name):
    """
    Analyze the node types and edge types of a single graph

    Args:
        graph_data: Loaded graph data
        graph_name: Name of the graph
    """
    print(f"\n{'='*60}")
    print(f"Analyzing graph: {graph_name}")
    print(f"{'='*60}")

    try:
        # Get node features and edge attributes
        x = graph_data.x  # Node features
        edge_attr = graph_data.edge_attr  # Edge attributes

        print(f"Graph basic information:")
        print(f"  Node count: {x.shape[0]}")
        print(f"  Edge count: {edge_attr.shape[0]}")
        print(f"  Node feature dimension: {x.shape[1]}")
        print(f"  Edge feature dimension: {edge_attr.shape[1]}")

        # Analyze node types (10th dimension, index 9)
        node_types = x[:, 9].long()  # 10th dimension is node_type

        # Count and print each node type
        type_names = {v: k for k, v in NODE_TYPES.items()}  # id -> name
        unique_types, type_counts = torch.unique(node_types, return_counts=True)

        print(f"\nNode type count statistics:")
        print(f"{'-'*40}")

        # First output by known type IDs (from NODE_TYPES), ensuring 0, 1, 2, 3 are printed even if count is 0
        total_nodes = 0
        for t in sorted(type_names.keys()):
            count_t = int((node_types == t).sum().item())
            total_nodes += count_t
            percentage = (count_t / x.shape[0]) * 100 if x.shape[0] > 0 else 0
            print(f"  {type_names.get(t):>8} (ID={t}): {count_t:>8} ({percentage:>5.1f}%)")

        # If there are unknown type IDs (not in NODE_TYPES), also output them
        known_ids = set(type_names.keys())
        for t, c in zip(unique_types.tolist(), type_counts.tolist()):
            if t not in known_ids:
                percentage = (c / x.shape[0]) * 100 if x.shape[0] > 0 else 0
                print(f"  {'unknown':>8} (ID={t}): {c:>8} ({percentage:>5.1f}%)")
                total_nodes += c

        print(f"  {'Total':>8}: {total_nodes:>8} (100.0%)")

        # Analyze edge types (10th dimension, index 9)
        edge_types = edge_attr[:, 9].long()
        edge_type_names = {v: k for k, v in EDGE_TYPES.items()}  # id -> name
        unique_edge_types, edge_type_counts = torch.unique(edge_types, return_counts=True)

        print(f"\nEdge type count statistics:")
        print(f"{'-'*40}")

        # First print by known edge type IDs (even if count is 0)
        total_edges = 0
        for et_id in sorted(edge_type_names.keys()):
            count_et = int((edge_types == et_id).sum().item())
            total_edges += count_et
            percentage = (count_et / edge_attr.shape[0]) * 100 if edge_attr.shape[0] > 0 else 0
            print(f"  {edge_type_names.get(et_id):>12} (ID={et_id}): {count_et:>8} ({percentage:>5.1f}%)")

        # If there are unknown edge type IDs (not in EDGE_TYPES), also output them
        known_edge_ids = set(edge_type_names.keys())
        for et_id, c in zip(unique_edge_types.tolist(), edge_type_counts.tolist()):
            if et_id not in known_edge_ids:
                percentage = (c / edge_attr.shape[0]) * 100 if edge_attr.shape[0] > 0 else 0
                print(f"  {'unknown':>12} (ID={et_id}): {c:>8} ({percentage:>5.1f}%)")
                total_edges += c

        print(f"  {'Total':>12}: {total_edges:>8} (100.0%)")

        # Check for global features
        if hasattr(graph_data, 'global_features') and graph_data.global_features is not None:
            print(f"\nGlobal features:")
            print(f"  Shape: {graph_data.global_features.shape}")

        if hasattr(graph_data, 'global_y') and graph_data.global_y is not None:
            print(f"  Global label shape: {graph_data.global_y.shape}")

    except Exception as e:
        print(f"Error analyzing graph {graph_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function: analyze all merged large graphs
    """
    # Create log directory
    log_dir = Path("/Users/david/Desktop/route1.3/check_log")
    log_dir.mkdir(exist_ok=True)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M")
    log_filename = f"{timestamp}_route_homographs.txt"
    log_path = log_dir / log_filename

    # Create log writer
    log_writer = LogWriter(log_path)

    # Redirect standard output to log writer
    original_stdout = sys.stdout
    sys.stdout = log_writer

    try:
        # Set base path
        base_dir = Path("/Users/david/Desktop/route1.3/route_homo_v1.3")

        # List of graph files to analyze
        graph_files = [
            "route_B_homograph.pt",
            "route_C_homograph.pt",
            "route_D_homograph.pt",
            "route_E_homograph.pt",
            "route_F_homograph.pt"
        ]

        print("Starting analysis of all merged large graphs...")
        print(f"Base directory: {base_dir}")
        print(f"Log file: {log_path}")
        print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Analyze each graph file
        for graph_file in graph_files:
            graph_path = base_dir / graph_file

            if graph_path.exists():
                try:
                    # Load graph data
                    print(f"\nLoading: {graph_file}")
                    graph_data = torch.load(graph_path, map_location='cpu', weights_only=False)

                    # Analyze graph type statistics
                    analyze_graph_types(graph_data, graph_file)

                except Exception as e:
                    print(f"Error loading file {graph_file}: {e}")
            else:
                print(f"\nWarning: File {graph_file} does not exist, skipping analysis")

        print(f"\n{'='*60}")
        print("All graph analysis completed!")
        print(f"{'='*60}")
        print(f"Analysis results saved to: {log_path}")

    finally:
        # Restore standard output
        sys.stdout = original_stdout
        log_writer.close()
        print(f"Analysis completed, log saved to: {log_path}")

if __name__ == "__main__":
    main()