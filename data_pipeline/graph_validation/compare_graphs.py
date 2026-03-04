#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heterograph and Homograph Consistency Analysis Script
Used to verify consistency between heterographs and homographs of the same design in terms of node and edge type counts

Author: EDA for AI Team
"""

import re
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

class GraphDataParser:
    """Graph data parser for parsing statistics from log files"""
    
    def __init__(self):
        # Built-in mapping table - consistent with hetero_to_homo_converter.py
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

        # Homograph edge type mapping (simplified names to standard names)
        self.homo_edge_mapping = {
            "gate_gate": "(gate, connects_to, gate)",
            "gate_net": "(gate, connects_to, net)",
            "gate_pin": "(gate, has, pin)",
            "io_pin_gate": "(io_pin, connects_to, gate)",
            "io_pin_net": "(io_pin, connects_to, net)",
            "io_pin_pin": "(io_pin, connects_to, pin)",
            "pin_net": "(pin, connects_to, net)",
            "pin_pin": "(pin, connects_to, pin)",
            "pin_gate_pin": "(pin, gate_connects, pin)"
        }

    def parse_homograph_log(self, log_file_path):
        """Parse homograph log file"""
        graphs_data = {}
        
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content by graph
        graph_sections = re.split(r'analyzing graph: (route_[BCDEF]_homograph\.pt)', content)

        for i in range(1, len(graph_sections), 2):
            graph_name = graph_sections[i]
            graph_content = graph_sections[i + 1]

            # Extract design name (B, C, D, E, F)
            design_name = graph_name.split('_')[1]  # route_B_homograph.pt -> B

            # Parse node type statistics
            node_stats = {}
            node_section = re.search(r'nodetype countstatistics:(.*?)edgetype countstatistics:', graph_content, re.DOTALL)
            if node_section:
                node_lines = node_section.group(1).strip().split('\n')
                for line in node_lines:
                    # Match format: gate (ID=0): 739944 ( 19.8%)
                    match = re.search(r'(\w+)\s+\(ID=(\d+)\):\s+(\d+)\s+\([\d\s.]+%\)', line.strip())
                    if match:
                        node_type, type_id, count = match.groups()
                        if node_type != 'total':  # Skip total line
                            node_stats[node_type] = int(count)

            # Parse edge type statistics
            edge_stats = {}
            edge_section = re.search(r'edgetype countstatistics:(.*?)(?:globalfeature:|loading:|$)', graph_content, re.DOTALL)
            if edge_section:
                edge_lines = edge_section.group(1).strip().split('\n')
                for line in edge_lines:
                    # Match format: gate_gate (ID=0): 0 ( 0.0%)
                    match = re.search(r'(\w+(?:_\w+)*)\s+\(ID=(\d+)\):\s+(\d+)\s+\([\d\s.]+%\)', line.strip())
                    if match:
                        edge_type, type_id, count = match.groups()
                        if edge_type != 'total':  # Skip total line
                            # Convert to standard edge type name
                            standard_edge_type = self.homo_edge_mapping.get(edge_type, edge_type)
                            edge_stats[standard_edge_type] = int(count)

            graphs_data[design_name] = {
                'nodes': node_stats,
                'edges': edge_stats,
                'type': 'homograph'
            }
        
        return graphs_data

    def parse_heterograph_log(self, log_file_path):
        """Parse heterograph log file"""
        graphs_data = {}
        
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content by graph
        graph_sections = re.split(r'analyzing heterograph: (R[BCDEF]_merged_heterograph\.pt)', content)

        for i in range(1, len(graph_sections), 2):
            graph_name = graph_sections[i]
            graph_content = graph_sections[i + 1]

            # Extract design name (B, C, D, E, F)
            design_name = graph_name.split('_')[0][1:]  # Remove R prefix, get B, C, D, E, F

            # Parse node type statistics
            node_stats = {}
            node_section = re.search(r'nodetype countstatistics:(.*?)edgetype countstatistics:', graph_content, re.DOTALL)
            if node_section:
                node_lines = node_section.group(1).strip().split('\n')
                for line in node_lines:
                    # Match format: gate (ID=0): 739944 ( 19.8%)
                    match = re.search(r'(\w+)\s+\(ID=(\d+)\):\s+(\d+)\s+\([\d\s.]+%\)', line.strip())
                    if match:
                        node_type, type_id, count = match.groups()
                        if node_type != 'total':  # Skip total line
                            node_stats[node_type] = int(count)

            # Parse edge type statistics
            edge_stats = {}
            edge_section = re.search(r'edgetype countstatistics:(.*?)globalfeatureinformation:', graph_content, re.DOTALL)
            if edge_section:
                edge_lines = edge_section.group(1).strip().split('\n')
                for line in edge_lines:
                    # Match format: (gate, connects_to, gate) (ID=0): 0 ( 0.0%)
                    match = re.search(r'\(([^)]+)\)\s+\(ID=(\d+)\):\s+(\d+)\s+\([\d\s.]+%\)', line.strip())
                    if match:
                        edge_type_content, type_id, count = match.groups()
                        # Reconstruct edge type format
                        edge_type = f"({edge_type_content})"
                        if edge_type_content != 'total':  # Skip total line
                            edge_stats[edge_type] = int(count)

            graphs_data[design_name] = {
                'nodes': node_stats,
                'edges': edge_stats,
                'type': 'heterograph'
            }
        
        return graphs_data

class GraphConsistencyAnalyzer:
    """Graph consistency analyzer"""
    
    def __init__(self):
        self.parser = GraphDataParser()
        
    def find_latest_log_files(self, log_dir):
        """Find the latest log files"""
        log_path = Path(log_dir)

        # Find homograph log files
        homo_files = list(log_path.glob("*_route_homographs.txt"))
        homo_files.sort(key=lambda x: x.name, reverse=True)

        # Find heterograph log files
        hetero_files = list(log_path.glob("*_route_heterographs.txt"))
        hetero_files.sort(key=lambda x: x.name, reverse=True)

        return homo_files[0] if homo_files else None, hetero_files[0] if hetero_files else None
    
    def compare_graphs(self, homo_data, hetero_data):
        """Compare homograph and heterograph data"""
        comparison_results = {}

        # Get all design names
        all_designs = set(homo_data.keys()) | set(hetero_data.keys())

        for design in sorted(all_designs):
            homo_graph = homo_data.get(design, {})
            hetero_graph = hetero_data.get(design, {})

            result = {
                'design': design,
                'homo_exists': design in homo_data,
                'hetero_exists': design in hetero_data,
                'node_consistency': {},
                'edge_consistency': {},
                'issues': []
            }

            if not result['homo_exists']:
                result['issues'].append(f"Missing homograph data")
            if not result['hetero_exists']:
                result['issues'].append(f"Missing heterograph data")

            if result['homo_exists'] and result['hetero_exists']:
                # Compare node type counts
                homo_nodes = homo_graph.get('nodes', {})
                hetero_nodes = hetero_graph.get('nodes', {})

                all_node_types = set(homo_nodes.keys()) | set(hetero_nodes.keys())
                for node_type in all_node_types:
                    homo_count = homo_nodes.get(node_type, 0)
                    hetero_count = hetero_nodes.get(node_type, 0)

                    result['node_consistency'][node_type] = {
                        'homo_count': homo_count,
                        'hetero_count': hetero_count,
                        'consistent': homo_count == hetero_count,
                        'difference': homo_count - hetero_count
                    }

                    if homo_count != hetero_count:
                        result['issues'].append(f"Node type '{node_type}' count mismatch: homograph={homo_count}, heterograph={hetero_count}")

                # Compare edge type counts
                homo_edges = homo_graph.get('edges', {})
                hetero_edges = hetero_graph.get('edges', {})

                all_edge_types = set(homo_edges.keys()) | set(hetero_edges.keys())
                for edge_type in all_edge_types:
                    homo_count = homo_edges.get(edge_type, 0)
                    hetero_count = hetero_edges.get(edge_type, 0)

                    result['edge_consistency'][edge_type] = {
                        'homo_count': homo_count,
                        'hetero_count': hetero_count,
                        'consistent': homo_count == hetero_count,
                        'difference': homo_count - hetero_count
                    }

                    if homo_count != hetero_count:
                        result['issues'].append(f"Edge type '{edge_type}' count mismatch: homograph={homo_count}, heterograph={hetero_count}")

            comparison_results[design] = result

        return comparison_results
    
    def generate_report(self, comparison_results, output_file=None):
        """Generate analysis report"""
        report_lines = []

        # Report header
        report_lines.append("="*80)
        report_lines.append("Heterograph and Homograph Consistency Analysis Report")
        report_lines.append("="*80)
        report_lines.append(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Overall overview
        total_designs = len(comparison_results)
        consistent_designs = sum(1 for result in comparison_results.values() if not result['issues'])

        report_lines.append("Overall Overview:")
        report_lines.append(f"  Total designs: {total_designs}")
        report_lines.append(f"  Consistent designs: {consistent_designs}")
        report_lines.append(f"  Designs with issues: {total_designs - consistent_designs}")
        report_lines.append("")

        # Detailed analysis
        for design, result in sorted(comparison_results.items()):
            report_lines.append(f"Design {design}:")
            report_lines.append("-" * 40)

            if result['issues']:
                report_lines.append(f"  Status: ❌ Inconsistencies found")
                report_lines.append(f"  Number of issues: {len(result['issues'])}")
                report_lines.append("")

                # Node type analysis
                if result['node_consistency']:
                    report_lines.append("  Node type analysis:")
                    for node_type, stats in result['node_consistency'].items():
                        status = "✓" if stats['consistent'] else "✗"
                        report_lines.append(f"    {status} {node_type:>8}: homograph={stats['homo_count']:>8}, heterograph={stats['hetero_count']:>8}")
                        if not stats['consistent']:
                            report_lines.append(f"      Difference: {stats['difference']:+d}")
                    report_lines.append("")

                # Edge type analysis
                if result['edge_consistency']:
                    report_lines.append("  Edge type analysis:")
                    for edge_type, stats in result['edge_consistency'].items():
                        status = "✓" if stats['consistent'] else "✗"
                        # Simplify edge type display
                        display_name = edge_type.replace("('", "").replace("')", "").replace("', '", "_")
                        report_lines.append(f"    {status} {display_name:>30}: homograph={stats['homo_count']:>8}, heterograph={stats['hetero_count']:>8}")
                        if not stats['consistent']:
                            report_lines.append(f"      Difference: {stats['difference']:+d}")
                    report_lines.append("")

                # Issues list
                report_lines.append("  Specific issues:")
                for issue in result['issues']:
                    report_lines.append(f"    • {issue}")
            else:
                report_lines.append(f"  Status: ✅ Fully consistent")

                # Display statistics summary
                total_nodes_homo = sum(stats['homo_count'] for stats in result['node_consistency'].values())
                total_edges_homo = sum(stats['homo_count'] for stats in result['edge_consistency'].values())

                report_lines.append(f"  Total nodes: {total_nodes_homo}")
                report_lines.append(f"  Total edges: {total_edges_homo}")

            report_lines.append("")

        # Issues summary
        all_issues = []
        for result in comparison_results.values():
            all_issues.extend(result['issues'])

        if all_issues:
            report_lines.append("Issues Summary:")
            report_lines.append("-" * 40)

            # Group by issue type
            node_issues = [issue for issue in all_issues if "nodetype" in issue]
            edge_issues = [issue for issue in all_issues if "edgetype" in issue]
            other_issues = [issue for issue in all_issues if "nodetype" not in issue and "edgetype" not in issue]

            if node_issues:
                report_lines.append(f"  Node type issues ({len(node_issues)}):")
                for issue in node_issues:
                    report_lines.append(f"    • {issue}")
                report_lines.append("")

            if edge_issues:
                report_lines.append(f"  Edge type issues ({len(edge_issues)}):")
                for issue in edge_issues:
                    report_lines.append(f"    • {issue}")
                report_lines.append("")

            if other_issues:
                report_lines.append(f"  Other issues ({len(other_issues)}):")
                for issue in other_issues:
                    report_lines.append(f"    • {issue}")
        else:
            report_lines.append("🎉 All heterographs and homographs are fully consistent!")

        report_lines.append("")
        report_lines.append("="*80)

        # Output report
        report_content = "\n".join(report_lines)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"Analysis report saved to: {output_file}")

        return report_content

def main():
    """Main function"""
    # Set path
    log_dir = "/Users/david/Desktop/route1.3/check_log"

    # Create analyzer
    analyzer = GraphConsistencyAnalyzer()

    # Find latest log files
    homo_log, hetero_log = analyzer.find_latest_log_files(log_dir)

    if not homo_log:
        print("❌ Homograph log file not found")
        return

    if not hetero_log:
        print("❌ Heterograph log file not found")
        return

    print(f"📊 Starting graph data consistency analysis...")
    print(f"Homograph log: {homo_log.name}")
    print(f"Heterograph log: {hetero_log.name}")
    print()

    # Parse log files
    print("🔍 Parsing homograph log...")
    homo_data = analyzer.parser.parse_homograph_log(homo_log)

    print("🔍 Parsing heterograph log...")
    hetero_data = analyzer.parser.parse_heterograph_log(hetero_log)

    # Perform comparison analysis
    print("⚖️  Performing consistency comparison...")
    comparison_results = analyzer.compare_graphs(homo_data, hetero_data)

    # Generate report
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M")
    report_file = Path(log_dir) / f"{timestamp}_consistency_report.txt"

    print("📝 Generating analysis report...")
    report_content = analyzer.generate_report(comparison_results, report_file)

    # Display report in console
    print("\n" + report_content)

if __name__ == "__main__":
    main()