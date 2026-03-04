#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA Flow Data Extraction Tool

Features:
1. Automatically extract config.mk configuration parameters
2. Parse key information from log files
3. Analyze performance metrics from report files
4. Generate structured datasets

Author: EDA for AI Team
Date: January 2025
"""

import os
import re
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EDADataExtractor:
    """EDA flow data extractor"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.data = {
            'designs': [],
            'configs': {},
            'synthesis_stats': {},
            'placement_stats': {},
            'routing_stats': {},
            'timing_stats': {}
        }
    
    def extract_config_data(self, config_path: str) -> Dict[str, Any]:
        """Extract configuration parameters from config.mk file"""
        config_data = {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract key configuration parameters
            patterns = {
                'design_name': r'export DESIGN_NAME\s*=\s*([^\s]+)',
                'design_nickname': r'export DESIGN_NICKNAME\s*=\s*([^\s]+)',
                'platform': r'export PLATFORM\s*=\s*([^\s]+)',
                'core_utilization': r'export CORE_UTILIZATION\s*\??=\s*(\d+)',
                'place_density': r'export PLACE_DENSITY_LB_ADDON\s*=\s*([\d\.]+)',
                'tns_end_percent': r'export TNS_END_PERCENT\s*=\s*(\d+)',
                'abc_area': r'export ABC_AREA\s*=\s*(\d+)',
                'synth_hierarchical': r'export SYNTH_HIERARCHICAL\s*=\s*(\d+)',
                'skip_gate_cloning': r'export SKIP_GATE_CLONING\s*=\s*(\d+)',
                'enable_dpo': r'export ENABLE_DPO\s*=\s*(\d+)',
                'rt_clock_min_layer': r'export RT_CLOCK_MIN_LAYER\s*=\s*(\d+)',
                'rt_clock_max_layer': r'export RT_CLOCK_MAX_LAYER\s*=\s*(\d+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    value = match.group(1)
                    # Try to convert to number
                    try:
                        config_data[key] = float(value) if '.' in value else int(value)
                    except ValueError:
                        config_data[key] = value

            # Extract Verilog file count
            verilog_pattern = r'export VERILOG_FILES\s*=\s*([^\n]+(?:\\\n[^\n]+)*)'
            verilog_match = re.search(verilog_pattern, content, re.MULTILINE)
            if verilog_match:
                verilog_content = verilog_match.group(1)
                # Count .v files
                v_files = len(re.findall(r'\.v\s', verilog_content))
                config_data['verilog_file_count'] = v_files

            logger.info(f"Successfully extracted config file: {config_path}")

        except Exception as e:
            logger.error(f"Failed to extract config file {config_path}: {e}")
        
        return config_data
    
    def extract_synthesis_stats(self, synth_stat_path: str) -> Dict[str, Any]:
        """Extract synthesis statistics"""
        stats = {}
        
        try:
            with open(synth_stat_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract basic statistics
            patterns = {
                'num_wires': r'Number of wires:\s*(\d+)',
                'num_wire_bits': r'Number of wire bits:\s*(\d+)',
                'num_public_wires': r'Number of public wires:\s*(\d+)',
                'num_ports': r'Number of ports:\s*(\d+)',
                'num_port_bits': r'Number of port bits:\s*(\d+)',
                'num_cells': r'Number of cells:\s*(\d+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    stats[key] = int(match.group(1))
            
            # Extract standard cell statistics
            cell_pattern = r'(\w+_X\d+)\s+(\d+)'
            cell_matches = re.findall(cell_pattern, content)
            
            cell_stats = {}
            total_cells = 0
            for cell_type, count in cell_matches:
                count = int(count)
                cell_stats[cell_type] = count
                total_cells += count
            
            stats['cell_distribution'] = cell_stats
            stats['total_std_cells'] = total_cells
            
            # Classify statistics
            logic_cells = 0
            sequential_cells = 0
            buffer_cells = 0
            
            for cell_type, count in cell_stats.items():
                if any(x in cell_type for x in ['NAND', 'NOR', 'AND', 'OR', 'AOI', 'INV']):
                    logic_cells += count
                elif any(x in cell_type for x in ['DFF', 'LATCH']):
                    sequential_cells += count
                elif any(x in cell_type for x in ['BUF', 'CLKBUF']):
                    buffer_cells += count
            
            stats['logic_cells'] = logic_cells
            stats['sequential_cells'] = sequential_cells
            stats['buffer_cells'] = buffer_cells
            
            if total_cells > 0:
                stats['logic_ratio'] = logic_cells / total_cells
                stats['sequential_ratio'] = sequential_cells / total_cells
                stats['buffer_ratio'] = buffer_cells / total_cells
            
            logger.info(f"Successfully extracted synthesis statistics: {synth_stat_path}")
            
        except Exception as e:
            logger.error(f"Failed to extract synthesis statistics {synth_stat_path}: {e}")
        
        return stats
    
    def extract_placement_stats(self, placement_rpt_path: str) -> Dict[str, Any]:
        """Extract placement statistics"""
        stats = {}
        
        try:
            with open(placement_rpt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract placement statistics
            patterns = {
                'total_instances': r'Total Instances:\s*(\d+)',
                'placed_instances': r'Placed Instances:\s*(\d+)',
                'fixed_instances': r'Fixed Instances:\s*(\d+)',
                'placement_completion': r'Placement Completion:\s*([\d\.]+)%',
                'die_area': r'Die Area:\s*([\d\.]+)\s*um',
                'core_area': r'Core Area:\s*([\d\.]+)\s*um',
                'instance_area': r'Instance Total Area:\s*([\d\.]+)\s*um',
                'utilization': r'Utilization:\s*([\d\.]+)%',
                'total_hpwl': r'Total HPWL:\s*([\d\.]+)\s*um',
                'avg_hpwl': r'Average HPWL per Net:\s*([\d\.]+)\s*um',
                'nets_analyzed': r'Nets Analyzed:\s*(\d+)',
                'clock_networks': r'Total Clock Networks:\s*(\d+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    value = match.group(1)
                    try:
                        stats[key] = float(value)
                    except ValueError:
                        stats[key] = value
            
            logger.info(f"Successfully extracted placement statistics: {placement_rpt_path}")
            
        except Exception as e:
            logger.error(f"Failed to extract placement statistics {placement_rpt_path}: {e}")
        
        return stats
    
    def extract_timing_stats(self, timing_rpt_path: str) -> Dict[str, Any]:
        """Extract timing statistics"""
        stats = {}
        
        try:
            with open(timing_rpt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract timing metrics
            patterns = {
                'tns': r'tns\s+([\d\.-]+)',
                'wns': r'wns\s+([\d\.-]+)',
                'worst_slack': r'worst slack\s+([\d\.-]+)',
                'setup_skew': r'([\d\.-]+)\s+setup skew'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    try:
                        stats[key] = float(match.group(1))
                    except ValueError:
                        stats[key] = match.group(1)
            
            logger.info(f"Successfully extracted timing statistics: {timing_rpt_path}")
            
        except Exception as e:
            logger.error(f"Failed to extract timing statistics {timing_rpt_path}: {e}")
        
        return stats
    
    def extract_routing_stats(self, routing_log_path: str) -> Dict[str, Any]:
        """Extract routing statistics"""
        stats = {}
        
        try:
            with open(routing_log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract routing statistics
            patterns = {
                'num_layers': r'Number of layers:\s*(\d+)',
                'num_macros': r'Number of macros:\s*(\d+)',
                'num_vias': r'Number of vias:\s*(\d+)',
                'num_components': r'Number of components:\s*(\d+)',
                'num_terminals': r'Number of terminals:\s*(\d+)',
                'num_nets': r'Number of nets:\s*(\d+)',
                'die_area_x': r'Die area:\s*\(\s*\d+\s+\d+\s*\)\s*\(\s*(\d+)',
                'die_area_y': r'Die area:\s*\(\s*\d+\s+\d+\s*\)\s*\(\s*\d+\s+(\d+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    try:
                        stats[key] = int(match.group(1))
                    except ValueError:
                        stats[key] = match.group(1)
            
            # Extract large fanout net warnings
            large_net_pattern = r'Large net (\w+) has (\d+) pins'
            large_nets = re.findall(large_net_pattern, content)
            if large_nets:
                stats['large_nets'] = [(net, int(pins)) for net, pins in large_nets]
                stats['max_fanout'] = max(int(pins) for _, pins in large_nets)
            
            logger.info(f"Successfully extracted routing statistics: {routing_log_path}")
            
        except Exception as e:
            logger.error(f"Failed to extract routing statistics {routing_log_path}: {e}")
        
        return stats
    
    def scan_project(self) -> None:
        """Scan the entire project and extract all data"""
        logger.info(f"Starting to scan project: {self.project_root}")
        
        # Scan config.mk files in RTL_Sources directory
        rtl_sources = self.project_root / "RTL_Sources" / "RTL and config"
        if rtl_sources.exists():
            for config_file in rtl_sources.rglob("config.mk"):
                design_path = config_file.parent.parent
                design_name = design_path.name
                
                config_data = self.extract_config_data(str(config_file))
                if config_data:
                    config_data['design_path'] = str(design_path)
                    self.data['configs'][design_name] = config_data
        
        # Extract statistics for current project
        # Synthesis statistics
        synth_stat_file = self.project_root / "report" / "synth_stat.txt"
        if synth_stat_file.exists():
            self.data['synthesis_stats'] = self.extract_synthesis_stats(str(synth_stat_file))
        
        # Placement statistics
        placement_rpt_file = self.project_root / "report" / "3_placement_report.rpt"
        if placement_rpt_file.exists():
            self.data['placement_stats'] = self.extract_placement_stats(str(placement_rpt_file))
        
        # Timing statistics
        timing_rpt_file = self.project_root / "report" / "6_finish.rpt"
        if timing_rpt_file.exists():
            self.data['timing_stats'] = self.extract_timing_stats(str(timing_rpt_file))
        
        # Routing statistics
        routing_log_file = self.project_root / "log" / "5_2_route.log"
        if routing_log_file.exists():
            self.data['routing_stats'] = self.extract_routing_stats(str(routing_log_file))
        
        logger.info("Project scan completed")
    
    def export_to_json(self, output_path: str) -> None:
        """Export data to JSON format"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            logger.info(f"Data exported to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
    
    def export_to_csv(self, output_dir: str) -> None:
        """Export data to CSV format"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Export configuration data
            if self.data['configs']:
                with open(output_dir / "config_data.csv", 'w', newline='', encoding='utf-8') as f:
                    if self.data['configs']:
                        all_keys = set()
                        for config in self.data['configs'].values():
                            all_keys.update(config.keys())
                        
                        writer = csv.DictWriter(f, fieldnames=['design_name'] + sorted(all_keys))
                        writer.writeheader()
                        
                        for design_name, config in self.data['configs'].items():
                            row = {'design_name': design_name}
                            row.update(config)
                            writer.writerow(row)
            
            # ExportSynthesis statistics
            if self.data['synthesis_stats']:
                with open(output_dir / "synthesis_stats.csv", 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.data['synthesis_stats'].keys())
                    writer.writeheader()
                    writer.writerow(self.data['synthesis_stats'])
            
            # ExportPlacement statistics
            if self.data['placement_stats']:
                with open(output_dir / "placement_stats.csv", 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.data['placement_stats'].keys())
                    writer.writeheader()
                    writer.writerow(self.data['placement_stats'])
            
            # ExportTiming statistics
            if self.data['timing_stats']:
                with open(output_dir / "timing_stats.csv", 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.data['timing_stats'].keys())
                    writer.writeheader()
                    writer.writerow(self.data['timing_stats'])
            
            # ExportRouting statistics
            if self.data['routing_stats']:
                with open(output_dir / "routing_stats.csv", 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.data['routing_stats'].keys())
                    writer.writeheader()
                    writer.writerow(self.data['routing_stats'])
            
            logger.info(f"CSV files exported to: {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate data summary"""
        summary = {
            'total_designs': len(self.data['configs']),
            'config_parameters': list(set().union(*[config.keys() for config in self.data['configs'].values()])) if self.data['configs'] else [],
            'synthesis_available': bool(self.data['synthesis_stats']),
            'placement_available': bool(self.data['placement_stats']),
            'timing_available': bool(self.data['timing_stats']),
            'routing_available': bool(self.data['routing_stats'])
        }
        
        # Calculate configuration parameter distribution
        if self.data['configs']:
            param_stats = {}
            for param in summary['config_parameters']:
                values = [config.get(param) for config in self.data['configs'].values() if param in config]
                if values:
                    if all(isinstance(v, (int, float)) for v in values):
                        param_stats[param] = {
                            'min': min(values),
                            'max': max(values),
                            'avg': sum(values) / len(values),
                            'count': len(values)
                        }
                    else:
                        param_stats[param] = {
                            'unique_values': list(set(values)),
                            'count': len(values)
                        }
            summary['parameter_statistics'] = param_stats
        
        return summary
 
    def extract_placement_data(self) -> List[Dict[str, Any]]:
        """Extract placement-specific data: CORE_UTILIZATION and HPWL"""
        placement_data = []
        
        # Scan config directory
        config_dir = self.project_root / "config"
        if not config_dir.exists():
            logger.error(f"Config directory does not exist: {config_dir}")
            return placement_data
        
        for design_folder in config_dir.iterdir():
            if design_folder.is_dir():
                design_name = design_folder.name
                data_entry = {'design_name': design_name}
                
                # Extract CORE_UTILIZATION from config.mk
                config_file = design_folder / "config.mk"
                if config_file.exists():
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Extract CORE_UTILIZATION
                        util_pattern = r'export CORE_UTILIZATION\s*=\s*(\d+)'
                        util_match = re.search(util_pattern, content)
                        if util_match:
                            data_entry['core_utilization'] = int(util_match.group(1))
                        else:
                            data_entry['core_utilization'] = None
                            logger.warning(f"CORE_UTILIZATION not found: {config_file}")
                    
                    except Exception as e:
                        logger.error(f"Failed to read config file {config_file}: {e}")
                        data_entry['core_utilization'] = None
                else:
                    logger.warning(f"Config file does not exist: {config_file}")
                    data_entry['core_utilization'] = None
                
                # Extract HPWL from placement logs
                # Handle folders with possible spaces in names
                log_dir = self.project_root / "log" / design_name
                if not log_dir.exists():
                    # Try to find folders with spaces in names
                    log_base_dir = self.project_root / "log"
                    for folder in log_base_dir.iterdir():
                        if folder.is_dir() and folder.name.strip() == design_name:
                            log_dir = folder
                            break
                
                if log_dir.exists():
                    # Look for placement-related log files
                    placement_log_files = [
                        "3_5_place_dp.log",
                        "3_place.log",
                        "placement.log"
                    ]
                    
                    hpwl_found = False
                    for log_filename in placement_log_files:
                        log_file = log_dir / log_filename
                        if log_file.exists():
                            try:
                                with open(log_file, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                
                                # Extract HPWL information
                                hpwl_pattern = r'HPWL after\s+([\d\.]+)\s*u'
                                hpwl_matches = re.findall(hpwl_pattern, content)
                                if hpwl_matches:
                                    # Take the last HPWL value (usually the final value)
                                    data_entry['hpwl_after'] = float(hpwl_matches[-1])
                                    hpwl_found = True
                                
                                # Extract design area and utilization information
                                area_pattern = r'Design area\s+([\d\.]+)\s*u\^2\s+([\d\.]+)%\s+utilization'
                                area_match = re.search(area_pattern, content)
                                if area_match:
                                    data_entry['design_area'] = float(area_match.group(1))
                                    data_entry['design_utilization'] = float(area_match.group(2))
                                
                                if hpwl_matches or area_match:
                                    break
                            
                            except Exception as e:
                                logger.error(f"Failed to read log file {log_file}: {e}")
                    
                    if not hpwl_found:
                        data_entry['hpwl_after'] = None
                        logger.warning(f"HPWL information not found: {design_name}")
                else:
                    logger.warning(f"Log directory does not exist: {log_dir}")
                    data_entry['hpwl_after'] = None
                
                placement_data.append(data_entry)
                logger.info(f"Processing design: {design_name} - CORE_UTILIZATION: {data_entry['core_utilization']}, HPWL: {data_entry['hpwl_after']}")
        
        return placement_data
    
    def export_placement_data_to_csv(self, output_path: str) -> None:
        """Export placement data to CSV file"""
        placement_data = self.extract_placement_data()
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if placement_data:
                    fieldnames = ['design_name', 'core_utilization', 'hpwl_after', 'design_area', 'design_utilization']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for data in placement_data:
                        writer.writerow(data)
                    
                    logger.info(f"Placement data exported to: {output_path}")
                    print(f"\nSuccessfully exported placement data for {len(placement_data)} designs to: {output_path}")
                else:
                    logger.warning("No placement data found")
        
        except Exception as e:
            logger.error(f"Failed to export placement data CSV: {e}")

def main():
    """Main function"""
    project_root = "/Users/david/Desktop/EDA_data_sorting"
    
    # Create data extractor
    extractor = EDADataExtractor(project_root)
    
    # Scan project
    extractor.scan_project()
    
    # Generate summary
    summary = extractor.generate_summary()
    print("\n=== Data Extraction Summary ===")
    print(f"Total designs: {summary['total_designs']}")
    print(f"Configuration parameters: {len(summary['config_parameters'])}")
    print(f"Synthesis data: {'Available' if summary['synthesis_available'] else 'Unavailable'}")
    print(f"Placement data: {'Available' if summary['placement_available'] else 'Unavailable'}")
    print(f"Timing data: {'Available' if summary['timing_available'] else 'Unavailable'}")
    print(f"Routing data: {'Available' if summary['routing_available'] else 'Unavailable'}")
    
    # Export data
    output_dir = Path(project_root) / "extracted_data"
    output_dir.mkdir(exist_ok=True)
    
    extractor.export_to_json(str(output_dir / "eda_data.json"))
    extractor.export_to_csv(str(output_dir))
    
    # Export placement data
    extractor.export_placement_data_to_csv(str(output_dir / "place_data_extract.csv"))
    
    # Save summary
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nData exported to: {output_dir}")

if __name__ == "__main__":
    main()