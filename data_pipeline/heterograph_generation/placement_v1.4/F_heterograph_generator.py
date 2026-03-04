#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F graph heterograph generator (Gate-Gate connection mode)

This script implements F graph transformation based on D_heterograph_generator.py with features:
1. Remove pin nodes, IO_Pin-Pin edges, Pin-Pin edges and gate-pin edges
2. Build new gate-gate edges with 6-dimensional edge features: [pin_type_id, cell_type_id] + [net_type_id, connection_count] + [pin_type_id, cell_type_id]
3. Build IO_Pin-gate edges with 4-dimensional edge features: [net_type_id, connection_count] + [pin_type_id, cell_type_id]

Transformation logic:
- For each net, identify output pins (e.g., ZN, Z, Q) as source nodes
- Connect the gate containing output pins to gates containing all input pins in the net
- Edge features include source gate pin information, net information, and target gate pin information

Author: EDA for AI Team
date: 2024

Usage:
    python F_heterograph_generator.py input.def -o output.pt
"""

import re
import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional
import argparse
import os
from dataclasses import dataclass, field
import traceback
from typing import Any

# ============================================================================
# DATA STRUCTURES SECTION - Data structure definition region
# ============================================================================

# Complete cell type mapping - global definition
COMPLETE_CELL_TYPE_MAPPING = {
    # Basic logic gate series
    'INV_X1': 0, 'INV_X2': 1, 'INV_X4': 2, 'INV_X8': 3, 'INV_X16': 4, 'INV_X32': 5,
    'BUF_X1': 6, 'BUF_X2': 7, 'BUF_X4': 8, 'BUF_X8': 9, 'BUF_X16': 10, 'BUF_X32': 11,
    'CLKBUF_X1': 12, 'CLKBUF_X2': 13, 'CLKBUF_X3': 14,
    'NAND2_X1': 15, 'NAND2_X2': 16, 'NAND2_X4': 17,
    'NAND3_X1': 18, 'NAND3_X2': 19, 'NAND3_X4': 20,
    'NAND4_X1': 21, 'NAND4_X2': 22, 'NAND4_X4': 23,
    'NOR2_X1': 24, 'NOR2_X2': 25, 'NOR2_X4': 26,
    'NOR3_X1': 27, 'NOR3_X2': 28, 'NOR3_X4': 29,
    'NOR4_X1': 30, 'NOR4_X2': 31, 'NOR4_X4': 32,
    'AND2_X1': 33, 'AND2_X2': 34, 'AND2_X4': 35,
    'AND3_X1': 36, 'AND3_X2': 37, 'AND3_X4': 38,
    'AND4_X1': 39, 'AND4_X2': 40, 'AND4_X4': 41,
    'OR2_X1': 42, 'OR2_X2': 43, 'OR2_X4': 44,
    'OR3_X1': 45, 'OR3_X2': 46, 'OR3_X4': 47,
    'OR4_X1': 48, 'OR4_X2': 49, 'OR4_X4': 50,
    'XOR2_X1': 51, 'XOR2_X2': 52,
    'XNOR2_X1': 53, 'XNOR2_X2': 54,
    'AOI21_X1': 55, 'AOI21_X2': 56, 'AOI21_X4': 57,
    'AOI22_X1': 58, 'AOI22_X2': 59, 'AOI22_X4': 60,
    'OAI21_X1': 61, 'OAI21_X2': 62, 'OAI21_X4': 63,
    'OAI22_X1': 64, 'OAI22_X2': 65, 'OAI22_X4': 66,
    'MUX2_X1': 67, 'MUX2_X2': 68,
    'FA_X1': 69, 'HA_X1': 70,
    'DFF_X1': 71, 'DFF_X2': 72,
    'DFFR_X1': 73, 'DFFR_X2': 74,
    'DFFS_X1': 75, 'DFFS_X2': 76,
    'DFFSR_X1': 77, 'DFFSR_X2': 78,
    'TBUF_X1': 79, 'TBUF_X2': 80, 'TBUF_X4': 81, 'TBUF_X8': 82, 'TBUF_X16': 83,
    'TINV_X1': 84, 'TINV_X2': 85,
    'FILLCELL_X1': 86, 'FILLCELL_X2': 87, 'FILLCELL_X4': 88, 'FILLCELL_X8': 89, 'FILLCELL_X16': 90, 'FILLCELL_X32': 91,
    'ANTENNA_X1': 92,
    'LOGIC0_X1': 93, 'LOGIC1_X1': 94,
    'UNKNOWN': 95
}

@dataclass
class ComponentInfo:
    """Component information data structure"""
    name: str                                   # Component instance name
    cell_type: str                              # Cell type (e.g., INV_X1, NAND2_X1, etc.)
    position: Tuple[float, float]               # Component position coordinates (x, y)(x, y)
    orientation: str = 'N'                      # Component orientation (N, S, E, W, FN, FS, FE, FW)
    pins: Dict[str, Any] = field(default_factory=dict)  # Component pin information dictionary
    size: float = 1.0                           # Component relative size (based on drive strength)
    
@dataclass
class NetInfo:
    """Net information data structure"""
    name: str                                   # Net name
    connections: List[Tuple[str, str]] = field(default_factory=list)  # Connection list: (component name, pin name)
    routing: List[Dict] = field(default_factory=list)                 # Routing information list
    net_type: int = 0                          # Net type encoding (0-signal, 1-power, 2-ground, 3-clock, etc.)
    weight: float = 1.0                        # Net weight (based on connection complexity)

@dataclass
class PinInfo:
    """Pin information data structure"""
    name: str                                   # Pin name
    component: str                              # Parent component name
    direction: str = 'INOUT'                    # Pin direction (INPUT, OUTPUT, INOUT, FEEDTHRU)
    layer: str = 'metal1'                       # Metal layer where pin is located
    position: Tuple[float, float] = (0.0, 0.0)  # Pin position coordinates
    net: str = ''                               # Connected net name

@dataclass
class GateGateEdge:
    """Gate-Gate edge information data structure"""
    source_gate: str                            # Source gate name
    target_gate: str                            # Target gate name
    source_pin_type: int                        # Source pin type encoding
    source_cell_type: int                       # Source cell type encoding
    net_type: int                              # Net type encoding
    connection_count: int                       # Connection count
    target_pin_type: int                        # Target pin type encoding
    target_cell_type: int                       # Target cell type encoding
    net_name: str = ''                         # Net name (for debugging)

@dataclass
class IO_PinGateEdge:
    """IO_Pin-Gate edge information data structure"""
    io_pin_name: str                           # IO pin name
    gate_name: str                             # Gate name
    net_type: int                              # Net type encoding (net_type_id)
    connection_count: int                       # Connection count
    gate_pin_type: int                         # Gate pin type encoding (pin_type_id)
    gate_cell_type: int                        # Gate cell type encoding (cell_type_id)
    net_name: str = ''                         # Net name (for debugging)
    net_hpwl: float = 0.0                      # Net HPWL label (corresponding to D graph's IO Pin-Pin edge label)

# ============================================================================
# DEF PARSER SECTION - DEF file parser region (reusing D graph script parser)
# ============================================================================

class DEFParser:
    """DEF file parser
    
    Responsible for parsing DEF files and extracting structured data, including:
    - Design information (name, units, die area)
    - Component information (position, type, orientation)
    - Pin information (IO pin position and orientation)
    - Net information (connections)
    - Layer information (metal layers and via layers)
    - Row information (standard cell rows)
    """
    
    def __init__(self, def_file_path: str):
        self.def_file_path = def_file_path
        self.def_data = {
            'design_name': '',
            'units': 1000,
            'die_area': (0, 0, 0, 0),
            'components': {},
            'pins': {},
            'nets': {},
            'internal_pins': {},  # new: internal pin information
            'tracks': {},
            'vias': {},
            'rows': {}
        }
    
    def parse(self) -> Dict:
        """Parse DEF file and return structured data"""
        try:
            with open(self.def_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse each section
            self._parse_design_info(content)
            self._parse_components(content)
            self._parse_pins(content)
            self._parse_nets(content)
            self._parse_tracks(content)
            self._parse_vias(content)
            self._parse_rows(content)
            
            return self.def_data
            
        except Exception as e:
            print(f"[X] DEF file parsing  failed: {e}")
            raise
    
    def _parse_design_info(self, content: str):
        """Parse basic design information"""
        # Parse design name
        design_match = re.search(r'DESIGN\s+(\w+)', content)
        if design_match:
            self.def_data['design_name'] = design_match.group(1)
        
        # Parse units
        units_match = re.search(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', content)
        if units_match:
            self.def_data['units'] = {
                'dbu_per_micron': int(units_match.group(1))
            }
        
        # Parse die area
        diearea_match = re.search(r'DIEAREA\s+\(\s*([\d\-]+)\s+([\d\-]+)\s*\)\s+\(\s*([\d\-]+)\s+([\d\-]+)\s*\)', content)
        if diearea_match:
            self.def_data['die_area'] = tuple(map(int, diearea_match.groups()))
    
    def _parse_components(self, content: str):
        """Parse component information"""
        components_section = re.search(r'COMPONENTS\s+(\d+)\s*;(.*?)END\s+COMPONENTS', content, re.DOTALL)
        if not components_section:
            return
        
        components_text = components_section.group(2)
        component_pattern = r'-\s+([^\s]+)\s+(\w+)\s+.*?\+\s+PLACED\s+\(\s*([\d\-]+)\s+([\d\-]+)\s*\)\s+(\w+)'
        
        for match in re.finditer(component_pattern, components_text):
            comp_name = match.group(1)
            cell_type = match.group(2)
            x_pos = int(match.group(3))
            y_pos = int(match.group(4))
            orientation = match.group(5)
            
            self.def_data['components'][comp_name] = {
                'cell_type': cell_type,
                'position': (x_pos, y_pos),
                'orientation': orientation
            }
    
    def _parse_pins(self, content: str):
        """Parse IO pin information - Extract real position coordinates and layer information"""
        pins_section = re.search(r'PINS\s+(\d+)\s*;(.*?)END\s+PINS', content, re.DOTALL)
        if not pins_section:
            return
        
        pins_text = pins_section.group(2)
        
        # Use more precise regex to split pin definitions
        # Each pin definition starts with "- pin_name" and ends with ";"
        pin_pattern = r'-\s+([\w\[\]_]+)\s+\+\s+NET\s+([\w\[\]_]+)\s+\+\s+DIRECTION\s+(\w+).*?;'
        
        for match in re.finditer(pin_pattern, pins_text, re.DOTALL):
            pin_name = match.group(1)
            net_name = match.group(2)
            direction = match.group(3)
            pin_def = match.group(0)
            
            # Parse PORT section layer information
            layer_match = re.search(r'\+\s+LAYER\s+(\w+)', pin_def)
            layer = layer_match.group(1) if layer_match else 'metal1'
            
            # Parse PLACED position information
            position_match = re.search(r'\+\s+PLACED\s+\(\s*([\d\-]+)\s+([\d\-]+)\s*\)', pin_def)
            if position_match:
                x_pos = int(position_match.group(1))
                y_pos = int(position_match.group(2))
                position = (x_pos, y_pos)
            else:
                position = (0, 0)  # Default position
            
            # Parse USE information (optional)
            use_match = re.search(r'\+\s+USE\s+(\w+)', pin_def)
            use_type = use_match.group(1) if use_match else 'SIGNAL'
            
            self.def_data['pins'][pin_name] = {
                'net': net_name,
                'direction': direction,
                'position': position,
                'layer': layer,
                'use': use_type
            }
    
    def _parse_nets(self, content: str):
        """Parse net information and internal pin information"""
        nets_section = re.search(r'NETS\s+(\d+)\s*;(.*?)END\s+NETS', content, re.DOTALL)
        if not nets_section:
            return
        
        nets_text = nets_section.group(2)
        # Improved net parsing - supports complex net names and connection formats
        # Match format: - net_name ( comp1 pin1 ) ( comp2 pin2 ) ... + USE SIGNAL ;
        net_pattern = r'-\s+([\w\[\]_$\\]+)\s+((?:\([^)]+\)\s*)+)'
        
        for match in re.finditer(net_pattern, nets_text):
            net_name = match.group(1)
            connections_text = match.group(2)
            
            connections = []
            # Match each connection: ( component_name pin_name )
            conn_pattern = r'\(\s*([^\s]+)\s+([^\s]+)\s*\)'
            for conn_match in re.finditer(conn_pattern, connections_text):
                comp_name = conn_match.group(1)
                pin_name = conn_match.group(2)
                connections.append((comp_name, pin_name))
                
                # Extract internal pin information
                self._extract_internal_pin(comp_name, pin_name, net_name)
            
            self.def_data['nets'][net_name] = {
                'connections': connections,
                'net_type': 0,  # Default to signal net
                'weight': 1.0
            }
    
    def _parse_tracks(self, content: str):
        """Parse layer information - Correctly handle X and Y direction TRACKS"""
        track_pattern = r'TRACKS\s+(\w+)\s+([\d\-]+)\s+DO\s+(\d+)\s+STEP\s+(\d+)\s+LAYER\s+(\w+)'
        
        for match in re.finditer(track_pattern, content):
            direction = match.group(1)
            start = int(match.group(2))
            count = int(match.group(3))
            step = int(match.group(4))
            layer = match.group(5)
            
            # Initialize layer data structure (if not exists)
            if layer not in self.def_data['tracks']:
                self.def_data['tracks'][layer] = {
                    'x_direction': None,
                    'y_direction': None,
                    'count': 0,
                    'step': 0
                }
            
            # Store data based on direction
            if direction == 'X':
                self.def_data['tracks'][layer]['x_direction'] = {
                    'start': start,
                    'count': count,
                    'step': step
                }
                # Use X direction data as main feature (X direction is usually more important)
                self.def_data['tracks'][layer]['direction'] = direction
                self.def_data['tracks'][layer]['start'] = start
                self.def_data['tracks'][layer]['count'] = count
                self.def_data['tracks'][layer]['step'] = step
            elif direction == 'Y':
                self.def_data['tracks'][layer]['y_direction'] = {
                    'start': start,
                    'count': count,
                    'step': step
                }
                # If main feature not set yet, use Y direction data
                if self.def_data['tracks'][layer]['direction'] is None:
                    self.def_data['tracks'][layer]['direction'] = direction
                    self.def_data['tracks'][layer]['start'] = start
                    self.def_data['tracks'][layer]['count'] = count
                    self.def_data['tracks'][layer]['step'] = step
    
    def _parse_vias(self, content: str):
        """Parse via information"""
        # Simplified via parsing
        via_pattern = r'VIA\s+(\w+)'
        for match in re.finditer(via_pattern, content):
            via_name = match.group(1)
            self.def_data['vias'][via_name] = {
                'layers': ['metal1', 'metal2'],  # Simplified handling
                'position': (0, 0)
            }
    
    def _parse_rows(self, content: str):
        """Parse row information"""
        row_pattern = r'ROW\s+(\w+)\s+(\w+)\s+([\d\-]+)\s+([\d\-]+)\s+(\w+)'
        
        for match in re.finditer(row_pattern, content):
            row_name = match.group(1)
            site_name = match.group(2)
            x_pos = int(match.group(3))
            y_pos = int(match.group(4))
            orientation = match.group(5)
            
            self.def_data['rows'][row_name] = {
                'site': site_name,
                'position': (x_pos, y_pos),
                'orientation': orientation
            }
    
    def _extract_internal_pin(self, comp_name: str, pin_name: str, net_name: str):
        """Extract internal pin information"""
        if comp_name == 'PIN':  # Skip IO pin
            return
        
        # Estimate pin position based on component position
        comp_info = self.def_data['components'].get(comp_name, {})
        comp_pos = comp_info.get('position', (0, 0))
        
        # Create unique identifier for internal pin
        internal_pin_id = f"{comp_name}_{pin_name}"
        
        self.def_data['internal_pins'][internal_pin_id] = {
            'name': pin_name,
            'component': comp_name,
            'pin_type': self._infer_pin_type(pin_name),
            'net': net_name,
            'position': comp_pos  # simplified: using component positions
        }
    
    def _infer_pin_type(self, pin_name: str) -> str:
        """Infer pin type based on pin name"""
        pin_name_upper = pin_name.upper()
        
        # Output pin patterns
        output_patterns = ['Z', 'ZN', 'Q', 'QN', 'Y', 'CO', 'S']
        if any(pin_name_upper.startswith(pattern) for pattern in output_patterns):
            return 'OUTPUT'
        
        # Input pin patterns
        input_patterns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'CK', 'CLK', 'CP', 'R', 'RN', 'RST', 'EN', 'S', 'SE', 'SI', 'SN']
        if any(pin_name_upper.startswith(pattern) for pattern in input_patterns):
            return 'INPUT'
        
        return 'UNKNOWN'

# ============================================================================
# FEATURE CONFIG SECTION - Feature configuration region (reusing D graph script configuration)
# ============================================================================

class FeatureConfig:
    """Feature configuration class - Centralized management of all feature mappings and parameters"""
    
    # Use global mapping table
    # COMPLETE_CELL_TYPE_MAPPING is defined globally
    
    # Orientation mapping
    ORIENTATION_MAPPING = {
        'N': 0, 'S': 1, 'E': 2, 'W': 3,
        'FN': 4, 'FS': 5, 'FE': 6, 'FW': 7
    }
    
    # Pin direction mapping
    PIN_DIRECTION_MAPPING = {
        'INPUT': 0, 'OUTPUT': 1, 'INOUT': 2, 'FEEDTHRU': 3
    }
    
    # Pin type mapping - Fully continuous encoding, optimized based on actual DEF file analysis
    PIN_TYPE_MAPPING = {
        # Basic input pin types (0-4) - Continuous encoding
        'A': 0, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0,  # Data input A
        'B': 1, 'B1': 1, 'B2': 1, 'B3': 1, 'B4': 1,  # Data input B
        'C': 2, 'C1': 2, 'C2': 2, 'C3': 2, 'C4': 2,  # Data input C
        'D': 3, 'D1': 3, 'D2': 3, 'D3': 3, 'D4': 3,  # Data input D
        'S': 4, 'S0': 4, 'S1': 4,  # Selection signal
        
        # Control signal pin types (5-7) - Continuous encoding
        'CI': 5, 'CIN': 5,  # Carry input
        'CLK': 6, 'CK': 6,  # Clock input - Keep original encoding 6, as it's used in actual data
        'EN': 7, 'G': 7,    # Enable signal
        
        # Output pin types (8-11) - Partially keep original encoding for compatibility with existing data
        'Y': 8, 'CO': 8, 'COUT': 8,  # Combinational logic output
        'Q': 9, 'QN': 9,             # Sequential logic output
        'Z': 10, 'ZN': 10,           # Main output - Keep original encoding 10, widely used in actual data
        'OUT': 11,                   # Other outputs - Keep original encoding 11, used in actual data
        
        # Power pin types (12-13) - Continuous encoding
        'VDD': 12, 'VCC': 12, 'VPWR': 12,  # Power
        'VSS': 13, 'GND': 13, 'VGND': 13,  # Ground
        
        # Unknown pin type
        'UNKNOWN': 14  # Changed to continuous encoding
    }
    
    # Layer mapping
    LAYER_MAPPING = {
        'metal1': 0, 'metal2': 1, 'metal3': 2, 'metal4': 3, 'metal5': 4, 'metal6': 5,
        'metal7': 6, 'metal8': 7, 'metal9': 8, 'metal10': 9,
        'via1': 10, 'via2': 11, 'via3': 12, 'via4': 13, 'via5': 14,
        'via6': 15, 'via7': 16, 'via8': 17, 'via9': 18,
        'poly': 19, 'contact': 20, 'unknown': 21
    }
    
    # Net type mapping
    NET_TYPE_MAPPING = {
        'signal': 0, 'power': 1, 'ground': 2, 'clock': 3, 'reset': 4, 'scan': 5
    }
    
    # Feature dimension configuration
    FEATURE_DIMS = {
        'component_complete_type': len(COMPLETE_CELL_TYPE_MAPPING),  # Complete cell type (base type + drive strength)
        'component_orientation': len(ORIENTATION_MAPPING),
        'pin_direction': len(PIN_DIRECTION_MAPPING),
        'pin_type': len(PIN_TYPE_MAPPING),        # Pin type dimension
        'net_type': len(NET_TYPE_MAPPING),
        'layer_type': len(LAYER_MAPPING),
        'coordinate_dims': 2,  # x, y (removed width, height estimation)
        'area_dim': 1,
        'count_dim': 1         # Count dimension
    }

# ============================================================================
# ENCODING UTILS SECTION - Encoding utilities region (reusing D graph script encoding tools)
# ============================================================================

class EncodingUtils:
    """Encoding utilities class - Provides unified encoding mapping functions"""
    
    # Reference global mapping configuration
    # COMPLETE_CELL_TYPE_MAPPING uses global definition
    ORIENTATION_MAPPING = FeatureConfig.ORIENTATION_MAPPING
    DIRECTION_MAPPING = FeatureConfig.PIN_DIRECTION_MAPPING
    PIN_TYPE_MAPPING = FeatureConfig.PIN_TYPE_MAPPING
    LAYER_MAPPING = FeatureConfig.LAYER_MAPPING
    NET_TYPE_MAPPING = FeatureConfig.NET_TYPE_MAPPING
    
    @classmethod
    def encode_cell_type(cls, cell_type: str) -> int:
        """Encode complete cell type to integer - Use global mapping table"""
        # Directly use global mapping table for encoding
        if cell_type in COMPLETE_CELL_TYPE_MAPPING:
            return COMPLETE_CELL_TYPE_MAPPING[cell_type]
        else:
            return COMPLETE_CELL_TYPE_MAPPING['UNKNOWN']
    
    @classmethod
    def encode_orientation(cls, orientation: str) -> int:
        """Encode orientation to integer"""
        return cls.ORIENTATION_MAPPING.get(orientation, 0)
    
    @classmethod
    def encode_pin_direction(cls, direction: str) -> int:
        """Encode pin direction to integer"""
        return cls.DIRECTION_MAPPING.get(direction, 2)  # Default to INOUT
    
    @classmethod
    def encode_pin_type(cls, pin_name: str) -> int:
        """Encode pin type to integer - Infer based on pin name"""
        pin_name_upper = pin_name.upper()
        
        # Directly match complete pin name
        if pin_name_upper in cls.PIN_TYPE_MAPPING:
            return cls.PIN_TYPE_MAPPING[pin_name_upper]
        
        # Prefix matching - in priority order
        for pattern, type_id in cls.PIN_TYPE_MAPPING.items():
            if pin_name_upper.startswith(pattern):
                return type_id
        
        return cls.PIN_TYPE_MAPPING['UNKNOWN']
    
    @classmethod
    def encode_layer(cls, layer: str) -> int:
        """Encode layer to integer"""
        return cls.LAYER_MAPPING.get(layer, cls.LAYER_MAPPING['unknown'])
    
    @classmethod
    def encode_net_type(cls, net_name: str) -> int:
        """Infer net type based on net name and encode"""
        net_name_lower = net_name.lower()
        
        # Powernet
        if any(keyword in net_name_lower for keyword in ['vdd', 'vcc', 'vpwr', 'power']):
            return cls.NET_TYPE_MAPPING['power']
        
        # Groundnet
        if any(keyword in net_name_lower for keyword in ['vss', 'gnd', 'vgnd', 'ground']):
            return cls.NET_TYPE_MAPPING['ground']
        
        # Clock nets
        if any(keyword in net_name_lower for keyword in ['clk', 'clock', 'ck']):
            return cls.NET_TYPE_MAPPING['clock']
        
        # Reset nets
        if any(keyword in net_name_lower for keyword in ['rst', 'reset', 'rn']):
            return cls.NET_TYPE_MAPPING['reset']
        
        # Scan nets
        if any(keyword in net_name_lower for keyword in ['scan', 'se', 'si', 'so']):
            return cls.NET_TYPE_MAPPING['scan']
        
        # Default to signal net
        return cls.NET_TYPE_MAPPING['signal']
    
    @classmethod
    def encode_placement_status(cls, status: str) -> int:
        """Encode placement status to integer"""
        status_mapping = {
            'PLACED': 0, 'FIXED': 1, 'COVER': 2, 'UNPLACED': 3
        }
        return status_mapping.get(status, 0)

# ============================================================================
# FEATURE ENGINEERING SECTION - Feature engineering region (reusing D graph script feature engineering)
# ============================================================================

class FeatureEngineering:
    """Feature engineering class - Provides advanced feature calculation functions"""
    
    @staticmethod
    def calculate_component_size(cell_type: str, def_data: Dict) -> float:
        """Calculate component actual area - Based on process library actual area data
        
        Engineering optimization:
        - Use actual area data from NangateOpenCellLibrary_typical.lib
        - Convert to DBU² units, unified with coordinate system
        - Reflect actual physical area of cells, providing accurate basis for area optimization
        
        Args:
            cell_type: Cell type string (e.g., 'INV_X1', 'NAND2_X4', etc.)
            def_data: DEF data dictionary, used to get DBU conversion factor
            
        Returns:
            Area size in DBU² units, unified with coordinate units
        """
        # DBU conversion factor: Dynamically get from DEF file UNITS DISTANCE MICRONS
        # Prefer getting from def_data, otherwise use default value
        if def_data and 'units' in def_data:
            DBU_PER_MICRON = float(def_data['units'].get('dbu_per_micron', 2000))
        else:
            DBU_PER_MICRON = 2000  # Default value
        
        # 1 um = DBU_PER_MICRON DBU, so 1 um² = (DBU_PER_MICRON)² DBU²
        AREA_CONVERSION_FACTOR = DBU_PER_MICRON * DBU_PER_MICRON
        
        # 🔬 Process library actual area data (directly extracted from NangateOpenCellLibrary_typical.lib)
        # [CHART] Data source: area attribute of NangateOpenCellLibrary_typical.lib
        # 📐 Unit: um² (square micrometers)
        # [OK] Engineering significance: Directly reflects physical area of cells, based on real process library
        actual_cell_areas = {
            # Basic logic gate series - Actual area (um²)
            'INV_X1': 0.532,        # Inverter reference area
            'INV_X2': 0.798,        # 2x drive strength inverter
            'INV_X4': 1.197,        # 4x drive strength inverter
            'INV_X8': 1.995,        # 8x drive strength inverter
            'INV_X16': 3.990,       # 16x drive strength inverter
            'INV_X32': 7.980,       # 32x drive strength inverter
            
            # NAND gate series - Actual area (um²)
            'NAND2_X1': 0.798,      # 2-input NAND gate
            'NAND2_X2': 1.064,      # 2-input NAND gate, 2x drive
            'NAND2_X4': 1.596,      # 2-input NAND gate, 4x drive
            'NAND3_X1': 1.064,      # 3-input NAND gate
            'NAND3_X2': 1.330,      # 3-input NAND gate, 2x drive
            'NAND3_X4': 1.995,      # 3-input NAND gate, 4x drive
            'NAND4_X1': 1.330,      # 4-input NAND gate
            'NAND4_X2': 1.729,      # 4-input NAND gate, 2x drive
            'NAND4_X4': 2.594,      # 4-input NAND gate, 4x drive
            
            # NOR gate series - Actual area (um²)
            'NOR2_X1': 0.798,       # 2-input NOR gate
            'NOR2_X2': 1.064,       # 2-input NOR gate, 2x drive
            'NOR2_X4': 1.596,       # 2-input NOR gate, 4x drive
            'NOR3_X1': 1.064,       # 3-input NOR gate
            'NOR3_X2': 1.330,       # 3-input NOR gate, 2x drive
            'NOR3_X4': 1.995,       # 3-input NOR gate, 4x drive
            'NOR4_X1': 1.330,       # 4-input NOR gate
            'NOR4_X2': 1.729,       # 4-input NOR gate, 2x drive
            'NOR4_X4': 2.594,       # 4-input NOR gate, 4x drive
            
            # AND gate series - Actual area (um²)
            'AND2_X1': 1.064,       # 2-input AND gate
            'AND2_X2': 1.330,       # 2-input AND gate, 2x drive
            'AND2_X4': 1.995,       # 2-input AND gate, 4x drive
            'AND3_X1': 1.330,       # 3-input AND gate
            'AND3_X2': 1.729,       # 3-input AND gate, 2x drive
            'AND3_X4': 2.594,       # 3-input AND gate, 4x drive
            'AND4_X1': 1.596,       # 4-input AND gate
            'AND4_X2': 1.995,       # 4-input AND gate, 2x drive
            'AND4_X4': 2.992,       # 4-input AND gate, 4x drive
            
            # OR gate series - Actual area (um²)
            'OR2_X1': 1.064,        # 2-input OR gate
            'OR2_X2': 1.330,        # 2-input OR gate, 2x drive
            'OR2_X4': 1.995,        # 2-input OR gate, 4x drive
            'OR3_X1': 1.330,        # 3-input OR gate
            'OR3_X2': 1.729,        # 3-input OR gate, 2x drive
            'OR3_X4': 2.594,        # 3-input OR gate, 4x drive
            'OR4_X1': 1.596,        # 4-input OR gate
            'OR4_X2': 1.995,        # 4-input OR gate, 2x drive
            'OR4_X4': 2.992,        # 4-input OR gate, 4x drive
            
            # XOR gate series - Actual area (um²)
            'XOR2_X1': 1.596,       # 2-input XOR gate
            'XOR2_X2': 2.128,       # 2-input XOR gate, 2x drive
            'XNOR2_X1': 1.596,      # 2-input XNOR gate
            'XNOR2_X2': 2.128,      # 2-input XNOR gate, 2x drive
            
            # AOI/OAI complex gate series - Actual area (um²)
            'AOI21_X1': 1.064,      # AOI21 gate
            'AOI21_X2': 1.330,      # AOI21 gate, 2x drive
            'AOI21_X4': 1.995,      # AOI21 gate, 4x drive
            'AOI22_X1': 1.330,      # AOI22 gate
            'AOI22_X2': 1.729,      # AOI22 gate, 2x drive
            'AOI22_X4': 2.594,      # AOI22 gate, 4x drive
            'OAI21_X1': 1.064,      # OAI21 gate
            'OAI21_X2': 1.330,      # OAI21 gate, 2x drive
            'OAI21_X4': 1.995,      # OAI21 gate, 4x drive
            'OAI22_X1': 1.330,      # OAI22 gate
            'OAI22_X2': 1.729,      # OAI22 gate, 2x drive
            'OAI22_X4': 2.594,      # OAI22 gate, 4x drive
            
            # Buffer series - Actual area (um²)
            'BUF_X1': 0.798,        # Buffer
            'BUF_X2': 1.064,        # Buffer, 2x drive
            'BUF_X4': 1.596,        # Buffer, 4x drive
            'BUF_X8': 2.660,        # Buffer, 8x drive
            'BUF_X16': 4.788,       # Buffer, 16x drive
            'BUF_X32': 9.044,       # Buffer, 32x drive
            
            # Multiplexer series - Actual area (um²)
            'MUX2_X1': 2.660,       # 2-to-1 multiplexer
            'MUX2_X2': 3.458,       # 2-to-1 multiplexer, 2x drive
            
            # Flip-flop series - Actual area (um²)
            'DFF_X1': 4.522,        # D flip-flop
            'DFF_X2': 5.852,        # D flip-flop, 2x drive
            'DFFR_X1': 5.320,       # D flip-flop with reset
            'DFFR_X2': 6.650,       # D flip-flop with reset, 2x drive
            'DFFS_X1': 5.320,       # D flip-flop with set
            'DFFS_X2': 6.650,       # D flip-flop with set, 2x drive
            'DFFRS_X1': 6.118,      # D flip-flop with reset and set
            'DFFRS_X2': 7.448,      # D flip-flop with reset and set, 2x drive
            
            # Latch series - Actual area (um²)
            'DLH_X1': 2.394,        # D latch
            'DLH_X2': 3.192,        # D latch, 2x drive
            'DLHR_X1': 2.926,       # D latch with reset
            'DLHR_X2': 3.724,       # D latch with reset, 2x drive
            'DLHS_X1': 2.926,       # D latch with set
            
            # Physical cell series - Actual area (um²)
            'TAPCELL_X1': 0.266,    # substrate contact cell
            'FILLCELL_X1': 0.266,   # Fill cell
            'FILLCELL_X2': 0.532,   # Fill cell, 2x width
            'FILLCELL_X4': 1.064,   # Fill cell, 4x width
            'FILLCELL_X8': 2.128,   # Fill cell, 8x width
            'FILLCELL_X16': 4.256,  # Fill cell, 16x width
            'FILLCELL_X32': 8.512,  # Fill cell, 32x width
            
            # Logic constants - Actual area (um²)
            'LOGIC0_X1': 0.532,     # Logic 0 constant
            'LOGIC1_X1': 0.532,     # Logic 1 constant
            
            # Antenna protection cell - Actual area (um²)
            'ANTENNA_X1': 0.266,    # Antenna protection cell
        }
        
        normalized_cell_type = cell_type.upper()
        actual_area_um2 = actual_cell_areas.get(normalized_cell_type, 0.532)  # Default INV_X1 area
        
        # Convert to DBU² units, unified with coordinate units
        area_dbu2 = actual_area_um2 * AREA_CONVERSION_FACTOR
        return area_dbu2
    
    @staticmethod
    def calculate_cell_power(cell_type: str) -> float:
        """Get actual power feature based on cell type - Based on NangateOpenCellLibrary_typical.lib
        
        Engineering optimization:
        - Use actual data from process library cell_leakage_power attribute
        - Unit: Original unit consistent with library file (static leakage power)
        - Reflect transistor subthreshold current characteristics, providing accurate basis for power optimization
        
        Args:
            cell_type: Cell type string (e.g., 'INV_X1', 'NAND2_X4', etc.)
            
        Returns:
            Static leakage power value of the cell
        """
        # 🔬 Actual process library power data (extracted from NangateOpenCellLibrary_typical.lib)
        # [CHART] Data source: cell_leakage_power attribute of NangateOpenCellLibrary_typical.lib
        # ⚡ Unit: Original unit consistent with library file // cell power leakage in static state, reflecting transistor subthreshold current characteristics
        real_cell_powers = {
            # Inverter series - Directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'INV_X1': 14.353185,    # Inverter, 1x drive
            'INV_X2': 28.706376,    # Inverter, 2x drive
            'INV_X4': 57.412850,    # Inverter, 4x drive
            'INV_X8': 114.826305,   # Inverter, 8x drive
            'INV_X16': 229.651455,  # Inverter, 16x drive
            'INV_X32': 459.302800,  # Inverter, 32x drive
            
            # Buffer[CN][CN] - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'BUF_X1': 21.438247,    # Buffer, 1x drive
            'BUF_X2': 43.060820,    # Buffer, 2x drive
            'BUF_X4': 86.121805,    # Buffer, 4x drive
            'BUF_X8': 172.244545,   # Buffer, 8x drive
            'BUF_X16': 344.488100,  # Buffer, 16x drive
            'BUF_X32': 688.976200,  # Buffer, 32x drive
            
            # NAND gate series - Directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'NAND2_X1': 17.393360,  # 2-input NAND gate, 1x drive
            'NAND2_X2': 34.786630,  # 2input[CN][CN]gate, 2x drive
            'NAND2_X4': 69.573240,  # 2input[CN][CN]gate, 4x drive
            'NAND3_X1': 18.104768,  # 3input[CN][CN]gate, 1x drive
            'NAND3_X2': 36.209558,  # 3input[CN][CN]gate, 2x drive
            'NAND3_X4': 72.419123,  # 3input[CN][CN]gate, 4x drive
            'NAND4_X1': 18.126843,  # 4input[CN][CN]gate, 1x drive
            'NAND4_X2': 36.253723,  # 4input[CN][CN]gate, 2x drive
            'NAND4_X4': 72.506878,  # 4input[CN][CN]gate, 4x drive
            
            # NOR gate series - Directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'NOR2_X1': 21.199545,   # 2input[CN][CN]gate, 1x drive
            'NOR2_X2': 42.399074,   # 2input[CN][CN]gate, 2x drive
            'NOR2_X4': 84.798143,   # 2input[CN][CN]gate, 4x drive
            'NOR3_X1': 26.831667,   # 3input[CN][CN]gate, 1x drive
            'NOR3_X2': 53.663264,   # 3input[CN][CN]gate, 2x drive
            'NOR3_X4': 107.325918,  # 3input[CN][CN]gate, 4x drive
            'NOR4_X1': 32.601474,   # 4input[CN][CN]gate, 1x drive
            'NOR4_X2': 65.202889,   # 4input[CN][CN]gate, 2x drive
            'NOR4_X4': 130.405147,  # 4input[CN][CN]gate, 4x drive
            
            # AND gate series - Directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'AND2_X1': 25.066064,   # 2input[CN]gate, 1x drive
            'AND2_X2': 50.353160,   # 2input[CN]gate, 2x drive
            'AND2_X4': 100.706457,  # 2input[CN]gate, 4x drive
            'AND3_X1': 26.481460,   # 3input[CN]gate, 1x drive
            'AND3_X2': 53.190270,   # 3input[CN]gate, 2x drive
            'AND3_X4': 106.380663,  # 3input[CN]gate, 4x drive
            'AND4_X1': 27.024804,   # 4input[CN]gate, 1x drive
            'AND4_X2': 54.274743,   # 4input[CN]gate, 2x drive
            'AND4_X4': 108.549590,  # 4input[CN]gate, 4x drive
            
            # ORgate[CN][CN] - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'OR2_X1': 22.694975,    # 2input[CN]gate, 1x drive
            'OR2_X2': 45.656022,    # 2input[CN]gate, 2x drive
            'OR2_X4': 91.312375,    # 2input[CN]gate, 4x drive
            'OR3_X1': 24.414625,    # 3input[CN]gate, 1x drive
            'OR3_X2': 49.162437,    # 3input[CN]gate, 2x drive
            'OR3_X4': 98.325150,    # 3input[CN]gate, 4x drive
            'OR4_X1': 26.733490,    # 4input[CN]gate, 1x drive
            'OR4_X2': 53.869509,    # 4input[CN]gate, 2x drive
            'OR4_X4': 107.739253,   # 4input[CN]gate, 4x drive
            
            # XOR gate series - Directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'XOR2_X1': 36.163718,   # 2-input XOR gate, 1x drive
            'XOR2_X2': 72.593483,   # 2input[CN][CN]gate, 2x drive
            'XNOR2_X1': 36.441009,  # 2-input XNOR gate, 1x drive
            'XNOR2_X2': 73.102975,  # 2input[CN][CN][CN]gate, 2x drive
            
            # AOI/OAI complex gate series - Directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'AOI21_X1': 27.858395,  # AOI21 gate, 1x drive
            'AOI21_X2': 55.716720,  # AOI21 gate, 2x drive
            'AOI21_X4': 111.433338, # AOI21 gate, 4x drive
            'AOI22_X1': 32.611944,  # AOI22 gate, 1x drive
            'AOI22_X2': 65.223838,  # AOI22 gate, 2x drive
            'AOI22_X4': 130.447551, # AOI22 gate, 4x drive
            'OAI21_X1': 22.619394,  # OAI21 gate, 1x drive
            'OAI21_X2': 45.238687,  # OAI21 gate, 2x drive
            'OAI21_X4': 90.477187,  # OAI21 gate, 4x drive
            'OAI22_X1': 34.026125,  # OAI22 gate, 1x drive
            'OAI22_X2': 68.052131,  # OAI22 gate, 2x drive
            'OAI22_X4': 136.103946, # OAI22 gate, 4x drive
            
            # Multiplexer series - Directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'MUX2_X1': 61.229735,   # 2-to-1 multiplexer, 1x drive
            'MUX2_X2': 68.648566,   # 2-to-1 multiplexer, 2x drive
            
            # Flip-flop series - Directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'DFF_X1': 100.684799,   # D flip-flop, 1x drive
            'DFF_X2': 136.676074,   # D flip-flop, 2x drive
            'DFFR_X1': 105.258197,  # D flip-flop with reset, 1x drive
            'DFFR_X2': 141.961514,  # D flip-flop with reset, 2x drive
            'DFFS_X1': 107.724855,  # D flip-flop with set, 1x drive
            'DFFS_X2': 140.592991,  # D flip-flop with set, 2x drive
            'DFFRS_X1': 100.161505, # D flip-flop with reset and set, 1x drive
            'DFFRS_X2': 142.302832, # D flip-flop with reset and set, 2x drive
            
            # Latch series - Directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'DLH_X1': 40.863240,    # D latch, 1x drive
            'DLH_X2': 57.430452,    # D latch, 2x drive
            'DLHR_X1': 40.863416,   # D latch with reset, 1x drive
            'DLHR_X2': 57.430445,   # D latch with reset, 2x drive
            'DLHS_X1': 75.762253,   # D latch with set, 1x drive
            
            # Fill cell[CN][CN] -  based on[CN][CN][CN][CN][CN]([CN][CN] cell[CN][CN] power[CN][CN])
            'FILLCELL_X1': 0.000000, # Fill cell,1x(ANTENNA_X1[CN][CN])
            'FILLCELL_X2': 0.000000, # Fill cell,2x
            'FILLCELL_X4': 0.000000, # Fill cell,4x
            'FILLCELL_X8': 0.000000, # Fill cell,8x
            'FILLCELL_X16': 0.000000,# Fill cell,16x
            'FILLCELL_X32': 0.000000,# Fill cell,32x
            
            # Logic constant cells - Directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'LOGIC0_X1': 35.928390, # Logic 0 cell
            'LOGIC1_X1': 17.885841, # Logic 1 cell
            
            # Antenna protection cell - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'ANTENNA_X1': 0.000000, # Antenna protection cell
        }
        
        normalized_cell_type = cell_type.upper()
        return real_cell_powers.get(normalized_cell_type, 14.353185)  # Default INV_X1 power

# ============================================================================
# F GRAPH GENERATOR SECTION - F graph generator region
# ============================================================================

class FHeterographGenerator:
    """F graph heterograph generator
    
    Generate F graph heterograph based on DEF files, including:
    1. Gate nodes: Keep only gate components
    2. IO_Pin nodes: IO pin nodes
    3. Gate-Gate edges: 6-dimensional features [pin_type_id, cell_type_id] + [net_type_id, connection_count] + [pin_type_id, cell_type_id]
    4. IO_Pin-Gate edges: 4-dimensional features [net_type_id, connection_count] + [pin_type_id, cell_type_id]
    """
    
    # ==================== Standard type mapping definition ====================
    # Keep mapping order consistent with node_edge_analysis_report.md
    
    @property
    def node_type_to_id(self) -> Dict[str, int]:
        """Node type to ID mapping - Standard order"""
        return {
            "gate": 0,
            "io_pin": 1, 
            "net": 2,
            "pin": 3
        }
    
    @property  
    def edge_type_to_id(self) -> Dict[str, int]:
        """Edge type to ID mapping - Standard order"""
        return {
            "('gate', 'connects_to', 'gate')": 0,
            "('gate', 'connects_to', 'net')": 1,
            "('gate', 'has', 'pin')": 2,
            "('io_pin', 'connects_to', 'gate')": 3,
            "('io_pin', 'connects_to', 'net')": 4,
            "('io_pin', 'connects_to', 'pin')": 5,
            "('pin', 'connects_to', 'net')": 6,
            "('pin', 'connects_to', 'pin')": 7,
            "('pin', 'gate_connects', 'pin')": 8
        }
    
    # ========================================================
    
    def __init__(self, def_data: Dict, def_file_path: str = None):
        self.def_data = def_data
        self.def_file_path = def_file_path
        self.gate_gate_edges = []
        self.io_pin_gate_edges = []
        
        # Statistics information
        self.stats = {
            'total_gates': 0,
            'total_io_pins': 0,
            'total_gate_gate_edges': 0,
            'total_io_pin_gate_edges': 0,
            'nets_processed': 0
        }
    
    def generate(self) -> HeteroData:
        """Generate F graph heterograph"""
        print("[START] Start generating F graph heterograph...")
        
        try:
            # 1. Build gate-gate edges and IO_Pin-gate edges
            self._build_edges()
            
            # 2. Create heterograph data structure
            hetero_data = self._create_hetero_data()
            
            # 3. Output statistics information
            self._print_statistics()
            
            return hetero_data
            
        except Exception as e:
            print(f"[X] F graph generation  failed: {e}")
            traceback.print_exc()
            raise
    
    def _build_edges(self):
        """Build gate-gate edges and IO_Pin-gate edges"""
        print("[CHART] Building edge connections...")
        
        nets = self.def_data.get('nets', {})
        components = self.def_data.get('components', {})
        pins = self.def_data.get('pins', {})
        
        for net_name, net_info in nets.items():
            self.stats['nets_processed'] += 1
            connections = net_info.get('connections', [])
            
            if len(connections) < 2:
                continue  # Skip nets with only one connection
            
            # Separate IO pins and gate pins
            io_pins = []
            gate_pins = []
            
            for comp_name, pin_name in connections:
                if comp_name == 'PIN':
                    # This is IO pin
                    io_pins.append((comp_name, pin_name))
                elif comp_name in components:
                    # This is gate pin
                    gate_pins.append((comp_name, pin_name))
            
            # Build gate-gate edges
            self._build_gate_gate_edges(net_name, net_info, gate_pins, components)
            
            # Build IO_Pin-gate edges
            self._build_io_pin_gate_edges(net_name, net_info, io_pins, gate_pins, components)
    
    def _build_gate_gate_edges(self, net_name: str, net_info: Dict, gate_pins: List[Tuple[str, str]], components: Dict):
        """Build gate-gate edges"""
        if len(gate_pins) < 2:
            return
        
        # Identify output pins and input pins
        output_pins = []
        input_pins = []
        
        for comp_name, pin_name in gate_pins:
            pin_type = self._infer_pin_type(pin_name)
            if pin_type == 'OUTPUT':
                output_pins.append((comp_name, pin_name))
            else:
                input_pins.append((comp_name, pin_name))
        
        # If no clear output pin, use first pin as output
        if not output_pins and gate_pins:
            output_pins = [gate_pins[0]]
            input_pins = gate_pins[1:]
        
        # From each output pin connect to all input pins
        for source_comp, source_pin in output_pins:
            for target_comp, target_pin in input_pins:
                if source_comp == target_comp:
                    continue  # Skip self-connection
                
                # Create gate-gate edge
                edge = GateGateEdge(
                    source_gate=source_comp,
                    target_gate=target_comp,
                    source_pin_type=EncodingUtils.encode_pin_type(source_pin),
                    source_cell_type=EncodingUtils.encode_cell_type(components[source_comp]['cell_type']),
                    net_type=EncodingUtils.encode_net_type(net_name),
                    connection_count=len(gate_pins),
                    target_pin_type=EncodingUtils.encode_pin_type(target_pin),
                    target_cell_type=EncodingUtils.encode_cell_type(components[target_comp]['cell_type']),
                    net_name=net_name
                )
                
                self.gate_gate_edges.append(edge)
                self.stats['total_gate_gate_edges'] += 1
    
    def _build_io_pin_gate_edges(self, net_name: str, net_info: Dict, io_pins: List[Tuple[str, str]],
                                gate_pins: List[Tuple[str, str]], components: Dict):
        """Build IO_Pin-gate edge"""
        if not io_pins or not gate_pins:
            return
        
        # Calculate HPWL of this net as edge label (corresponding to D graph's IO Pin-Pin edge label)
        net_hpwl = self._calculate_net_hpwl(net_name)
        
        # From each IO pin connect to all gate pins
        for io_comp, io_pin in io_pins:
            for gate_comp, gate_pin in gate_pins:
                # Create IO_Pin-gate edge
                edge = IO_PinGateEdge(
                    io_pin_name=io_pin,
                    gate_name=gate_comp,
                    net_type=EncodingUtils.encode_net_type(net_name),
                    connection_count=len(gate_pins) + len(io_pins),
                    gate_pin_type=EncodingUtils.encode_pin_type(gate_pin),
                    gate_cell_type=EncodingUtils.encode_cell_type(components[gate_comp]['cell_type']),
                    net_name=net_name,
                    net_hpwl=net_hpwl
                )
                
                self.io_pin_gate_edges.append(edge)
                self.stats['total_io_pin_gate_edges'] += 1
    
    def _infer_pin_type(self, pin_name: str) -> str:
        """Infer pin type based on pin name"""
        pin_name_upper = pin_name.upper()
        
        # Output pin patterns
        output_patterns = ['Z', 'ZN', 'Q', 'QN', 'Y', 'CO', 'S']
        if any(pin_name_upper.startswith(pattern) for pattern in output_patterns):
            return 'OUTPUT'
        
        # Input pin patterns
        input_patterns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'CK', 'CLK', 'CP', 'R', 'RN', 'RST', 'EN', 'S', 'SE', 'SI', 'SN']
        if any(pin_name_upper.startswith(pattern) for pattern in input_patterns):
            return 'INPUT'
        
        return 'UNKNOWN'
    
    def _create_hetero_data(self) -> HeteroData:
        """Create heterograph data structure"""
        print("🏗️ Creating heterograph data structure...")
        
        hetero_data = HeteroData()
        
        # 1. Create gate nodes
        self._create_gate_nodes(hetero_data)
        
        # 2. Create IO_Pin nodes
        self._create_io_pin_nodes(hetero_data)
        
        # 3. Create edges - in standard order
        self._create_gate_gate_edges(hetero_data)     # Gate-Gate edges (ID: 0)
        self._create_io_pin_gate_edges(hetero_data)   # IO_Pin-Gate edges (ID: 3)
        
        # 5. Add global features and labels (keep consistent with D graph)
        self._add_global_features(hetero_data)
        
        return hetero_data
    
    def _create_gate_nodes(self, hetero_data: HeteroData):
        """Create gate nodes (keep feature format and order consistent with D graph)"""
        components = self.def_data.get('components', {})
        
        # Filter out actual gate components (exclude fill cells, etc.)
        gate_components = {}
        for comp_name, comp_info in components.items():
            cell_type = comp_info['cell_type']
            if (not cell_type.startswith('FILLCELL') and 
                not cell_type.startswith('ANTENNA') and 
                not cell_type.startswith('TAPCELL')):
                gate_components[comp_name] = comp_info
        
        self.stats['total_gates'] = len(gate_components)
        
        if not gate_components:
            print("[!]️ Warning: No gate components found")
            hetero_data['gate'].x = torch.empty((0, 7), dtype=torch.float)
            return
        
        # Create gate to index mapping
        self.gate_to_idx = {name: idx for idx, name in enumerate(gate_components.keys())}
        
        # Build gate node features (keep feature order consistent with D graph)
        gate_features = []
        names = []
        for comp_name, comp_info in gate_components.items():
            # Gate node features: [x, y, cell_type_id, orientation_id, area, placement_status, power] - Keep consistent with D graph
            pos = comp_info.get('position', (0, 0))
            cell_type_id = EncodingUtils.encode_cell_type(comp_info.get('cell_type', ''))
            orientation_id = EncodingUtils.encode_orientation(comp_info.get('orientation', 'N'))
            area = FeatureEngineering.calculate_component_size(comp_info.get('cell_type', ''), self.def_data)
            placement_status = EncodingUtils.encode_placement_status(comp_info.get('placement_status', 'PLACED'))
            power = FeatureEngineering.calculate_cell_power(comp_info.get('cell_type', ''))
            
            features = [pos[0], pos[1], cell_type_id, orientation_id, area, placement_status, power]
            gate_features.append(features)
            names.append(comp_name)
        
        hetero_data['gate'].x = torch.tensor(gate_features, dtype=torch.float)
        hetero_data['gate'].names = names
        print(f"[OK] Created {len(gate_components)} gate nodes (feature format consistent with D graph)")
    
    def _create_io_pin_nodes(self, hetero_data: HeteroData):
        """Create IO_Pin nodes (keep feature format consistent with D graph)"""
        pins = self.def_data.get('pins', {})
        self.stats['total_io_pins'] = len(pins)
        
        if not pins:
            print("[!]️ Warning: No IO pins found")
            hetero_data['io_pin'].x = torch.empty((0, 4), dtype=torch.float)
            return
        
        # Create IO_Pin to index mapping
        self.io_pin_to_idx = {name: idx for idx, name in enumerate(pins.keys())}
        
        # Build IO_Pin node features (keep consistent with D graph)
        io_pin_features = []
        names = []
        for pin_name, pin_info in pins.items():
            # IO_Pin node features: [x, y, direction_id, layer_id] - Keep consistent with D graph
            pos = pin_info.get('position', (0, 0))
            direction_id = EncodingUtils.encode_pin_direction(pin_info.get('direction', 'INOUT'))
            layer_id = EncodingUtils.encode_layer(pin_info.get('layer', 'metal1'))
            
            features = [pos[0], pos[1], direction_id, layer_id]
            io_pin_features.append(features)
            names.append(pin_name)
        
        hetero_data['io_pin'].x = torch.tensor(io_pin_features, dtype=torch.float)
        hetero_data['io_pin'].names = names
        print(f"[OK] Created {len(pins)} IO_Pin nodes (feature format consistent with D graph)")
    
    def _create_gate_gate_edges(self, hetero_data: HeteroData):
        """Create gate-gate edges (add HPWL labels from D graph pin_pin edges)"""
        if not self.gate_gate_edges:
            print("[!]️ Warning: No gate-gate edges")
            hetero_data['gate', 'connects_to', 'gate'].edge_index = torch.empty((2, 0), dtype=torch.long)
            hetero_data['gate', 'connects_to', 'gate'].edge_attr = torch.empty((0, 6), dtype=torch.float)
            hetero_data['gate', 'connects_to', 'gate'].edge_label = torch.empty((0,), dtype=torch.float)
            return
        
        # Build edge indices, edge features, and edge labels
        edge_indices = []
        edge_features = []
        edge_labels = []
        
        for edge in self.gate_gate_edges:
            if edge.source_gate in self.gate_to_idx and edge.target_gate in self.gate_to_idx:
                source_idx = self.gate_to_idx[edge.source_gate]
                target_idx = self.gate_to_idx[edge.target_gate]
                
                edge_indices.append([source_idx, target_idx])
                
                # 6-dimensional edge features: [source_pin_type, source_cell_type, net_type, connection_count, target_pin_type, target_cell_type]
                features = [
                    edge.source_pin_type,
                    edge.source_cell_type,
                    edge.net_type,
                    edge.connection_count,
                    edge.target_pin_type,
                    edge.target_cell_type
                ]
                edge_features.append(features)
                
                # Calculate HPWL of this net as edge label (corresponding to D graph pin_pin edge label)
                net_hpwl = self._calculate_net_hpwl(edge.net_name)
                edge_labels.append(net_hpwl)
        
        if edge_indices:
            hetero_data['gate', 'connects_to', 'gate'].edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            hetero_data['gate', 'connects_to', 'gate'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
            hetero_data['gate', 'connects_to', 'gate'].edge_label = torch.tensor(edge_labels, dtype=torch.float)
            print(f"[OK] Created {len(edge_indices)} gate-gate edges (including HPWL labels)")
        else:
            hetero_data['gate', 'connects_to', 'gate'].edge_index = torch.empty((2, 0), dtype=torch.long)
            hetero_data['gate', 'connects_to', 'gate'].edge_attr = torch.empty((0, 6), dtype=torch.float)
            hetero_data['gate', 'connects_to', 'gate'].edge_label = torch.empty((0,), dtype=torch.float)
    
    def _create_io_pin_gate_edges(self, hetero_data: HeteroData):
        """Create IO_Pin-gate edges (add HPWL labels from D graph IO_Pin-Pin edges)"""
        if not self.io_pin_gate_edges:
            print("[!]️ Warning: No IO_Pin-gate edges")
            hetero_data['io_pin', 'connects_to', 'gate'].edge_index = torch.empty((2, 0), dtype=torch.long)
            hetero_data['io_pin', 'connects_to', 'gate'].edge_attr = torch.empty((0, 4), dtype=torch.float)
            hetero_data['io_pin', 'connects_to', 'gate'].edge_label = torch.empty((0,), dtype=torch.float)
            return
        
        # Build edge indices, edge features, and edge labels
        edge_indices = []
        edge_features = []
        edge_labels = []
        
        for edge in self.io_pin_gate_edges:
            if edge.io_pin_name in self.io_pin_to_idx and edge.gate_name in self.gate_to_idx:
                io_pin_idx = self.io_pin_to_idx[edge.io_pin_name]
                gate_idx = self.gate_to_idx[edge.gate_name]
                
                edge_indices.append([io_pin_idx, gate_idx])
                
                # 4-dimensional edge features: [net_type_id, connection_count, pin_type_id, cell_type_id]
                features = [
                    edge.net_type,
                    edge.connection_count,
                    edge.gate_pin_type,
                    edge.gate_cell_type
                ]
                edge_features.append(features)
                
                # Edge label: net HPWL (corresponding to D graph's IO Pin-Pin edge label)
                edge_labels.append(edge.net_hpwl)
        
        if edge_indices:
            hetero_data['io_pin', 'connects_to', 'gate'].edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            hetero_data['io_pin', 'connects_to', 'gate'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
            hetero_data['io_pin', 'connects_to', 'gate'].edge_label = torch.tensor(edge_labels, dtype=torch.float)
            print(f"[OK] Created {len(edge_indices)} IO_Pin-gate edges (including HPWL labels)")
        else:
            hetero_data['io_pin', 'connects_to', 'gate'].edge_index = torch.empty((2, 0), dtype=torch.long)
            hetero_data['io_pin', 'connects_to', 'gate'].edge_attr = torch.empty((0, 4), dtype=torch.float)
            hetero_data['io_pin', 'connects_to', 'gate'].edge_label = torch.empty((0,), dtype=torch.float)
    
    def _add_global_features(self, hetero_data: HeteroData):
        """Add global features and labels (completely consistent with D graph)"""
        print("[GLOBE] Adding global features and labels...")
        
        # Get basic chip information
        die_area = self.def_data.get('die_area', (0, 0, 100000, 100000))
        units_info = self.def_data.get('units', {'dbu_per_micron': 1000})
        dbu_per_micron = units_info.get('dbu_per_micron', 1000) if isinstance(units_info, dict) else units_info
        
        # Basic global features: [chip width, chip height, chip area, DBU unit]
        chip_width = die_area[2] - die_area[0]
        chip_height = die_area[3] - die_area[1]
        chip_area = chip_width * chip_height
        
        # Read CORE_UTILIZATION configuration from config.mk
        core_utilization_config = self._read_config_utilization()
        
        # Extended global features: [chip width, chip height, chip area, DBU unit, configured utilization]
        global_features = torch.tensor([chip_width, chip_height, chip_area, dbu_per_micron, core_utilization_config], dtype=torch.float)
        hetero_data.global_features = global_features
        
        # Chip coordinate information - 2x2 matrix: [[bottom_left_x, bottom_left_y], [top_right_x, top_right_y]]
        die_coordinates = torch.tensor([[die_area[0], die_area[1]], [die_area[2], die_area[3]]], dtype=torch.float)
        hetero_data.die_coordinates = die_coordinates
        
        # Read global labels from placement report
        utilization, hpwl = self._read_placement_labels()
        
        # Calculate Pin Density (internal pin utilization rate)
        internal_pin_count = len(self.def_data.get('internal_pins', {}))
        total_pin_count = internal_pin_count + len(self.def_data.get('pins', {}))
        pin_density = (internal_pin_count / total_pin_count * 100.0) if total_pin_count > 0 else 0.0
        
        # Global labels: [actual utilization, HPWL, Pin Density]
        y = torch.tensor([utilization, hpwl, pin_density], dtype=torch.float)
        hetero_data.y = y
        
        print(f"  ✓ Global features: 5 dimensions (width={chip_width}, height={chip_height}, area={chip_area}, DBU={dbu_per_micron}, configured_utilization={core_utilization_config})")
        print(f"  ✓ Global labels: 3 dimensions (actual_utilization={utilization}%, HPWL={hpwl}um, Pin_density={pin_density:.2f}%)")
    
    def _calculate_total_hpwl(self) -> float:
        """Calculate total half-perimeter wire length (HPWL)"""
        nets = self.def_data.get('nets', {})
        components = self.def_data.get('components', {})
        pins = self.def_data.get('pins', {})
        total_hpwl = 0.0
        
        for net_name, net_info in nets.items():
            connections = net_info.get('connections', [])
            if len(connections) < 2:
                continue
            
            # Collect coordinates of all connection points
            x_coords = []
            y_coords = []
            
            for comp_name, pin_name in connections:
                if comp_name == 'PIN' and pin_name in pins:
                    # IO pin
                    pos = pins[pin_name].get('position', (0, 0))
                    x_coords.append(pos[0])
                    y_coords.append(pos[1])
                elif comp_name in components:
                    # Component pin
                    pos = components[comp_name].get('position', (0, 0))
                    x_coords.append(pos[0])
                    y_coords.append(pos[1])
            
            if len(x_coords) >= 2:
                # Calculate HPWL = (max_x - min_x) + (max_y - min_y)
                hpwl = (max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords))
                total_hpwl += hpwl
        
        return total_hpwl
    
    def _calculate_net_hpwl(self, net_name: str) -> float:
        """Calculate single net HPWL (for gate_gate edge labels)"""
        nets = self.def_data.get('nets', {})
        components = self.def_data.get('components', {})
        pins = self.def_data.get('pins', {})
        
        if net_name not in nets:
            return 0.0
            
        net_info = nets[net_name]
        connections = net_info.get('connections', [])
        if len(connections) < 2:
            return 0.0
            
        # Collect coordinates of all connection points
        x_coords = []
        y_coords = []
        
        for comp_name, pin_name in connections:
            if comp_name == 'PIN' and pin_name in pins:
                # IO pin
                pos = pins[pin_name].get('position', (0, 0))
                x_coords.append(pos[0])
                y_coords.append(pos[1])
            elif comp_name in components:
                # Component pin
                pos = components[comp_name].get('position', (0, 0))
                x_coords.append(pos[0])
                y_coords.append(pos[1])
        
        if len(x_coords) >= 2:
            # Calculate HPWL = (max_x - min_x) + (max_y - min_y)
            return (max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords))
        
        return 0.0
    
    def _read_config_utilization(self):
        """Read CORE_UTILIZATION configuration from place_data_extract.csv file for the corresponding design"""
        try:
            import csv
            import os
            
            # Extract design name from DEF file path (remove _place.def suffix)
            if hasattr(self, 'def_file_path') and self.def_file_path:
                import os
                filename = os.path.basename(self.def_file_path)
                # Extract section before _place as design name
                design_name = filename.replace('_place.def', '')
            else:
                # Backup plan: Use design name inside DEF file
                design_name = self.def_data.get('design_name', '')
            
            # CSV file path - Use file in current directory
            csv_path = 'place_data_extract.csv'
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Improved design name matching logic
                    csv_design_name = row['design_name']
                    match_found = False  # Initialize match flag
                    
                    # 1. Exact match
                    if design_name == csv_design_name:
                        match_found = True
                    # 2. DEF design name is in CSV name
                    elif design_name in csv_design_name:
                        match_found = True
                    # 3. CSV name is in DEF design name
                    elif csv_design_name in design_name:
                        match_found = True
                    # 4. Handle parentheses case, extract name inside parentheses for matching
                    elif '(' in csv_design_name and ')' in csv_design_name:
                        bracket_name = csv_design_name.split('(')[1].split(')')[0]
                        if design_name == bracket_name:
                            match_found = True
                        else:
                            match_found = False
                    # 5. Reverse parentheses matching
                    elif '(' in csv_design_name:
                        prefix_name = csv_design_name.split('(')[0]
                        if design_name == prefix_name:
                            match_found = True
                        else:
                            match_found = False
                    # 6. Handle common name variants
                    elif (design_name.replace('_top', '') == csv_design_name or 
                          design_name.replace('_core', '') == csv_design_name or
                          design_name == csv_design_name + '_top' or
                          design_name == csv_design_name + '_core')::
                        match_found = True
                    else:
                        match_found = False
                    
                    if match_found:
                        core_util = row['core_utilization']
                        if core_util and core_util.strip():  # Check if empty
                            print(f"[OK] Successfully matched design: '{design_name}' in DEF <-> '{csv_design_name}' in CSV (core_utilization={core_util})")
                            return float(core_util)
            
            print(f"[!]️  Warning: core_utilization data for design {design_name} not found in CSV file")
            return 0.0  # Default value
        except Exception as e:
            print(f"[!]️  Warning: Unable to read place_data_extract.csv file: {e}")
            return 0.0  # Default value
    
    def _read_placement_labels(self):
        """Read design_utilization and hpwl_after labels from place_data_extract.csv file for the corresponding design"""
        try:
            import csv
            import os
            
            # Extract design name from DEF file path (remove _place.def suffix)
            if hasattr(self, 'def_file_path') and self.def_file_path:
                import os
                filename = os.path.basename(self.def_file_path)
                # Extract section before _place as design name
                design_name = filename.replace('_place.def', '')
            else:
                # Backup plan: Use design name inside DEF file
                design_name = self.def_data.get('design_name', '')
            
            # CSV file path - Use file in current directory
            csv_path = 'place_data_extract.csv'
            
            utilization = 0.0
            hpwl = 0.0
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Improved design name matching logic
                    csv_design_name = row['design_name']
                    
                    # 1. Exact match
                    if design_name == csv_design_name:
                        match_found = True
                    # 2. DEF design name is in CSV name
                    elif design_name in csv_design_name:
                        match_found = True
                    # 3. CSV name is in DEF design name
                    elif csv_design_name in design_name:
                        match_found = True
                    # 4. Handle parentheses case, extract name inside parentheses for matching
                    elif '(' in csv_design_name and ')' in csv_design_name:
                        bracket_name = csv_design_name.split('(')[1].split(')')[0]
                        if design_name == bracket_name:
                            match_found = True
                        else:
                            match_found = False
                    # 5. Reverse parentheses matching
                    elif '(' in csv_design_name:
                        prefix_name = csv_design_name.split('(')[0]
                        if design_name == prefix_name:
                            match_found = True
                        else:
                            match_found = False
                    # 6. Handle common name variants
                    elif (design_name.replace('_top', '') == csv_design_name or 
                          design_name.replace('_core', '') == csv_design_name or
                          design_name == csv_design_name + '_top' or
                          design_name == csv_design_name + '_core')::
                        match_found = True
                    else:
                        match_found = False
                    
                    if match_found:
                        # Read design_utilization
                        design_util = row['design_utilization']
                        if design_util and design_util.strip():
                            utilization = float(design_util)
                        
                        # Read hpwl_after
                        hpwl_after = row['hpwl_after']
                        if hpwl_after and hpwl_after.strip():
                            hpwl = float(hpwl_after)
                        
                        print(f"[OK] Successfully matched design: '{design_name}' in DEF <-> '{csv_design_name}' in CSV")
                        return utilization, hpwl
            
            print(f"[!]️  Warning: Placement label data for design {design_name} not found in CSV file")
            return 0.0, 0.0  # Default values
        except Exception as e:
            print(f"[!]️  Warning: Unable to read place_data_extract.csv file: {e}")
            return 0.0, 0.0  # Default values
    
    def _calculate_component_area(self, cell_type: str) -> float:
        """Calculate component area based on cell type"""
        # Simplified area calculation - Based on drive strength
        if '_X1' in cell_type:
            return 1.0
        elif '_X2' in cell_type:
            return 2.0
        elif '_X4' in cell_type:
            return 4.0
        elif '_X8' in cell_type:
            return 8.0
        elif '_X16' in cell_type:
            return 16.0
        elif '_X32' in cell_type:
            return 32.0
        else:
            return 1.0  # Default area
    
    def _print_statistics(self):
        """Output statistics information"""
        print("\n[CHART] F graph generation statistics:")
        print(f"   Total gates: {self.stats['total_gates']}")
        print(f"   Total IO_Pins: {self.stats['total_io_pins']}")
        print(f"   Gate-Gate edges: {self.stats['total_gate_gate_edges']}")
        print(f"   IO_Pin-Gate edges: {self.stats['total_io_pin_gate_edges']}")
        print(f"   Nets processed: {self.stats['nets_processed']}")

# ============================================================================
# MAIN FUNCTION SECTION - Main function region
# ============================================================================

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='F graph heterograph generator')
    parser.add_argument('def_file', help='Input DEF file path')
    parser.add_argument('-o', '--output', help='Output .pt file path', default=None)

    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.def_file):
        print(f"[X] Error: DEF file does not exist: {args.def_file}")
        return
    
    # Set output file path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.def_file))[0]
        args.output = f"{base_name}_f_graph.pt"
    
    try:
        print(f"[SEARCH] Parsing DEF file: {args.def_file}")
        
        # 1. Parse DEF file
        parser = DEFParser(args.def_file)
        def_data = parser.parse()
        
        # 2. Generate F graph
        generator = FHeterographGenerator(def_data, args.def_file)
        hetero_data = generator.generate()
        
        # 3. Save result
        print(f"[SAVE] Saving F graph to: {args.output}")
        torch.save(hetero_data, args.output)
        

        
        print("[OK] F graph generation completed!")
        
    except Exception as e:
        print(f"[X] Processing  failed: {e}")
        traceback.print_exc()



if __name__ == "__main__":
    main()