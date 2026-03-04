#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Basic Heterograph Generator (with Internal Pin Support)

This script is based on the original place_def_to_heterograph.py, retaining only the basic heterograph generation functionality,
and adds support for internal pin node types.

Functionality:
- Parse design information from DEF files
- Generate basic heterographs (stage a), containing the following node and edge types:
  * Node types: Gate, Net, IO_Pin, Pin
  * Edge types: IO_Pin-Net, Pin-Net, Gate-Pin
- Support internal pin parsing: extract unit internal port information from network connections (such as A1, A2, B1, B2, ZN, etc.)
- Preserve all basic feature information and original mapping tables
- Remove advanced graph structure conversion functions (Pin Edge, Net Edge, integrated heterograph)
- Remove redundant Gate-Net edges, achieving equivalent connections through the combination of Gate-Pin and Pin-Net edges

Author: EDA for AI Team
Date: 2024

Usage:
    python basic_heterograph_generator.py input.def -o output.pt
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
import csv

# ============================================================================
# DATA STRUCTURES SECTION - Data structure definition area
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
    position: Tuple[float, float]               # Component position coordinates (x, y)
    orientation: str = 'N'                      # Component orientation (N, S, E, W, FN, FS, FE, FW)
    pins: Dict[str, Any] = field(default_factory=dict)  # Component pin information dictionary
    size: float = 1.0                           # Component relative size (based on drive strength)
    
@dataclass
class NetInfo:
    """Net information data structure"""
    name: str                                   # Net name
    connections: List[Tuple[str, str]] = field(default_factory=list)  # Connection list: (component name, pin name)
    routing: List[Dict] = field(default_factory=list)                 # Routing information list
    net_type: int = 0                          # Net type code (0-signal, 1-power, 2-ground, 3-clock, etc.)
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
class InternalPinInfo:
    """Internal pin information data structure"""
    name: str                                   # Pin name
    component: str                              # Parent component name
    pin_type: str = 'UNKNOWN'                   # Pin type (INPUT, OUTPUT)
    net: str = ''                               # Connected net name
    position: Tuple[float, float] = (0.0, 0.0)  # Pin position coordinates (based on component position)

# ============================================================================
# DEF PARSER SECTION - DEF file parser area
# ============================================================================

class DEFParser:
    """DEF file parser

    Responsible for parsing DEF files and extracting structured data, including:
    - Design information (name, units, die area)
    - Component information (position, type, orientation)
    - Pin information (position and direction of IO pins)
    - Net information (connection relationships)
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
            'internal_pins': {},  # New: internal pin information
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
            print(f"❌ DEF file parsing failed: {e}")
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
        """Parse IO pin information - extract real position coordinates and layer information"""
        pins_section = re.search(r'PINS\s+(\d+)\s*;(.*?)END\s+PINS', content, re.DOTALL)
        if not pins_section:
            return
        
        pins_text = pins_section.group(2)
        
        # Use more precise regular expression to split pin definitions
        # Each pin definition starts with "- pin_name" and ends with ";"
        pin_pattern = r'-\s+([\w\[\]_]+)\s+\+\s+NET\s+([\w\[\]_]+)\s+\+\s+DIRECTION\s+(\w+).*?;'
        
        for match in re.finditer(pin_pattern, pins_text, re.DOTALL):
            pin_name = match.group(1)
            net_name = match.group(2)
            direction = match.group(3)
            pin_def = match.group(0)
            
            # Parse LAYER information from PORT section
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
        """Parse net information and internal pin information (including NETS and SPECIALNETS)"""
        # Parse regular nets (NETS section)
        nets_section = re.search(r'NETS\s+(\d+)\s*;(.*?)END\s+NETS', content, re.DOTALL)
        if nets_section:
            nets_text = nets_section.group(2)
            self._parse_nets_section(nets_text, net_type=0)  # Signal nets
        
        # Parse special nets (SPECIALNETS section)
        specialnets_section = re.search(r'SPECIALNETS\s+(\d+)\s*;(.*?)END\s+SPECIALNETS', content, re.DOTALL)
        if specialnets_section:
            specialnets_text = specialnets_section.group(2)
            self._parse_nets_section(specialnets_text, net_type=1)  # Special nets (power/ground)
    
    def _parse_nets_section(self, nets_text: str, net_type: int):
        """Generic method for parsing net sections"""
        # Improved net parsing - supports complex net names and connection formats, includes dot characters
        # Match format: - net_name ( comp1 pin1 ) ( comp2 pin2 ) ... + USE SIGNAL ;
        net_pattern = r'-\s+([\w\[\]_$\\.]+)\s+((?:\([^)]+\)\s*)+)'
        
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
                'net_type': net_type,  # Use the passed net type
                'weight': 1.0
            }
    
    def _parse_tracks(self, content: str):
        """Parse layer information - correctly handle X and Y direction TRACKS"""
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
                # Use X direction data as main feature (usually X direction is more important)
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
                # If main feature is not set yet, use Y direction data
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
        if comp_name == 'PIN':  # Skip IO pins
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
            'position': comp_pos  # Simplified: use component position
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
# FEATURE CONFIG SECTION - Feature configuration area
# ============================================================================

class FeatureConfig:
    """Feature configuration class - centrally manage all feature mappings and parameters"""

    # Use global mapping table
    # COMPLETE_CELL_TYPE_MAPPING has been defined globally

    # Orientation mapping
    ORIENTATION_MAPPING = {
        'N': 0, 'S': 1, 'E': 2, 'W': 3,
        'FN': 4, 'FS': 5, 'FE': 6, 'FW': 7
    }

    # Pin direction mapping
    PIN_DIRECTION_MAPPING = {
        'INPUT': 0, 'OUTPUT': 1, 'INOUT': 2, 'FEEDTHRU': 3
    }

    # Pin type mapping - fully continuous encoding, optimized based on actual DEF file analysis
    # Original data usage types: 0,1,2,3,4,6,10,11 -> remapped to continuous encoding 0-7
    PIN_TYPE_MAPPING = {
        # Basic input pin types (0-4) - continuous encoding
        'A': 0, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0,  # Data input A
        'B': 1, 'B1': 1, 'B2': 1, 'B3': 1, 'B4': 1,  # Data input B
        'C': 2, 'C1': 2, 'C2': 2, 'C3': 2, 'C4': 2,  # Data input C
        'D': 3, 'D1': 3, 'D2': 3, 'D3': 3, 'D4': 3,  # Data input D
        'S': 4, 'S0': 4, 'S1': 4,  # Select signal

        # Control signal pin types (5-7) - continuous encoding
        'CI': 5, 'CIN': 5,  # Carry input
        'CLK': 6, 'CK': 6,  # Clock input - keep original encoding 6 because actual data uses it
        'EN': 7, 'G': 7,    # Enable signal

        # Output pin types (8-11) - partially keep original encoding to be compatible with existing data
        'Y': 8, 'CO': 8, 'COUT': 8,  # Combinational logic output
        'Q': 9, 'QN': 9,             # Sequential logic output
        'Z': 10, 'ZN': 10,           # Main output - keep original encoding 10, extensively used in actual data
        'OUT': 11,                   # Other outputs - keep original encoding 11, used in actual data

        # Power pin types (12-13) - continuous encoding
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

    # Network type mapping
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
# ENCODING UTILS SECTION - Encoding utility class area
# ============================================================================

class EncodingUtils:
    """Encoding utility class - provides unified encoding mapping functionality"""

    # Reference global mapping configuration
    # COMPLETE_CELL_TYPE_MAPPING uses global definition
    ORIENTATION_MAPPING = FeatureConfig.ORIENTATION_MAPPING
    DIRECTION_MAPPING = FeatureConfig.PIN_DIRECTION_MAPPING
    PIN_TYPE_MAPPING = FeatureConfig.PIN_TYPE_MAPPING
    LAYER_MAPPING = FeatureConfig.LAYER_MAPPING
    NET_TYPE_MAPPING = FeatureConfig.NET_TYPE_MAPPING
    
    @classmethod
    def encode_cell_type(cls, cell_type: str) -> int:
        """Encode complete cell type to integer - use global mapping table"""
        # Directly use global mapping table for encoding
        if cell_type in COMPLETE_CELL_TYPE_MAPPING:
            return COMPLETE_CELL_TYPE_MAPPING[cell_type]

        return COMPLETE_CELL_TYPE_MAPPING['UNKNOWN']  # Return 95 for unknown type
    
    @classmethod
    def encode_orientation(cls, orientation: str) -> int:
        """Encode orientation to integer"""
        return cls.ORIENTATION_MAPPING.get(orientation, 0)

    @classmethod
    def encode_direction(cls, direction: str) -> int:
        """Encode pin direction to integer"""
        return cls.DIRECTION_MAPPING.get(direction, 2)

    @classmethod
    def encode_layer(cls, layer: str) -> int:
        """Encode layer name to integer"""
        return cls.LAYER_MAPPING.get(layer, cls.LAYER_MAPPING['unknown'])

    @classmethod
    def encode_pin_type(cls, pin_name: str) -> int:
        """Encode pin type to integer"""
        return cls.PIN_TYPE_MAPPING.get(pin_name.upper(), cls.PIN_TYPE_MAPPING['UNKNOWN'])

    @classmethod
    def encode_net_type(cls, net_name: str) -> int:
        """Infer network type based on network name and encode"""
        net_name_lower = net_name.lower()
        if 'vdd' in net_name_lower or 'vcc' in net_name_lower or 'power' in net_name_lower:
            return cls.NET_TYPE_MAPPING['power']
        elif 'vss' in net_name_lower or 'gnd' in net_name_lower or 'ground' in net_name_lower:
            return cls.NET_TYPE_MAPPING['ground']
        elif 'clk' in net_name_lower or 'clock' in net_name_lower:
            return cls.NET_TYPE_MAPPING['clock']
        elif 'rst' in net_name_lower or 'reset' in net_name_lower:
            return cls.NET_TYPE_MAPPING['reset']
        elif 'scan' in net_name_lower:
            return cls.NET_TYPE_MAPPING['scan']
        else:
            return cls.NET_TYPE_MAPPING['signal']
    
    @classmethod
    def encode_placement_status(cls, placement_status: str) -> int:
        """Encode placement status to integer (placement stage specific)

        Args:
            placement_status: Placement status string (e.g., 'PLACED', 'FIXED', 'COVER', etc.)

        Returns:
            Corresponding integer encoding
        """
        placement_mapping = {
            'PLACED': 0,   # Placed (movable)
            'FIXED': 1,    # Fixed position (non-movable)
            'COVER': 2,    # Cover placement
            'UNPLACED': 3  # Unplaced
        }
        return placement_mapping.get(placement_status, 0)

# ============================================================================
# FEATURE ENGINEERING SECTION - Feature engineering utility class area
# ============================================================================

class FeatureEngineering:
    """Feature engineering utility class - provides coordinate normalization, size calculation, etc."""

    @staticmethod
    def normalize_coordinates(position: Tuple[float, float], die_area: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Normalize coordinates to [0,1] range"""
        x, y = position
        x_min, y_min, x_max, y_max = die_area
        
        if x_max > x_min and y_max > y_min:
            norm_x = (x - x_min) / (x_max - x_min)
            norm_y = (y - y_min) / (y_max - y_min)
        else:
            norm_x, norm_y = 0.0, 0.0
        
        return (norm_x, norm_y)
    
    @staticmethod
    def calculate_component_size(cell_type: str, def_data: Dict = None) -> float:
        """Get DBU unit area based on cell type - unify coordinate and area units

        Engineering optimization:
        - Convert process library area (um²) to DBU² units, unified with coordinate units
        - Conversion formula: area_dbu² = area_um² × (2000 DBU/um)²
        - Based on DEF file UNITS DISTANCE MICRONS 2000 unit definition

        Args:
            cell_type: Cell type string (e.g., 'INV_X1', 'NAND2_X4', etc.)
            def_data: DEF data dictionary, used to get DBU conversion factor

        Returns:
            Area size in DBU² units, unified with coordinate units
        """
        # DBU conversion factor: dynamically obtained from DEF file UNITS DISTANCE MICRONS
        # Priority to get from def_data, otherwise use default value
        if def_data and 'units' in def_data:
            DBU_PER_MICRON = float(def_data['units'].get('dbu_per_micron', 2000))
        else:
            DBU_PER_MICRON = 2000  # Default value

        # 1 um = DBU_PER_MICRON DBU, therefore 1 um² = (DBU_PER_MICRON)² DBU²
        AREA_CONVERSION_FACTOR = DBU_PER_MICRON * DBU_PER_MICRON

        # Process library actual area data (directly extracted from NangateOpenCellLibrary_typical.lib)
        # Data source: area attribute of NangateOpenCellLibrary_typical.lib
        # Unit: um² (square micrometers)
        # Engineering significance: Directly reflects the physical area of the cell, based on real process library
        actual_cell_areas = {
            # Basic logic gate series - actual area (um²)
            'INV_X1': 0.532,        # Inverter reference area
            'INV_X2': 0.798,        # 2x drive strength inverter
            'INV_X4': 1.330000,        # 4x drive strength inverter
            'INV_X8': 2.394000,        # 8x drive strength inverter
            'INV_X16': 4.522000,       # 16x drive strength inverter
            'INV_X32': 8.778000,       # 32x drive strength inverter

            # NAND gate series - actual area (um²)
            'NAND2_X1': 0.798,      # 2-input NAND gate
            'NAND2_X2': 1.330000,      # 2-input NAND gate, 2x drive
            'NAND2_X4': 2.394000,      # 2-input NAND gate, 4x drive
            'NAND3_X1': 1.064,      # 3-input NAND gate
            'NAND3_X2': 1.862000,      # 3-input NAND gate, 2x drive
            'NAND3_X4': 3.458000,      # 3-input NAND gate, 4x drive
            'NAND4_X1': 1.330,      # 4-input NAND gate
            'NAND4_X2': 2.394000,      # 4-input NAND gate, 2x drive
            'NAND4_X4': 4.788000,      # 4-input NAND gate, 4x drive

            # NOR gate series - actual area (um²)
            'NOR2_X1': 0.798,       # 2-input NOR gate
            'NOR2_X2': 1.330000,       # 2-input NOR gate, 2x drive
            'NOR2_X4': 2.394000,       # 2-input NOR gate, 4x drive
            'NOR3_X1': 1.064,       # 3-input NOR gate
            'NOR3_X2': 1.862000,       # 3-input NOR gate, 2x drive
            'NOR3_X4': 3.724000,       # 3-input NOR gate, 4x drive
            'NOR4_X1': 1.330,       # 4-input NOR gate
            'NOR4_X2': 2.394000,       # 4-input NOR gate, 2x drive
            'NOR4_X4': 4.788000,       # 4-input NOR gate, 4x drive

            # AND gate series - actual area (um²)
            'AND2_X1': 1.064,       # 2-input AND gate
            'AND2_X2': 1.330,       # 2-input AND gate, 2x drive
            'AND2_X4': 2.394000,       # 2-input AND gate, 4x drive
            'AND3_X1': 1.330,       # 3-input AND gate
            'AND3_X2': 1.596000,       # 3-input AND gate, 2x drive
            'AND3_X4': 2.926000,       # 3-input AND gate, 4x drive
            'AND4_X1': 1.596,       # 4-input AND gate
            'AND4_X2': 1.862000,       # 4-input AND gate, 2x drive
            'AND4_X4': 3.458000,       # 4-input AND gate, 4x drive

            # OR gate series - actual area (um²)
            'OR2_X1': 1.064,        # 2-input OR gate
            'OR2_X2': 1.330,        # 2-input OR gate, 2x drive
            'OR2_X4': 2.394000,        # 2-input OR gate, 4x drive
            'OR3_X1': 1.330,        # 3-input OR gate
            'OR3_X2': 1.596000,        # 3-input OR gate, 2x drive
            'OR3_X4': 2.926000,        # 3-input OR gate, 4x drive
            'OR4_X1': 1.596,        # 4-input OR gate
            'OR4_X2': 1.862000,        # 4-input OR gate, 2x drive
            'OR4_X4': 3.458000,        # 4-input OR gate, 4x drive

            # XOR gate series - actual area (um²)
            'XOR2_X1': 1.596,       # 2-input XOR gate
            'XOR2_X2': 2.394000,       # 2-input XOR gate, 2x drive
            'XNOR2_X1': 1.596,      # 2-input XNOR gate
            'XNOR2_X2': 2.660000,      # 2-input XNOR gate, 2x drive

            # AOI/OAI complex gate series - actual area (um²)
            'AOI21_X1': 1.064,      # AOI21 gate
            'AOI21_X2': 1.862000,      # AOI21 gate, 2x drive
            'AOI21_X4': 3.458000,      # AOI21 gate, 4x drive
            'AOI22_X1': 1.330,      # AOI22 gate
            'AOI22_X2': 2.394000,      # AOI22 gate, 2x drive
            'AOI22_X4': 4.522000,      # AOI22 gate, 4x drive
            'OAI21_X1': 1.064,      # OAI21 gate
            'OAI21_X2': 1.862000,      # OAI21 gate, 2x drive
            'OAI21_X4': 3.458000,      # OAI21 gate, 4x drive
            'OAI22_X1': 1.330,      # OAI22 gate
            'OAI22_X2': 2.394000,      # OAI22 gate, 2x drive
            'OAI22_X4': 4.522000,      # OAI22 gate, 4x drive

            # Buffer series - actual area (um²)
            'BUF_X1': 0.798,        # Buffer
            'BUF_X2': 1.064,        # Buffer, 2x drive
            'BUF_X4': 1.862000,        # Buffer, 4x drive
            'BUF_X8': 3.458000,        # Buffer, 8x drive
            'BUF_X16': 6.650000,       # Buffer, 16x drive
            'BUF_X32': 13.034000,       # Buffer, 32x drive

            # Multiplexer series - actual area (um²)
            'MUX2_X1': 1.862000,       # 2-to-1 multiplexer
            'MUX2_X2': 2.394000,       # 2-to-1 multiplexer, 2x drive

            # Flip-flop series - actual area (um²)
            'DFF_X1': 4.522,        # D flip-flop
            'DFF_X2': 5.054000,        # D flip-flop, 2x drive
            'DFFR_X1': 5.320,       # D flip-flop with reset
            'DFFR_X2': 5.852000,       # D flip-flop with reset, 2x drive
            'DFFS_X1': 5.320,       # D flip-flop with set
            'DFFS_X2': 5.586000,       # D flip-flop with set, 2x drive
            'DFFRS_X1': 6.384000,      # D flip-flop with reset and set
            'DFFRS_X2': 6.916000,      # D flip-flop with reset and set, 2x drive

            # Latch series - actual area (um²)
            'DLH_X1': 2.660000,        # D latch
            'DLH_X2': 2.926000,        # D latch, 2x drive
            'DLHR_X1': 2.926,       # D latch with reset
            'DLHR_X2': 3.724,       # D latch with reset, 2x drive
            'DLHS_X1': 2.926,       # D latch with set

            # Physical cells - actual area (um²)
            'TAPCELL_X1': 0.266,    # Substrate contact cell
            'FILLCELL_X1': 0.266,   # Fill cell
            'FILLCELL_X2': 0.266000,   # Fill cell, 2x width
            'FILLCELL_X4': 1.064,   # Fill cell, 4x width
            'FILLCELL_X8': 2.128,   # Fill cell, 8x width
            'FILLCELL_X16': 4.256,  # Fill cell, 16x width
            'FILLCELL_X32': 8.512,  # Fill cell, 32x width

            # Logic constants - actual area (um²)
            'LOGIC0_X1': 0.532,     # Logic 0 constant
            'LOGIC1_X1': 0.532,     # Logic 1 constant

            # Antenna protection cell - actual area (um²)
            'ANTENNA_X1': 0.266,    # Antenna protection cell
        }

        normalized_cell_type = cell_type.upper()
        actual_area_um2 = actual_cell_areas.get(normalized_cell_type, 0.532)  # Default INV_X1 area

        # Convert to DBU² units, unified with coordinate units
        area_dbu2 = actual_area_um2 * AREA_CONVERSION_FACTOR
        return area_dbu2
    
    @staticmethod
    def calculate_net_weight(connections: List[Tuple[str, str]]) -> float:
        """Calculate network weight based on connection count"""
        return max(1.0, len(connections) / 2.0)

    @staticmethod
    def calculate_cell_power(cell_type: str) -> float:
        """Get real power consumption feature based on cell type - based on NangateOpenCellLibrary_typical.lib

        Engineering optimization:
        - Directly use real data from process library cell_leakage_power attribute
        - Unit: original unit consistent with library file (static leakage power)
        - Reflects subthreshold current characteristics of transistors, providing accurate basis for power optimization

        Args:
            cell_type: Cell type string (e.g., 'INV_X1', 'NAND2_X4', etc.)

        Returns:
            Static leakage power value of the cell
        """
        # Real process library power data (extracted from NangateOpenCellLibrary_typical.lib)
        # Data source: cell_leakage_power attribute of NangateOpenCellLibrary_typical.lib
        # Unit: original unit consistent with library file // cell power leakage in static state, reflecting subthreshold current characteristics of transistors
        real_cell_powers = {
            # Inverter series - directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'INV_X1': 14.353185,    # Inverter, 1x drive
            'INV_X2': 28.706376,    # Inverter, 2x drive
            'INV_X4': 57.412850,    # Inverter, 4x drive
            'INV_X8': 114.826305,   # Inverter, 8x drive
            'INV_X16': 229.651455,  # Inverter, 16x drive
            'INV_X32': 459.302800,  # Inverter, 32x drive

            # Buffer series - directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'BUF_X1': 21.438247,    # Buffer, 1x drive
            'BUF_X2': 43.060820,    # Buffer, 2x drive
            'BUF_X4': 86.121805,    # Buffer, 4x drive
            'BUF_X8': 172.244545,   # Buffer, 8x drive
            'BUF_X16': 344.488100,  # Buffer, 16x drive
            'BUF_X32': 688.976200,  # Buffer, 32x drive

            # NAND gate series - directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'NAND2_X1': 17.393360,  # 2-input NAND gate, 1x drive
            'NAND2_X2': 34.786630,  # 2-input NAND gate, 2x drive
            'NAND2_X4': 69.573240,  # 2-input NAND gate, 4x drive
            'NAND3_X1': 18.104768,  # 3-input NAND gate, 1x drive
            'NAND3_X2': 36.209558,  # 3-input NAND gate, 2x drive
            'NAND3_X4': 72.419123,  # 3-input NAND gate, 4x drive
            'NAND4_X1': 18.126843,  # 4-input NAND gate, 1x drive
            'NAND4_X2': 36.253723,  # 4-input NAND gate, 2x drive
            'NAND4_X4': 72.506878,  # 4-input NAND gate, 4x drive

            # NOR gate series - directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'NOR2_X1': 21.199545,   # 2-input NOR gate, 1x drive
            'NOR2_X2': 42.399074,   # 2-input NOR gate, 2x drive
            'NOR2_X4': 84.798143,   # 2-input NOR gate, 4x drive
            'NOR3_X1': 26.831667,   # 3-input NOR gate, 1x drive
            'NOR3_X2': 53.663264,   # 3-input NOR gate, 2x drive
            'NOR3_X4': 107.325918,  # 3-input NOR gate, 4x drive
            'NOR4_X1': 32.601474,   # 4-input NOR gate, 1x drive
            'NOR4_X2': 65.202889,   # 4-input NOR gate, 2x drive
            'NOR4_X4': 130.405147,  # 4-input NOR gate, 4x drive

            # AND gate series - directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'AND2_X1': 25.066064,   # 2-input AND gate, 1x drive
            'AND2_X2': 50.353160,   # 2-input AND gate, 2x drive
            'AND2_X4': 100.706457,  # 2-input AND gate, 4x drive
            'AND3_X1': 26.481460,   # 3-input AND gate, 1x drive
            'AND3_X2': 53.190270,   # 3-input AND gate, 2x drive
            'AND3_X4': 106.380663,  # 3-input AND gate, 4x drive
            'AND4_X1': 27.024804,   # 4-input AND gate, 1x drive
            'AND4_X2': 54.274743,   # 4-input AND gate, 2x drive
            'AND4_X4': 108.549590,  # 4-input AND gate, 4x drive

            # OR gate series - directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'OR2_X1': 22.694975,    # 2-input OR gate, 1x drive
            'OR2_X2': 45.656022,    # 2-input OR gate, 2x drive
            'OR2_X4': 91.312375,    # 2-input OR gate, 4x drive
            'OR3_X1': 24.414625,    # 3-input OR gate, 1x drive
            'OR3_X2': 49.162437,    # 3-input OR gate, 2x drive
            'OR3_X4': 98.325150,    # 3-input OR gate, 4x drive
            'OR4_X1': 26.733490,    # 4-input OR gate, 1x drive
            'OR4_X2': 53.869509,    # 4-input OR gate, 2x drive
            'OR4_X4': 107.739253,   # 4-input OR gate, 4x drive

            # XOR gate series - directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'XOR2_X1': 36.163718,   # 2-input XOR gate, 1x drive
            'XOR2_X2': 72.593483,   # 2-input XOR gate, 2x drive
            'XNOR2_X1': 36.441009,  # 2-input XNOR gate, 1x drive
            'XNOR2_X2': 73.102975,  # 2-input XNOR gate, 2x drive

            # AOI/OAI complex gate series - directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
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

            # Multiplexer series - directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'MUX2_X1': 35.928390,   # 2-to-1 multiplexer, 1x drive
            'MUX2_X2': 68.648566,   # 2-to-1 multiplexer, 2x drive

            # Flip-flop series - directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'DFF_X1': 79.112308,   # D flip-flop, 1x drive
            'DFF_X2': 115.103670,   # D flip-flop, 2x drive
            'DFFR_X1': 86.212780,  # D flip-flop with reset, 1x drive
            'DFFR_X2': 125.988288,  # D flip-flop with reset, 2x drive
            'DFFS_X1': 84.395957,  # D flip-flop with set, 1x drive
            'DFFS_X2': 121.107028,  # D flip-flop with set, 2x drive
            'DFFRS_X1': 100.161505, # D flip-flop with reset and set, 1x drive
            'DFFRS_X2': 142.302832, # D flip-flop with reset and set, 2x drive

            # Latch series - directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'DLH_X1': 40.863240,    # D latch, 1x drive
            'DLH_X2': 57.430452,    # D latch, 2x drive
            'DLHR_X1': 40.863416,   # D latch with reset, 1x drive
            'DLHR_X2': 57.430445,   # D latch with reset, 2x drive
            'DLHS_X1': 75.762253,   # D latch with set, 1x drive

            # Fill cell series - calculated based on library file (physical cells usually have very low power)
            'FILLCELL_X1': 0.000000, # Fill cell, 1x (value of ANTENNA_X1)
            'FILLCELL_X2': 0.000000, # Fill cell, 2x
            'FILLCELL_X4': 0.000000, # Fill cell, 4x
            'FILLCELL_X8': 0.000000, # Fill cell, 8x
            'FILLCELL_X16': 0.000000,# Fill cell, 16x
            'FILLCELL_X32': 0.000000,# Fill cell, 32x

            # Logic constant cells - directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'LOGIC0_X1': 35.928390, # Logic 0 cell
            'LOGIC1_X1': 17.885841, # Logic 1 cell

            # Antenna protection cell - directly from NangateOpenCellLibrary_typical.lib cell_leakage_power
            'ANTENNA_X1': 0.000000, # Antenna protection cell
        }

        normalized_cell_type = cell_type.upper()
        return real_cell_powers.get(normalized_cell_type, 14.353185)  # Default to INV_X1 power

# ============================================================================
# BASE HETEROGRAPH BUILDER SECTION - Basic heterograph builder area
# ============================================================================

class BaseHeteroGraphBuilder:
    """Basic heterograph builder

    Responsible for building the most basic heterograph structure (stage a), including:
    - Node types: Gate, Net, IO_Pin, Pin
    - Edge types: IO_Pin-Net, Pin-Net, Gate-Pin
    - Basic connection relationships and features
    """

    # ==================== B graph customized type mapping definition ====================
    # Based on actual usage of B script in node_edge_analysis_report.md
    
    @property
    def node_type_to_id(self) -> Dict[str, int]:
        """Node type to ID mapping for B graph"""
        return {
            "gate": 0,
            "io_pin": 1,
            "net": 2,
            "pin": 3
        }

    @property
    def edge_type_to_id(self) -> Dict[str, int]:
        """Edge type to ID mapping for B graph"""
        return {
            "('gate', 'has', 'pin')": 0,
            "('io_pin', 'connects_to', 'net')": 1,
            "('pin', 'connects_to', 'net')": 2
        }
    
    # ========================================================
    
    def __init__(self, def_data: Dict, def_file_path: str = None):
        self.def_data = def_data
        self.def_file_path = def_file_path  # Save DEF file path for reading configuration
        self.die_area = def_data.get('die_area', (0, 0, 1000, 1000))

        # Initialize heterograph data
        self.hetero_data = HeteroData()

        # Node name to index mapping - only keep four node types
        self.gate_name_to_idx = {}         # Gate node mapping
        self.net_name_to_idx = {}          # Net node mapping
        self.pin_name_to_idx = {}          # IO_Pin node mapping
        self.internal_pin_name_to_idx = {} # Pin node mapping

    def build(self, graph_id=0) -> HeteroData:
        """Build basic heterograph"""
        print("🔧 Building basic heterograph (stage a)")

        # Add four node types: in standard order Gate -> IO_Pin -> Net -> Pin
        self._add_gate_nodes()         # Gate node (ID: 0)
        self._add_pin_nodes()          # IO_Pin node (ID: 1)
        self._add_net_nodes()          # Net node (ID: 2)
        self._add_pin_nodes_internal() # Pin node (ID: 3)

        # Add three edge types: in standard order
        self._add_gate_pin_edges()         # Gate-Pin edge (ID: 2)
        self._add_pin_net_edges()          # IO_Pin-Net edge (ID: 4)
        self._add_pin_net_edges_internal() # Pin-Net edge (ID: 6)

        # Add global features
        self._add_global_features()

        print("✅ Basic heterograph construction completed")
        return self.hetero_data

    def _add_gate_nodes(self):
        """Add gate nodes (component nodes)"""
        components = self.def_data.get('components', {})
        if not components:
            print("⚠️  Warning: No component data found")
            return

        # Filter out real gate components (exclude fill cells, etc.)
        gate_components = {}
        for comp_name, comp_info in components.items():
            cell_type = comp_info.get('cell_type', '')
            if (not cell_type.startswith('FILLCELL') and
                not cell_type.startswith('ANTENNA') and
                not cell_type.startswith('TAPCELL')):
                gate_components[comp_name] = comp_info

        features = []
        names = []

        for comp_name, comp_info in gate_components.items():
            # Feature vector: [x, y, cell_type_id, orientation_id, area, placement_status, power]
            pos = comp_info.get('position', (0, 0))

            cell_type_id = EncodingUtils.encode_cell_type(comp_info.get('cell_type', ''))
            orient_id = EncodingUtils.encode_orientation(comp_info.get('orientation', 'N'))
            size = FeatureEngineering.calculate_component_size(comp_info.get('cell_type', ''), self.def_data)
            placement_status = EncodingUtils.encode_placement_status(comp_info.get('placement_status', 'PLACED'))
            power = FeatureEngineering.calculate_cell_power(comp_info.get('cell_type', ''))

            features.append([pos[0], pos[1], cell_type_id, orient_id, size, placement_status, power])
            names.append(comp_name)

        self.hetero_data['gate'].x = torch.tensor(features, dtype=torch.float)
        self.hetero_data['gate'].names = names
        self.gate_name_to_idx = {name: i for i, name in enumerate(names)}

        print(f"  ✓ Gate nodes: {len(names)}")
    
    def _calculate_net_hpwl(self, net_name: str, net_info: Dict) -> float:
        """Calculate HPWL (Half-Perimeter Wire Length) of network

        Args:
            net_name: Network name
            net_info: Network information dictionary

        Returns:
            float: HPWL value, calculated as (x_max - x_min) + (y_max - y_min)
        """
        connections = net_info.get('connections', [])
        if not connections:
            return 0.0

        # Collect coordinates of all gates connected to this network
        x_coords = []
        y_coords = []

        components = self.def_data.get('components', {})
        pins = self.def_data.get('pins', {})

        for comp_name, pin_name in connections:
            # Handle IO pin connections
            if comp_name == 'PIN' and pin_name in pins:
                pin_pos = pins[pin_name].get('position', (0, 0))
                x_coords.append(pin_pos[0])
                y_coords.append(pin_pos[1])
            # Handle component pin connections
            elif comp_name in components:
                comp_pos = components[comp_name].get('position', (0, 0))
                x_coords.append(comp_pos[0])
                y_coords.append(comp_pos[1])

        # If no valid coordinates, return 0
        if not x_coords or not y_coords:
            return 0.0

        # Calculate HPWL = (x_max - x_min) + (y_max - y_min)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        hpwl = (x_max - x_min) + (y_max - y_min)
        return float(hpwl)

    def _add_net_nodes(self):
        """Add network nodes"""
        nets = self.def_data.get('nets', {})
        if not nets:
            print("⚠️  Warning: No network data found")
            return

        features = []
        names = []
        hpwl_labels = []  # New: HPWL label list

        for net_name, net_info in nets.items():
            # Feature vector: [net_type_id, connection_count]
            net_type_id = EncodingUtils.encode_net_type(net_name)
            connections = net_info.get('connections', [])
            connection_count = len(connections)

            # Calculate HPWL as label
            hpwl = self._calculate_net_hpwl(net_name, net_info)

            features.append([net_type_id, connection_count])
            names.append(net_name)
            hpwl_labels.append(hpwl)

        self.hetero_data['net'].x = torch.tensor(features, dtype=torch.float)
        self.hetero_data['net'].y = torch.tensor(hpwl_labels, dtype=torch.float)  # New: HPWL label
        self.hetero_data['net'].names = names
        self.net_name_to_idx = {name: i for i, name in enumerate(names)}

        # Statistics of HPWL information
        if hpwl_labels:
            hpwl_stats = {
                'min': min(hpwl_labels),
                'max': max(hpwl_labels),
                'mean': sum(hpwl_labels) / len(hpwl_labels),
                'zero_count': sum(1 for h in hpwl_labels if h == 0.0)
            }
            print(f"  ✓ Net nodes: {len(names)}")
            print(f"  ✓ HPWL statistics: min={hpwl_stats['min']:.1f}, max={hpwl_stats['max']:.1f}, mean={hpwl_stats['mean']:.1f}, zero_count={hpwl_stats['zero_count']}")
        else:
            print(f"  ✓ Net nodes: {len(names)}")
    
    def _add_pin_nodes(self):
        """Add IO pin nodes"""
        pins = self.def_data.get('pins', {})
        if not pins:
            print("⚠️  Warning: No pin data found")
            return

        features = []
        names = []

        for pin_name, pin_info in pins.items():
            # Feature vector: [x, y, direction_id, layer_id]
            pos = pin_info.get('position', (0, 0))

            pin_direction_id = EncodingUtils.encode_direction(pin_info.get('direction', 'INOUT'))
            layer_id = EncodingUtils.encode_layer(pin_info.get('layer', 'metal1'))

            features.append([pos[0], pos[1], pin_direction_id, layer_id])
            names.append(pin_name)

        self.hetero_data['io_pin'].x = torch.tensor(features, dtype=torch.float)
        self.hetero_data['io_pin'].names = names
        self.pin_name_to_idx = {name: i for i, name in enumerate(names)}

        print(f"  ✓ IO_Pin nodes: {len(names)}")

    def _add_pin_nodes_internal(self):
        """Add internal pin nodes"""
        internal_pins = self.def_data.get('internal_pins', {})
        if not internal_pins:
            print("⚠️  Warning: No internal pin data found")
            return

        features = []
        names = []

        for pin_id, pin_info in internal_pins.items():
            # Feature vector: [pin_type_id, cell_type_id] - remove coordinate info because pin is bound to gate
            pin_type_id = EncodingUtils.encode_pin_type(pin_info.get('name', ''))

            # Get type of parent component
            comp_name = pin_info.get('component', '')
            comp_info = self.def_data.get('components', {}).get(comp_name, {})
            cell_type_id = EncodingUtils.encode_cell_type(comp_info.get('cell_type', ''))

            features.append([pin_type_id, cell_type_id])
            names.append(pin_id)

        self.hetero_data['pin'].x = torch.tensor(features, dtype=torch.float)
        self.hetero_data['pin'].names = names
        self.internal_pin_name_to_idx = {name: i for i, name in enumerate(names)}

        print(f"  ✓ Pin nodes: {len(names)}")

    # Deleted Layer, Via, Row node addition methods, only keep Gate, Net, IO_Pin, Pin four node types

    # Gate-Net edge deleted because of redundancy
    # Gate connects to Net through Pin, Gate-Net edge is equivalent to combination of Gate-Pin and Pin-Net edges
    
    def _add_pin_net_edges(self):
        """Add IO_Pin to Net edges"""
        if not self.pin_name_to_idx or not self.net_name_to_idx:
            print("⚠️  Warning: IO_Pin or Net nodes empty, skip Pin-Net edges")
            return

        edge_indices = []
        edge_features = []

        pins = self.def_data.get('pins', {})
        for pin_name, pin_info in pins.items():
            if pin_name not in self.pin_name_to_idx:
                continue

            pin_idx = self.pin_name_to_idx[pin_name]
            net_name = pin_info.get('net', '')

            if net_name in self.net_name_to_idx:
                net_idx = self.net_name_to_idx[net_name]
                edge_indices.append([pin_idx, net_idx])

                # Edge feature: [pin_direction_id, net_type_id]
                pin_direction_id = EncodingUtils.encode_direction(pin_info.get('direction', 'INOUT'))
                net_type_id = EncodingUtils.encode_net_type(net_name)

                edge_features.append([pin_direction_id, net_type_id])

        if edge_indices:
            self.hetero_data['io_pin', 'connects_to', 'net'].edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            self.hetero_data['io_pin', 'connects_to', 'net'].edge_attr = torch.tensor(edge_features, dtype=torch.float)

        print(f"  ✓ IO_Pin-Net edges: {len(edge_indices)}")

    def _add_pin_net_edges_internal(self):
        """Add Pin to Net edges"""
        if not self.internal_pin_name_to_idx or not self.net_name_to_idx:
            print("⚠️  Warning: Pin or Net nodes empty, skip Pin-Net edges")
            return

        edge_indices = []
        edge_features = []

        internal_pins = self.def_data.get('internal_pins', {})
        for pin_id, pin_info in internal_pins.items():
            if pin_id not in self.internal_pin_name_to_idx:
                continue

            pin_idx = self.internal_pin_name_to_idx[pin_id]
            net_name = pin_info.get('net', '')

            if net_name in self.net_name_to_idx:
                net_idx = self.net_name_to_idx[net_name]
                edge_indices.append([pin_idx, net_idx])

                # Edge feature: [pin_type_id, net_type_id]
                pin_type_id = EncodingUtils.encode_pin_type(pin_info.get('name', ''))
                net_type_id = EncodingUtils.encode_net_type(net_name)

                edge_features.append([pin_type_id, net_type_id])

        if edge_indices:
            self.hetero_data['pin', 'connects_to', 'net'].edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            self.hetero_data['pin', 'connects_to', 'net'].edge_attr = torch.tensor(edge_features, dtype=torch.float)

        print(f"  ✓ Pin-Net edges: {len(edge_indices)}")

    def _add_gate_pin_edges(self):
        """Add Gate to Pin edges"""
        if not self.gate_name_to_idx or not self.internal_pin_name_to_idx:
            print("⚠️  Warning: Gate or Pin nodes empty, skip Gate-Pin edges")
            return

        edge_indices = []
        edge_features = []

        internal_pins = self.def_data.get('internal_pins', {})
        for pin_id, pin_info in internal_pins.items():
            if pin_id not in self.internal_pin_name_to_idx:
                continue

            pin_idx = self.internal_pin_name_to_idx[pin_id]
            comp_name = pin_info.get('component', '')

            if comp_name in self.gate_name_to_idx:
                gate_idx = self.gate_name_to_idx[comp_name]
                edge_indices.append([gate_idx, pin_idx])

                # Edge feature: [cell_type_id, pin_type_id]
                comp_info = self.def_data['components'].get(comp_name, {})
                cell_type_id = EncodingUtils.encode_cell_type(comp_info.get('cell_type', ''))
                pin_type_id = EncodingUtils.encode_pin_type(pin_info.get('name', ''))

                edge_features.append([cell_type_id, pin_type_id])

        if edge_indices:
            self.hetero_data['gate', 'has', 'pin'].edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            self.hetero_data['gate', 'has', 'pin'].edge_attr = torch.tensor(edge_features, dtype=torch.float)

        print(f"  ✓ Gate-Pin edges: {len(edge_indices)}")

    # Deleted Via-Layer, Row-Gate, Net-Via edge addition methods, only keep three edge types
    
    def _add_global_features(self):
        """Add global features and labels"""
        die_area = self.die_area
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
        self.hetero_data.global_features = global_features

        # Chip coordinate information - 2x2 matrix: [[bottom_left_x, bottom_left_y], [top_right_x, top_right_y]]
        die_coordinates = torch.tensor([[die_area[0], die_area[1]], [die_area[2], die_area[3]]], dtype=torch.float)
        self.hetero_data.die_coordinates = die_coordinates

        # Read global labels from placement report
        utilization, hpwl = self._read_placement_labels()

        # Calculate Pin Density (internal pin utilization)
        internal_pin_count = len(self.def_data.get('internal_pins', {}))
        total_pin_count = internal_pin_count + len(self.def_data.get('pins', {}))
        pin_density = (internal_pin_count / total_pin_count * 100.0) if total_pin_count > 0 else 0.0

        # Global labels: [actual utilization, HPWL, Pin Density]
        y = torch.tensor([utilization, hpwl, pin_density], dtype=torch.float)
        self.hetero_data.y = y

        print(f"  ✓ Global features: 5 dimensions (width={chip_width}, height={chip_height}, area={chip_area}, DBU={dbu_per_micron}, config_utilization={core_utilization_config})")
        print(f"  ✓ Global labels: 3 dimensions (actual_utilization={utilization}%, HPWL={hpwl}um, Pin_density={pin_density:.2f}%)")

    def _read_config_utilization(self):
        """Read CORE_UTILIZATION configuration of corresponding design from place_data_extract.csv file"""
        try:
            # Extract design name from DEF file path (remove _place.def suffix)
            if self.def_file_path:
                import os
                filename = os.path.basename(self.def_file_path)
                # Extract part before _place as design name
                design_name = filename.replace('_place.def', '')
            else:
                # Backup plan: use design name inside DEF file
                design_name = self.def_data.get('design_name', '')

            # CSV file path
            csv_path = 'place_data_extract.csv'

            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Improved design name matching logic
                    csv_design_name = row['design_name']

                    # 1. Exact match
                    if design_name == csv_design_name:
                        match_found = True
                    # 2. DEF design name in CSV name
                    elif design_name in csv_design_name:
                        match_found = True
                    # 3. CSV name in DEF design name
                    elif csv_design_name in design_name:
                        match_found = True
                    # 4. Handle bracket case, extract name in brackets for matching
                    elif '(' in csv_design_name and ')' in csv_design_name:
                        bracket_name = csv_design_name.split('(')[1].split(')')[0]
                        if design_name == bracket_name:
                            match_found = True
                        else:
                            match_found = False
                    # 5. Reverse bracket matching
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
                          design_name == csv_design_name + '_core'):
                        match_found = True
                    else:
                        match_found = False

                    if match_found:
                        core_util = row['core_utilization']
                        if core_util and core_util.strip():  # Check if empty
                            print(f"✅ Successfully matched design: '{design_name}' in DEF <-> '{csv_design_name}' in CSV (core_utilization={core_util})")
                            return float(core_util)

            print(f"⚠️  Warning: core_utilization data for design {design_name} not found in CSV file")
            return 0.0  # Default value
        except Exception as e:
            print(f"⚠️  Warning: Unable to read place_data_extract.csv file: {e}")
            return 0.0  # Default value
    
    def _read_placement_labels(self):
        """Read design_utilization and hpwl_after labels of corresponding design from place_data_extract.csv file"""
        try:
            # Extract design name from DEF file path (remove _place.def suffix)
            if self.def_file_path:
                import os
                filename = os.path.basename(self.def_file_path)
                # Extract part before _place as design name
                design_name = filename.replace('_place.def', '')
            else:
                # Backup plan: use design name inside DEF file
                design_name = self.def_data.get('design_name', '')

            # CSV file path
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
                    # 2. DEF design name in CSV name (e.g., ac97_top in ac97_ctrl does not match, but aes in systemcaes(aes) matches)
                    elif design_name in csv_design_name:
                        match_found = True
                    # 3. CSV name in DEF design name (e.g., ac97_ctrl in ac97_top does not match)
                    elif csv_design_name in design_name:
                        match_found = True
                    # 4. Handle bracket case, extract name in brackets for matching (e.g., vga_lcd(vga_enh_top) -> vga_enh_top)
                    elif '(' in csv_design_name and ')' in csv_design_name:
                        bracket_name = csv_design_name.split('(')[1].split(')')[0]
                        if design_name == bracket_name:
                            match_found = True
                        else:
                            match_found = False
                    # 5. Reverse bracket matching, check if DEF name matches part before bracket in CSV
                    elif '(' in csv_design_name:
                        prefix_name = csv_design_name.split('(')[0]
                        if design_name == prefix_name:
                            match_found = True
                        else:
                            match_found = False
                    # 6. Handle common name variants (e.g., tv80_core vs tv80, uart_top vs uart)
                    elif (design_name.replace('_top', '') == csv_design_name or
                          design_name.replace('_core', '') == csv_design_name or
                          design_name == csv_design_name + '_top' or
                          design_name == csv_design_name + '_core'):
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

                        print(f"✅ Successfully matched design: '{design_name}' in DEF <-> '{csv_design_name}' in CSV")
                        return utilization, hpwl

            print(f"⚠️  Warning: placement label data for design {design_name} not found in CSV file")
            return 0.0, 0.0  # Default value
        except Exception as e:
            print(f"⚠️  Warning: Unable to read place_data_extract.csv file: {e}")
            return 0.0, 0.0  # Default value

# ============================================================================
# GRAPH VALIDATOR SECTION - Graph validator area
# ============================================================================

class GraphValidator:
    """Graph structure validator"""
    
    @staticmethod
    def validate_heterograph(hetero_data: HeteroData) -> bool:
        """Validate integrity of heterograph structure"""
        try:
            print("🔍 Validating heterograph structure...")

            # Check node types - only validate four retained node types
            expected_node_types = ['gate', 'net', 'io_pin', 'pin']
            actual_node_types = list(hetero_data.node_types)

            print(f"  Node types: {actual_node_types}")

            for node_type in expected_node_types:
                if node_type in actual_node_types:
                    node_count = hetero_data[node_type].x.shape[0]
                    feature_dim = hetero_data[node_type].x.shape[1]
                    print(f"  ✓ {node_type}: {node_count} nodes, {feature_dim}-dimensional features")
                else:
                    print(f"  ⚠️  Missing node type: {node_type}")

            # Check edge types
            edge_types = list(hetero_data.edge_types)
            print(f"  Edge types: {edge_types}")

            for edge_type in edge_types:
                edge_count = hetero_data[edge_type].edge_index.shape[1]
                if hasattr(hetero_data[edge_type], 'edge_attr'):
                    edge_feature_dim = hetero_data[edge_type].edge_attr.shape[1]
                    print(f"  ✓ {edge_type}: {edge_count} edges, {edge_feature_dim}-dimensional features")
                else:
                    print(f"  ✓ {edge_type}: {edge_count} edges, no edge features")

            # Check global features
            if hasattr(hetero_data, 'global_features'):
                global_shape = hetero_data.global_features.shape
                print(f"  ✓ Global features: {global_shape}")

            # Check global labels
            if hasattr(hetero_data, 'y'):
                label_shape = hetero_data.y.shape
                print(f"  ✓ Global labels: {label_shape}")

            # Check chip coordinate information
            if hasattr(hetero_data, 'die_coordinates'):
                coord_shape = hetero_data.die_coordinates.shape
                print(f"  ✓ Chip coordinates: {coord_shape} (bottom_left+top_right)")

            print("✅ Heterograph validation passed")
            return True

        except Exception as e:
            print(f"❌ Error during validation: {e}")
            return False

# ============================================================================
# MAIN CONVERSION FUNCTION SECTION - Main conversion function area
# ============================================================================

def convert_def_to_basic_heterograph(def_file_path: str, output_path: str = None, graph_id: int = 0) -> HeteroData:
    """Main conversion function: DEF file to basic heterograph

    Args:
        def_file_path: Path to input DEF file
        output_path: Path to output .pt file (optional)

    Returns:
        HeteroData: Completed basic heterograph object

    Raises:
        RuntimeError: Raised when graph structure validation fails
        FileNotFoundError: Raised when DEF file does not exist
    """
    print("=" * 80)
    print("🚀 DEF to Basic Heterograph Converter")
    print("=" * 80)

    # 1. Parse DEF file
    print(f"📖 Parsing DEF file: {def_file_path}")
    parser = DEFParser(def_file_path)
    def_data = parser.parse()

    print(f"📊 Parsing results:")
    print(f"  - Design name: {def_data.get('design_name', 'unknown')}")
    print(f"  - Component count: {len(def_data.get('components', {}))}")
    print(f"  - Pin count: {len(def_data.get('pins', {}))}")
    print(f"  - Net count: {len(def_data.get('nets', {}))}")
    print(f"  - Layer count:   {len(def_data.get('tracks', {}))}")
    print()

    # 2. Build basic heterograph
    builder = BaseHeteroGraphBuilder(def_data, def_file_path)
    basic_graph = builder.build(graph_id)

    # 3. Validate graph structure
    if not GraphValidator.validate_heterograph(basic_graph):
        raise RuntimeError("Graph structure validation failed")

    # 4. Save results
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(def_file_path))[0]
        output_path = f"{base_name}_basic_heterograph.pt"

    print(f"💾 Saving basic heterograph to: {output_path}")
    torch.save(basic_graph, output_path, pickle_protocol=4)

    print("\n🎉 Conversion completed!")
    print(f"📈 Final graph structure statistics:")
    print(f"  - Node types: {len(basic_graph.node_types)}")
    print(f"  - Edge types: {len(basic_graph.edge_types)}")

    total_nodes = sum(basic_graph[nt].x.shape[0] for nt in basic_graph.node_types)
    total_edges = sum(basic_graph[et].edge_index.shape[1] for et in basic_graph.edge_types)
    print(f"  - Total nodes: {total_nodes}")
    print(f"  - Total edges: {total_edges}")

    # Display global feature information
    if hasattr(basic_graph, 'global_features'):
        global_feat = basic_graph.global_features
        print(f"  - Global features: {global_feat.shape[0]} dimensions")
        print(f"    * Chip width: {global_feat[0]:.0f}")
        print(f"    * Chip height: {global_feat[1]:.0f}")
        print(f"    * Chip area: {global_feat[2]:.0f}")
        print(f"    * DBU unit: {global_feat[3]:.0f}")

    return basic_graph

# ============================================================================
# COMMAND LINE INTERFACE SECTION - Command line interface area
# ============================================================================

def main():
    """Main function

    Main entry point for command line interface, processes command line arguments and calls conversion function.
    Supported command line arguments:
    - def_file: Path to input DEF file (positional parameter)
    - -o/--output: Path to output .pt file (optional)

    Returns:
        int: Program exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description='Basic Heterograph Generator - DEF to Basic Heterograph Conversion Tool')
    parser.add_argument('def_file', nargs='?', default='def_graph_place/place_def/bp_multi_place.def',
                       help='Path to input DEF file')
    parser.add_argument('-o', '--output', default=None,
                       help='Path to output .pt file')

    args = parser.parse_args()

    if not os.path.exists(args.def_file):
        print(f"❌ Error: DEF file does not exist: {args.def_file}")
        return 1

    try:
        basic_graph = convert_def_to_basic_heterograph(
            args.def_file, args.output
        )
        return 0
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())