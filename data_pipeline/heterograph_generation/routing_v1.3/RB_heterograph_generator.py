#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Routing Stage Heterogeneous Graph Generator (with internal pin support)

This script is based on B_heterograph_generator.py from place stage, specialized for generating routing stage heterogeneous graphs.
Main changes:
1. Global labels: Total HPWL wire length + Core Utilization + Pin Density + Total_wire_length + Total_vias
2. NET node labels: Actual wire length + Via count (replaces original HPWL calculation)

Function Description:
- Parse design information from routing stage DEF file
- Generate basic heterogeneous graph, containing the following node and edge types:
  * Node types: Gate, Net, IO_Pin, Pin
  * Edge types: IO_Pin-Net, Pin-Net, Gate-Pin
- Support internal pin parsing: extract unit internal port information from network connections
- Parse ROUTED section to calculate actual wire length and via count for each NET
- Read routing stage global data from route_data_extract.csv

Author: EDA for AI Team
Date: 2024

Usage:
    python R_heterograph_generator.py input_route.def -o output.pt
"""

import re
import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Any
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
    net_type: int = 0                          # Net type encoding (0-signal, 1-power, 2-ground, 3-clock, etc.)
    weight: float = 1.0                        # Net weight (based on connection complexity)
    wire_length: float = 0.0                   # Actual wire length (in DBU units)
    via_count: int = 0                         # Via count

@dataclass
class PinInfo:
    """Pin information data structure"""
    name: str                                   # Pin name
    component: str                              # Parent component name
    direction: str = 'INOUT'                    # Pin direction (INPUT, OUTPUT, INOUT, FEEDTHRU)
    layer: str = 'metal1'                       # Metal layer where the pin is located
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

    Responsible for parsing routing stage DEF files and extracting structured data, including:
    - Design information (name, units, die area)
    - Component information (position, type, orientation)
    - Pin information (position and direction of IO pins)
    - Net information (connection relationships and routing information)
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
            'internal_pins': {},  # Internal pin information
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
        # Support additional attributes such as SOURCE TIMING to ensure complete parsing of clock tree components
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

            # Parse layer information in PORT section
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
        # Parse normal nets (NETS section)
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
        """General method for parsing net sections"""
        # Improved net parsing - supports complex net names and connection formats, including dot characters
        # Match format: - net_name ( comp1 pin1 ) ( comp2 pin2 ) ... + USE SIGNAL ;
        net_pattern = r'-\s+([\w\[\]_$\\.]+)\s+((?:\([^)]+\)\s*)+)(.*?);'

        for match in re.finditer(net_pattern, nets_text, re.DOTALL):
            net_name = match.group(1)
            connections_text = match.group(2)
            routing_text = match.group(3)

            connections = []
            # Match each connection: ( component_name pin_name )
            conn_pattern = r'\(\s*([^\s]+)\s+([^\s]+)\s*\)'
            for conn_match in re.finditer(conn_pattern, connections_text):
                comp_name = conn_match.group(1)
                pin_name = conn_match.group(2)
                connections.append((comp_name, pin_name))

                # Extract internal pin information
                self._extract_internal_pin(comp_name, pin_name, net_name)

            # Parse routing information, calculate wire length and via count
            wire_length, via_count = self._parse_routing_info(routing_text)

            self.def_data['nets'][net_name] = {
                'connections': connections,
                'net_type': net_type,  # Use the passed net type
                'weight': 1.0,
                'wire_length': wire_length,
                'via_count': via_count
            }

    def _parse_routing_info(self, routing_text: str) -> Tuple[float, int]:
        """Parse routing information, calculate wire length and via count"""
        wire_length = 0.0
        via_count = 0

        # Find ROUTED section
        if '+ ROUTED' not in routing_text:
            return wire_length, via_count

        # Extract all routing segments
        # Match format: metal2 ( x1 y1 ) ( x2 y2 ) or metal2 ( x1 y1 ) ( * y2 ), etc.
        route_pattern = r'(metal\d+)\s+\(\s*([\d\-\*]+)\s+([\d\-\*]+)\s*\)\s+\(\s*([\d\-\*]+)\s+([\d\-\*]+)\s*\)'

        for match in re.finditer(route_pattern, routing_text):
            layer = match.group(1)
            x1_str, y1_str = match.group(2), match.group(3)
            x2_str, y2_str = match.group(4), match.group(5)

            # Handle coordinates, * means same as previous coordinate
            try:
                if x1_str != '*' and y1_str != '*' and x2_str != '*' and y2_str != '*':
                    x1, y1 = int(x1_str), int(y1_str)
                    x2, y2 = int(x2_str), int(y2_str)
                    # Calculate Manhattan distance
                    wire_length += abs(x2 - x1) + abs(y2 - y1)
                elif x1_str != '*' and y1_str != '*':
                    # Handle partial * cases, simplified processing
                    if x2_str != '*':
                        x1, x2 = int(x1_str), int(x2_str)
                        wire_length += abs(x2 - x1)
                    if y2_str != '*':
                        y1, y2 = int(y1_str), int(y2_str)
                        wire_length += abs(y2 - y1)
            except ValueError:
                # Skip unparsable coordinates
                continue

        # Count via occurrences
        via_pattern = r'(via\d+_\d+)'
        via_matches = re.findall(via_pattern, routing_text)
        via_count = len(via_matches)

        return wire_length, via_count
    
    def _parse_tracks(self, content: str):
        """Parse layer information - correctly handle TRACKS in X and Y directions"""
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
                # Use X direction data as main features (X direction is usually more important)
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
                # If main features not set yet, use Y direction data
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
                'layers': ['metal1', 'metal2'],  # Simplified processing
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
    """Feature configuration class - centralized management of all feature mappings and parameters"""

    # Use global mapping table
    # COMPLETE_CELL_TYPE_MAPPING is already defined globally

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
    PIN_TYPE_MAPPING = {
        # Basic input pin types (0-4) - continuous encoding
        'A': 0, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0,  # Data input A
        'B': 1, 'B1': 1, 'B2': 1, 'B3': 1, 'B4': 1,  # Data input B
        'C': 2, 'C1': 2, 'C2': 2, 'C3': 2, 'C4': 2,  # Data input C
        'D': 3, 'D1': 3, 'D2': 3, 'D3': 3, 'D4': 3,  # Data input D
        'S': 4, 'S0': 4, 'S1': 4,  # Selection signal

        # Control signal pin types (5-7) - continuous encoding
        'CI': 5, 'CIN': 5,  # Carry input
        'CLK': 6, 'CK': 6,  # Clock input - keep original encoding 6, as it's used in actual data
        'EN': 7, 'G': 7,    # Enable signal

        # Output pin types (8-11) - partially keep original encoding for compatibility with existing data
        'Y': 8, 'CO': 8, 'COUT': 8,  # Combinational logic output
        'Q': 9, 'QN': 9,             # Sequential logic output
        'Z': 10, 'ZN': 10,           # Main output - keep original encoding 10, heavily used in actual data
        'OUT': 11,                   # Other output - keep original encoding 11, used in actual data

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
# ENCODING UTILS SECTION - Encoding utility class area
# ============================================================================

class EncodingUtils:
    """Encoding utility class - provides unified encoding mapping functions"""

    # Reference global mapping configuration
    # COMPLETE_CELL_TYPE_MAPPING uses global definition
    ORIENTATION_MAPPING = FeatureConfig.ORIENTATION_MAPPING
    DIRECTION_MAPPING = FeatureConfig.PIN_DIRECTION_MAPPING
    PIN_TYPE_MAPPING = FeatureConfig.PIN_TYPE_MAPPING
    LAYER_MAPPING = FeatureConfig.LAYER_MAPPING
    NET_TYPE_MAPPING = FeatureConfig.NET_TYPE_MAPPING

    # 🔬 Process library actual area data (extracted directly from NangateOpenCellLibrary_typical.lib)
    # 📊 Data source: Extracted directly from NangateOpenCellLibrary_typical.lib process library
    # 📐 Unit: square micrometers (μm²)
    # ⚠️ Only includes cells that actually exist in the process library
    actual_cell_areas = {
        'AND2_X1': 1.064,
        'AND2_X2': 1.33,
        'AND2_X4': 2.394,
        'AND3_X1': 1.33,
        'AND3_X2': 1.596,
        'AND3_X4': 2.926,
        'AND4_X1': 1.596,
        'AND4_X2': 1.862,
        'AND4_X4': 3.458,
        'ANTENNA_X1': 0.266,
        'AOI211_X1': 1.33,
        'AOI211_X2': 2.394,
        'AOI211_X4': 2.926,
        'AOI21_X1': 1.064,
        'AOI21_X2': 1.862,
        'AOI21_X4': 3.458,
        'AOI221_X1': 1.596,
        'AOI221_X2': 2.926,
        'AOI221_X4': 3.458,
        'AOI222_X1': 2.128,
        'AOI222_X2': 3.724,
        'AOI222_X4': 3.724,
        'AOI22_X1': 1.33,
        'AOI22_X2': 2.394,
        'AOI22_X4': 4.522,
        'BUF_X1': 0.798,
        'BUF_X16': 6.65,
        'BUF_X2': 1.064,
        'BUF_X32': 13.034,
        'BUF_X4': 1.862,
        'BUF_X8': 3.458,
        'CLKBUF_X1': 0.798,
        'CLKBUF_X2': 1.064,
        'CLKBUF_X3': 1.33,
        'CLKGATETST_X1': 3.99,
        'CLKGATETST_X2': 4.256,
        'CLKGATETST_X4': 5.32,
        'CLKGATETST_X8': 7.714,
        'CLKGATE_X1': 3.458,
        'CLKGATE_X2': 3.724,
        'CLKGATE_X4': 4.522,
        'CLKGATE_X8': 6.916,
        'DFFRS_X1': 6.384,
        'DFFRS_X2': 6.916,
        'DFFR_X1': 5.32,
        'DFFR_X2': 5.852,
        'DFFS_X1': 5.32,
        'DFFS_X2': 5.586,
        'DFF_X1': 4.522,
        'DFF_X2': 5.054,
        'DLH_X1': 2.66,
        'DLH_X2': 2.926,
        'DLL_X1': 2.66,
        'DLL_X2': 2.926,
        'FA_X1': 4.256,
        'HA_X1': 2.66,
        'INV_X1': 0.532,
        'INV_X16': 4.522,
        'INV_X2': 0.798,
        'INV_X32': 8.778,
        'INV_X4': 1.33,
        'INV_X8': 2.394,
        'MUX2_X1': 1.862,
        'MUX2_X2': 2.394,
        'NAND2_X1': 0.798,
        'NAND2_X2': 1.33,
        'NAND2_X4': 2.394,
        'NAND3_X1': 1.064,
        'NAND3_X2': 1.862,
        'NAND3_X4': 3.458,
        'NAND4_X1': 1.33,
        'NAND4_X2': 2.394,
        'NAND4_X4': 4.788,
        'NOR2_X1': 0.798,
        'NOR2_X2': 1.33,
        'NOR2_X4': 2.394,
        'NOR3_X1': 1.064,
        'NOR3_X2': 1.862,
        'NOR3_X4': 3.724,
        'NOR4_X1': 1.33,
        'NOR4_X2': 2.394,
        'NOR4_X4': 4.788,
        'OAI211_X1': 1.33,
        'OAI211_X2': 2.394,
        'OAI211_X4': 4.522,
        'OAI21_X1': 1.064,
        'OAI21_X2': 1.862,
        'OAI21_X4': 3.458,
        'OAI221_X1': 1.596,
        'OAI221_X2': 2.926,
        'OAI221_X4': 3.458,
        'OAI222_X1': 2.128,
        'OAI222_X2': 3.724,
        'OAI222_X4': 3.724,
        'OAI22_X1': 1.33,
        'OAI22_X2': 2.394,
        'OAI22_X4': 4.522,
        'OAI33_X1': 1.862,
        'OR2_X1': 1.064,
        'OR2_X2': 1.33,
        'OR2_X4': 2.394,
        'OR3_X1': 1.33,
        'OR3_X2': 1.596,
        'OR3_X4': 2.926,
        'OR4_X1': 1.596,
        'OR4_X2': 1.862,
        'OR4_X4': 3.458,
        'SDFFRS_X1': 7.714,
        'SDFFRS_X2': 8.246,
        'SDFFR_X1': 6.65,
        'SDFFR_X2': 6.916,
        'SDFFS_X1': 6.65,
        'SDFFS_X2': 7.182,
        'SDFF_X1': 6.118,
        'SDFF_X2': 6.384,
        'TBUF_X1': 2.128,
        'TBUF_X16': 6.916,
        'TBUF_X2': 2.394,
        'TBUF_X4': 2.926,
        'TBUF_X8': 4.788,
        'TINV_X1': 1.064,
        'TLAT_X1': 3.458,
        'XNOR2_X1': 1.596,
        'XNOR2_X2': 2.66,
        'XOR2_X1': 1.596,
    }
    
    # ⚡ Process library actual power data (extracted directly from NangateOpenCellLibrary_typical.lib)
    # 📊 Data source: Extracted directly from NangateOpenCellLibrary_typical.lib process library
    # ⚠️ Only includes cells that actually exist in the process library
    real_cell_powers = {
        'AND2_X1': 25.066064,
        'AND2_X2': 50.35316,
        'AND2_X4': 100.706457,
        'AND3_X1': 26.48146,
        'AND3_X2': 53.19027,
        'AND3_X4': 106.380663,
        'AND4_X1': 27.024804,
        'AND4_X2': 54.274743,
        'AND4_X4': 108.54959,
        'ANTENNA_X1': 0.0,
        'AOI211_X1': 34.565711,
        'AOI211_X2': 69.131362,
        'AOI211_X4': 126.001026,
        'AOI21_X1': 27.858395,
        'AOI21_X2': 55.71672,
        'AOI21_X4': 111.433338,
        'AOI221_X1': 41.741212,
        'AOI221_X2': 83.482383,
        'AOI221_X4': 131.584293,
        'AOI222_X1': 47.398844,
        'AOI222_X2': 94.797579,
        'AOI222_X4': 134.853561,
        'AOI22_X1': 32.611944,
        'AOI22_X2': 65.223838,
        'AOI22_X4': 130.447551,
        'BUF_X1': 21.438247,
        'BUF_X16': 344.4881,
        'BUF_X2': 43.06082,
        'BUF_X32': 688.9762,
        'BUF_X4': 86.121805,
        'BUF_X8': 172.244545,
        'CLKBUF_X1': 11.214093,
        'CLKBUF_X2': 22.91762,
        'CLKBUF_X3': 30.558215,
        'CLKGATETST_X1': 59.137918,
        'CLKGATETST_X2': 78.027946,
        'CLKGATETST_X4': 126.648876,
        'CLKGATETST_X8': 220.271011,
        'CLKGATE_X1': 48.64726,
        'CLKGATE_X2': 66.637175,
        'CLKGATE_X4': 109.65183,
        'CLKGATE_X8': 197.707057,
        'DFFRS_X1': 100.161505,
        'DFFRS_X2': 142.302832,
        'DFFR_X1': 86.21278,
        'DFFR_X2': 125.988288,
        'DFFS_X1': 84.395957,
        'DFFS_X2': 121.107028,
        'DFF_X1': 79.112308,
        'DFF_X2': 115.10367,
        'DLH_X1': 40.86324,
        'DLH_X2': 57.430452,
        'DLL_X1': 40.863416,
        'DLL_X2': 57.430445,
        'FA_X1': 75.762253,
        'HA_X1': 61.229735,
        'INV_X1': 14.353185,
        'INV_X16': 229.651455,
        'INV_X2': 28.706376,
        'INV_X32': 459.3028,
        'INV_X4': 57.41285,
        'INV_X8': 114.826305,
        'MUX2_X1': 35.92839,
        'MUX2_X2': 68.648566,
        'NAND2_X1': 17.39336,
        'NAND2_X2': 34.78663,
        'NAND2_X4': 69.57324,
        'NAND3_X1': 18.104768,
        'NAND3_X2': 36.209558,
        'NAND3_X4': 72.419123,
        'NAND4_X1': 18.126843,
        'NAND4_X2': 36.253723,
        'NAND4_X4': 72.506878,
        'NOR2_X1': 21.199545,
        'NOR2_X2': 42.399074,
        'NOR2_X4': 84.798143,
        'NOR3_X1': 26.831667,
        'NOR3_X2': 53.663264,
        'NOR3_X4': 107.325918,
        'NOR4_X1': 32.601474,
        'NOR4_X2': 65.202889,
        'NOR4_X4': 130.405147,
        'OAI211_X1': 22.039133,
        'OAI211_X2': 44.0782,
        'OAI211_X4': 88.156278,
        'OAI21_X1': 22.619394,
        'OAI21_X2': 45.238687,
        'OAI21_X4': 90.477187,
        'OAI221_X1': 33.937672,
        'OAI221_X2': 67.875298,
        'OAI221_X4': 116.346072,
        'OAI222_X1': 43.177867,
        'OAI222_X2': 86.355577,
        'OAI222_X4': 127.97856,
        'OAI22_X1': 34.026125,
        'OAI22_X2': 68.052131,
        'OAI22_X4': 136.103946,
        'OAI33_X1': 48.313952,
        'OR2_X1': 22.694975,
        'OR2_X2': 45.656022,
        'OR2_X4': 91.312375,
        'OR3_X1': 24.414625,
        'OR3_X2': 49.162437,
        'OR3_X4': 98.32515,
        'OR4_X1': 26.73349,
        'OR4_X2': 53.869509,
        'OR4_X4': 107.739253,
        'SDFFRS_X1': 122.721721,
        'SDFFRS_X2': 162.30086,
        'SDFFR_X1': 105.258197,
        'SDFFR_X2': 141.961514,
        'SDFFS_X1': 107.724855,
        'SDFFS_X2': 140.592991,
        'SDFF_X1': 100.684799,
        'SDFF_X2': 136.676074,
        'TBUF_X1': 32.749437,
        'TBUF_X16': 197.273423,
        'TBUF_X2': 58.901131,
        'TBUF_X4': 72.121528,
        'TBUF_X8': 144.426508,
        'TINV_X1': 17.885841,
        'TLAT_X1': 47.99611,
        'XNOR2_X1': 36.441009,
        'XNOR2_X2': 73.102975,
        'XOR2_X1': 36.163718,
    }
    
    @classmethod
    def encode_cell_type(cls, cell_type: str) -> int:
        """Encode complete cell type to integer - use global mapping table"""
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
        """Encode pin type to integer - infer type based on pin name"""
        pin_name_upper = pin_name.upper()

        # Traverse mapping table to find matching type
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
        net_name_upper = net_name.upper()

        if any(keyword in net_name_upper for keyword in ['VDD', 'VCC', 'VPWR', 'POWER']):
            return cls.NET_TYPE_MAPPING['power']
        elif any(keyword in net_name_upper for keyword in ['VSS', 'GND', 'VGND', 'GROUND']):
            return cls.NET_TYPE_MAPPING['ground']
        elif any(keyword in net_name_upper for keyword in ['CLK', 'CLOCK']):
            return cls.NET_TYPE_MAPPING['clock']
        elif any(keyword in net_name_upper for keyword in ['RST', 'RESET']):
            return cls.NET_TYPE_MAPPING['reset']
        elif any(keyword in net_name_upper for keyword in ['SCAN', 'TEST']):
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
            'COVER': 2,    # Cover placed
            'UNPLACED': 3  # Unplaced
        }
        return placement_mapping.get(placement_status, 0)

    @classmethod
    def calculate_component_size(cls, cell_type: str, def_data: Dict = None) -> float:
        """Calculate component area (based on real process library data)"""
        # DBU conversion factor: dynamically obtained from DEF file UNITS DISTANCE MICRONS
        if def_data and 'units' in def_data:
            DBU_PER_MICRON = float(def_data['units'].get('dbu_per_micron', 2000))
        else:
            DBU_PER_MICRON = 2000  # Default value

        # 1 um = DBU_PER_MICRON DBU, therefore 1 um² = (DBU_PER_MICRON)² DBU²
        AREA_CONVERSION_FACTOR = DBU_PER_MICRON * DBU_PER_MICRON

        normalized_cell_type = cell_type.upper()
        actual_area_um2 = cls.actual_cell_areas.get(normalized_cell_type, 0.532)  # Default INV_X1 area

        # Convert to DBU² units to align with coordinate units
        area_dbu2 = actual_area_um2 * AREA_CONVERSION_FACTOR
        return area_dbu2

    @classmethod
    def calculate_cell_power(cls, cell_type: str) -> float:
        """Calculate component power (based on real process library data)"""
        normalized_cell_type = cell_type.upper()
        return cls.real_cell_powers.get(normalized_cell_type, 15)  # Default to INV_X1 power

# ============================================================================
# HETEROGRAPH GENERATOR SECTION - Heterograph generator area
# ============================================================================

class RoutingHeterographGenerator:
    """Routing stage heterogeneous graph generator

    Responsible for converting DEF files to PyTorch Geometric heterogeneous graph format, specifically handling routing stage features and labels.
    """

    # ==================== RB graph customized type mapping definition ====================
    # Based on actual usage of RB script in node_edge_analysis_report.md

    @property
    def node_type_to_id(self) -> Dict[str, int]:
        """RB graph node type to ID mapping"""
        return {
            "gate": 0,
            "io_pin": 1,
            "net": 2,
            "pin": 3
        }

    @property
    def edge_type_to_id(self) -> Dict[str, int]:
        """RB graph edge type to ID mapping"""
        return {
            "('gate', 'has', 'pin')": 2,
            "('io_pin', 'connects_to', 'net')": 4,
            "('pin', 'connects_to', 'net')": 6
        }

    # ========================================================
    
    def __init__(self, def_file_path: str):
        self.def_file_path = def_file_path
        self.def_data = {}
        self.hetero_data = HeteroData()

        # Node mapping table
        self.component_to_idx = {}
        self.net_to_idx = {}
        self.io_pin_to_idx = {}
        self.internal_pin_to_idx = {}

        # Die area information
        self.die_area = (0, 0, 0, 0)
    
    def generate(self) -> HeteroData:
        """Generate heterogeneous graph"""
        print("🚀 Starting to generate Routing stage heterogeneous graph...")

        # 1. Parse DEF file
        print("📖 Parsing DEF file...")
        parser = DEFParser(self.def_file_path)
        self.def_data = parser.parse()
        self.die_area = self.def_data['die_area']

        # 2. Create node mappings
        print("🗺️  Creating node mappings...")
        self._create_node_mappings()

        # 3. Add node features - in standard order Gate -> IO_Pin -> Net -> Pin
        print("🎯 Adding node features...")
        self._add_gate_features()
        self._add_io_pin_features()
        self._add_net_features()
        self._add_internal_pin_features()

        # 4. Add edges
        print("🔗 Adding edges...")
        self._add_edges()

        # 5. Add global features and labels
        print("🌐 Adding global features and labels...")
        self._add_global_features()

        print("✅ Routing stage heterogeneous graph generation complete!")
        self._print_graph_summary()

        return self.hetero_data

    def _create_node_mappings(self):
        """Create node mapping table"""
        # Gate node mapping - use same filtering criteria as RF graph
        components = self.def_data.get('components', {})
        # Filter out true gate components (exclude fill cells, etc.)
        gate_components = {}
        for comp_name, comp_info in components.items():
            cell_type = comp_info['cell_type']
            if (not cell_type.startswith('FILLCELL') and
                not cell_type.startswith('ANTENNA') and
                not cell_type.startswith('TAPCELL')):
                gate_components[comp_name] = comp_info
        self.component_to_idx = {name: idx for idx, name in enumerate(gate_components.keys())}

        # Net node mapping
        nets = self.def_data.get('nets', {})
        self.net_to_idx = {name: idx for idx, name in enumerate(nets.keys())}

        # IO Pin node mapping
        io_pins = self.def_data.get('pins', {})
        self.io_pin_to_idx = {name: idx for idx, name in enumerate(io_pins.keys())}

        # Internal Pin node mapping
        internal_pins = self.def_data.get('internal_pins', {})
        self.internal_pin_to_idx = {name: idx for idx, name in enumerate(internal_pins.keys())}

        print(f"  ✓ Gate nodes: {len(self.component_to_idx)}")
        print(f"  ✓ Net nodes: {len(self.net_to_idx)}")
        print(f"  ✓ IO Pin nodes: {len(self.io_pin_to_idx)}")
        print(f"  ✓ Internal Pin nodes: {len(self.internal_pin_to_idx)}")
    
    def _add_gate_features(self):
        """Add Gate node features"""
        components = self.def_data.get('components', {})
        if not components:
            return

        # Filter out true gate components (exclude fill cells, etc.)
        gate_components = {}
        for comp_name, comp_info in components.items():
            cell_type = comp_info['cell_type']
            if (not cell_type.startswith('FILLCELL') and
                not cell_type.startswith('ANTENNA') and
                not cell_type.startswith('TAPCELL')):
                gate_components[comp_name] = comp_info

        features = []
        for comp_name in self.component_to_idx.keys():
            comp_info = gate_components[comp_name]

            # 7-dimensional feature vector: [x, y, cell_type_id, orientation_id, area, placement_status, power]
            x, y = comp_info['position']
            cell_type_id = EncodingUtils.encode_cell_type(comp_info['cell_type'])
            orientation_id = EncodingUtils.encode_orientation(comp_info['orientation'])
            area = EncodingUtils.calculate_component_size(comp_info['cell_type'], self.def_data)
            placement_status = EncodingUtils.encode_placement_status(comp_info.get('placement_status', 'PLACED'))  # All are PLACED in routing stage
            power = EncodingUtils.calculate_cell_power(comp_info['cell_type'])

            feature_vector = [float(x), float(y), cell_type_id, orientation_id, area, placement_status, power]
            features.append(feature_vector)

        self.hetero_data['gate'].x = torch.tensor(features, dtype=torch.float)
        print(f"  ✓ Gate features: {len(features)} nodes, 7-dimensional features")
    
    def _calculate_hpwl(self, net_name: str, net_info: Dict) -> float:
        """Calculate network HPWL (Half-Perimeter Wire Length)

        Args:
            net_name: Network name
            net_info: Network information dictionary

        Returns:
            float: HPWL value, calculation formula is (x_max - x_min) + (y_max - y_min)
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

    def _add_net_features(self):
        """Add Net node features and labels"""
        nets = self.def_data.get('nets', {})
        if not nets:
            return

        features = []
        labels = []

        for net_name, net_info in nets.items():
            # Feature vector: [net_type_id, connection_count, HPWL] - extended to 3-dimensional feature vector
            net_type_id = EncodingUtils.encode_net_type(net_name)
            connections = net_info.get('connections', [])
            connection_count = len(connections)
            hpwl = self._calculate_hpwl(net_name, net_info)

            feature_vector = [net_type_id, connection_count, hpwl]
            features.append(feature_vector)

            # Label: [actual wire length, via count] - keep as 2-dimensional label vector
            wire_length = net_info.get('wire_length', 0.0)
            via_count = net_info.get('via_count', 0)
            label_vector = [wire_length, float(via_count)]
            labels.append(label_vector)

        self.hetero_data['net'].x = torch.tensor(features, dtype=torch.float)
        self.hetero_data['net'].y = torch.tensor(labels, dtype=torch.float)

        print(f"  ✓ Net features: {len(features)} nodes, 3-dimensional features (net_type, connection_count, HPWL)")
        print(f"  ✓ Net labels: {len(labels)} nodes, 2-dimensional labels (wire length, via count)")
    
    def _add_io_pin_features(self):
        """Add IO Pin node features"""
        io_pins = self.def_data.get('pins', {})
        if not io_pins:
            return

        features = []
        for pin_name in self.io_pin_to_idx.keys():
            pin_info = io_pins[pin_name]

            # Feature vector: [x, y, direction_id, layer_id] - consistent with place stage
            x, y = pin_info['position']
            direction_id = EncodingUtils.encode_pin_direction(pin_info['direction'])
            layer_id = EncodingUtils.encode_layer(pin_info['layer'])

            feature_vector = [float(x), float(y), direction_id, layer_id]
            features.append(feature_vector)

        self.hetero_data['io_pin'].x = torch.tensor(features, dtype=torch.float)
        print(f"  ✓ IO Pin features: {len(features)} nodes, 4-dimensional features")

    def _add_internal_pin_features(self):
        """Add Internal Pin node features"""
        internal_pins = self.def_data.get('internal_pins', {})
        if not internal_pins:
            return

        features = []
        for pin_id in self.internal_pin_to_idx.keys():
            pin_info = internal_pins[pin_id]

            # Feature vector: [pin_type_id, cell_type_id] - consistent with place stage
            pin_type_id = EncodingUtils.encode_pin_type(pin_info['name'])

            # Get type of parent component
            comp_name = pin_info.get('component', '')
            comp_info = self.def_data.get('components', {}).get(comp_name, {})
            cell_type_id = EncodingUtils.encode_cell_type(comp_info.get('cell_type', ''))

            feature_vector = [pin_type_id, cell_type_id]
            features.append(feature_vector)

        self.hetero_data['pin'].x = torch.tensor(features, dtype=torch.float)
        print(f"  ✓ Internal Pin features: {len(features)} nodes, 2-dimensional features")

    def _add_edges(self):
        """Add edges - in standard order"""
        # 1. Gate-Pin edges (ID: 2)
        self._add_gate_pin_edges()

        # 2. IO_Pin-Net edges (ID: 4)
        self._add_io_pin_net_edges()

        # 3. Pin-Net edges (ID: 6)
        self._add_pin_net_edges()
    
    def _add_io_pin_net_edges(self):
        """Add IO Pin to Net edges"""
        io_pins = self.def_data.get('pins', {})

        source_nodes = []
        target_nodes = []
        edge_features = []

        for pin_name, pin_info in io_pins.items():
            net_name = pin_info['net']

            if pin_name in self.io_pin_to_idx and net_name in self.net_to_idx:
                pin_idx = self.io_pin_to_idx[pin_name]
                net_idx = self.net_to_idx[net_name]

                source_nodes.append(pin_idx)
                target_nodes.append(net_idx)

                # Edge features: [pin_direction_id, net_type_id]
                pin_direction_id = EncodingUtils.encode_pin_direction(pin_info['direction'])
                net_type_id = EncodingUtils.encode_net_type(net_name)

                edge_features.append([pin_direction_id, net_type_id])

        if source_nodes:
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            self.hetero_data['io_pin', 'connects_to', 'net'].edge_index = edge_index
            self.hetero_data['io_pin', 'connects_to', 'net'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
            print(f"  ✓ IO_Pin-Net edges: {len(source_nodes)} edges")

    def _add_pin_net_edges(self):
        """Add Internal Pin to Net edges"""
        internal_pins = self.def_data.get('internal_pins', {})

        source_nodes = []
        target_nodes = []
        edge_features = []

        for pin_id, pin_info in internal_pins.items():
            net_name = pin_info['net']

            if pin_id in self.internal_pin_to_idx and net_name in self.net_to_idx:
                pin_idx = self.internal_pin_to_idx[pin_id]
                net_idx = self.net_to_idx[net_name]

                source_nodes.append(pin_idx)
                target_nodes.append(net_idx)

                # Edge features: [pin_type_id, net_type_id]
                pin_type_id = EncodingUtils.encode_pin_type(pin_info['name'])
                net_type_id = EncodingUtils.encode_net_type(net_name)

                edge_features.append([pin_type_id, net_type_id])

        if source_nodes:
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            self.hetero_data['pin', 'connects_to', 'net'].edge_index = edge_index
            self.hetero_data['pin', 'connects_to', 'net'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
            print(f"  ✓ Pin-Net edges: {len(source_nodes)} edges")

    def _add_gate_pin_edges(self):
        """Add Gate to Internal Pin edges"""
        internal_pins = self.def_data.get('internal_pins', {})

        source_nodes = []
        target_nodes = []
        edge_features = []

        for pin_id, pin_info in internal_pins.items():
            comp_name = pin_info['component']

            if comp_name in self.component_to_idx and pin_id in self.internal_pin_to_idx:
                comp_idx = self.component_to_idx[comp_name]
                pin_idx = self.internal_pin_to_idx[pin_id]

                source_nodes.append(comp_idx)
                target_nodes.append(pin_idx)

                # Edge features: [cell_type_id, pin_type_id]
                comp_info = self.def_data['components'].get(comp_name, {})
                cell_type_id = EncodingUtils.encode_cell_type(comp_info.get('cell_type', ''))
                pin_type_id = EncodingUtils.encode_pin_type(pin_info['name'])

                edge_features.append([cell_type_id, pin_type_id])

        if source_nodes:
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            self.hetero_data['gate', 'has', 'pin'].edge_index = edge_index
            self.hetero_data['gate', 'has', 'pin'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
            print(f"  ✓ Gate-Pin edges: {len(source_nodes)} edges")
    
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
        self.hetero_data['global_features'] = global_features

        # Chip coordinate information - 2x2 matrix: [[bottom-left x, bottom-left y], [top-right x, top-right y]]
        die_coordinates = torch.tensor([[die_area[0], die_area[1]], [die_area[2], die_area[3]]], dtype=torch.float)
        self.hetero_data['die_coordinates'] = die_coordinates

        # Read global labels from routing report
        utilization, hpwl, total_wire_length, total_vias = self._read_routing_labels()

        # Calculate Pin Density (internal pin utilization rate)
        internal_pin_count = len(self.def_data.get('internal_pins', {}))
        total_pin_count = internal_pin_count + len(self.def_data.get('pins', {}))
        pin_density = (internal_pin_count / total_pin_count * 100.0) if total_pin_count > 0 else 0.0

        # Global labels: [actual utilization, HPWL, Pin Density, Total Wire Length, Total Vias]
        y = torch.tensor([utilization, hpwl, pin_density, total_wire_length, total_vias], dtype=torch.float)
        self.hetero_data.y = y

        print(f"  ✓ Global features: 5-dimensional (width={chip_width}, height={chip_height}, area={chip_area}, DBU={dbu_per_micron}, configured utilization={core_utilization_config})")
        print(f"  ✓ Global labels: 5-dimensional (actual utilization={utilization}%, HPWL={hpwl}um, Pin density={pin_density:.2f}%, total wire length={total_wire_length}, total vias={total_vias})")
    
    def _read_config_utilization(self):
        """Read CORE_UTILIZATION configuration for corresponding design from route_data_extract.csv file"""
        try:
            # Extract design name from DEF file path (remove _route.def suffix)
            if self.def_file_path:
                import os
                filename = os.path.basename(self.def_file_path)
                # Extract part before _route as design name
                design_name = filename.replace('_route.def', '')
            else:
                # Fallback: use design name inside DEF file
                design_name = self.def_data.get('design_name', '')

            # CSV file path
            csv_path = 'route_data_extract.csv'

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
                    # 4. Handle bracket cases, extract name inside brackets for matching
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

            print(f"⚠️  Warning: core_utilization data not found for design {design_name} in CSV file")
            return 0.0  # Default value
        except Exception as e:
            print(f"⚠️  Warning: Unable to read route_data_extract.csv file: {e}")
            return 0.0  # Default value
    
    def _read_routing_labels(self):
        """Read routing stage labels for corresponding design from route_data_extract.csv file"""
        try:
            # Extract design name from DEF file path (remove _route.def suffix)
            if self.def_file_path:
                import os
                filename = os.path.basename(self.def_file_path)
                # Extract part before _route as design name
                design_name = filename.replace('_route.def', '')
            else:
                # Fallback: use design name inside DEF file
                design_name = self.def_data.get('design_name', '')

            # CSV file path
            csv_path = 'route_data_extract.csv'

            utilization = 0.0
            hpwl = 0.0
            total_wire_length = 0.0
            total_vias = 0.0

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
                    # 4. Handle bracket cases, extract name inside brackets for matching
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
                        # Read design_utilization
                        design_util = row['design_utilization']
                        if design_util and design_util.strip():
                            utilization = float(design_util)

                        # Read hpwl_after
                        hpwl_after = row['hpwl_after']
                        if hpwl_after and hpwl_after.strip():
                            hpwl = float(hpwl_after)

                        # Read total_wire_length
                        wire_length = row['total_wire_length']
                        if wire_length and wire_length.strip():
                            total_wire_length = float(wire_length)

                        # Read total_vias
                        vias = row['total_vias']
                        if vias and vias.strip():
                            total_vias = float(vias)

                        print(f"✅ Successfully matched design: '{design_name}' in DEF <-> '{csv_design_name}' in CSV")
                        return utilization, hpwl, total_wire_length, total_vias

            print(f"⚠️  Warning: Routing label data not found for design {design_name} in CSV file")
            return 0.0, 0.0, 0.0, 0.0  # Default values
        except Exception as e:
            print(f"⚠️  Warning: Unable to read route_data_extract.csv file: {e}")
            return 0.0, 0.0, 0.0, 0.0  # Default values
    
    def _print_graph_summary(self):
        """Print graph structure summary"""
        print("\n📊 Heterogeneous graph structure summary:")
        print("=" * 50)

        # Node information
        print("🎯 Node information:")
        for node_type in self.hetero_data.node_types:
            if hasattr(self.hetero_data[node_type], 'x'):
                num_nodes = self.hetero_data[node_type].x.size(0)
                num_features = self.hetero_data[node_type].x.size(1)
                print(f"  {node_type}: {num_nodes} nodes, {num_features}-dimensional features")

        # Edge information
        print("\n🔗 Edge information:")
        for edge_type in self.hetero_data.edge_types:
            if hasattr(self.hetero_data[edge_type], 'edge_index'):
                num_edges = self.hetero_data[edge_type].edge_index.size(1)
                print(f"  {edge_type}: {num_edges} edges")

        # Global information
        print("\n🌐 Global information:")
        if hasattr(self.hetero_data, 'global_features'):
            print(f"  Global features: {self.hetero_data.global_features.size(0)}-dimensional")
        if hasattr(self.hetero_data, 'y'):
            print(f"  Global labels: {self.hetero_data.y.size(0)}-dimensional")

        print("=" * 50)

# ============================================================================
# MAIN FUNCTION SECTION - Main function area
# ============================================================================

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Routing stage heterogeneous graph generator')
    parser.add_argument('def_file', help='Input routing stage DEF file path')
    parser.add_argument('-o', '--output', help='Output heterogeneous graph file path', default='routing_heterograph.pt')

    args = parser.parse_args()

    try:
        # Check if input file exists
        if not os.path.exists(args.def_file):
            print(f"❌ Error: DEF file does not exist: {args.def_file}")
            return

        # Generate heterogeneous graph
        generator = RoutingHeterographGenerator(args.def_file)
        hetero_data = generator.generate()

        # Save heterogeneous graph
        torch.save(hetero_data, args.output)
        print(f"💾 Heterogeneous graph saved to: {args.output}")

        # Verify saved file
        loaded_data = torch.load(args.output, weights_only=False)
        print(f"✅ Verification successful: File size {os.path.getsize(args.output)} bytes")

    except Exception as e:
        print(f"❌ Error occurred while generating heterogeneous graph: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()