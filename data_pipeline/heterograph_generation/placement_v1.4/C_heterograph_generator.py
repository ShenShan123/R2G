#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C-Graph Heterogeneous Graph Generator - DEF to C-Graph Heterogeneous Graph Conversion Tool

Author: EDA for AI Team

Function Description:
- Based on B-graph generator, removes Pin nodes
- Removes Pin-Net and Gate-Pin edge types
- Adds Gate-Net edges with features [pin_type_id, cell_type_id] (inherits B-graph pin node features)
- Retains Gate, Net, and IO_Pin node types
- Retains IO_Pin-Net edges

Supported Node Types:
- Gate: Logic gate nodes, containing position, type, orientation, etc.
- Net: Network nodes, representing connections
- IO_Pin: Input/output pin nodes

Supported Edge Types:
- Gate-Net: Direct connection from gate to network
- IO_Pin-Net: Connection from IO pin to network
"""

import os
import sys
import argparse
import re
import csv
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

import torch
from torch_geometric.data import HeteroData

# ============================================================================
# DATA STRUCTURES SECTION - Data structure definition area
# ============================================================================

@dataclass
class ComponentInfo:
    """Component information data class"""
    name: str
    cell_type: str
    x: float
    y: float
    orientation: str
    placement_status: str

@dataclass
class NetInfo:
    """Net information data class"""
    name: str
    connections: List[str]

@dataclass
class PinInfo:
    """Pin information data class"""
    name: str
    net: str
    direction: str
    layer: str
    x: float
    y: float

@dataclass
class InternalPinInfo:
    """Internal pin information data class"""
    pin_id: str
    component: str
    name: str
    net: str
    x: float
    y: float

# ============================================================================
# DEF PARSER SECTION - DEF file parsing area
# ============================================================================

class DEFParser:
    """DEF file parser

    Responsible for parsing DEF files and extracting required design information, including components, networks, pins, etc.
    """

    def __init__(self, def_file_path: str):
        """Initialize DEF parser

        Args:
            def_file_path: DEF file path
        """
        self.def_file_path = def_file_path
        self.data = {
            'design_name': '',
            'units': {},
            'die_area': [],
            'tracks': {},
            'components': {},
            'pins': {},
            'nets': {},
            'internal_pins': {}  # Internal pin information
        }

    def parse(self) -> Dict[str, Any]:
        """Parse DEF file

        Returns:
            Dictionary containing all parsed data
        """
        print(f"📖 Starting to parse DEF file: {self.def_file_path}")

        with open(self.def_file_path, 'r') as f:
            content = f.read()

        # Parse each section
        self._parse_design_name(content)
        self._parse_units(content)
        self._parse_die_area(content)
        self._parse_tracks(content)
        self._parse_components(content)
        self._parse_pins(content)
        self._parse_nets(content)

        print(f"✅ DEF file parsing complete")
        return self.data
    
    def _parse_design_name(self, content: str):
        """Parse design name"""
        match = re.search(r'DESIGN\s+(\w+)', content)
        if match:
            self.data['design_name'] = match.group(1)

    def _parse_units(self, content: str):
        """Parse unit information"""
        match = re.search(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', content)
        if match:
            self.data['units'] = {'dbu_per_micron': int(match.group(1))}
        else:
            self.data['units'] = {'dbu_per_micron': 1000}  # Default value

    def _parse_die_area(self, content: str):
        """Parse die area"""
        match = re.search(r'DIEAREA\s+\(\s*([\d\-]+)\s+([\d\-]+)\s*\)\s+\(\s*([\d\-]+)\s+([\d\-]+)\s*\)', content)
        if match:
            self.data['die_area'] = [int(match.group(1)), int(match.group(2)),
                                   int(match.group(3)), int(match.group(4))]

    def _parse_tracks(self, content: str):
        """Parse track information"""
        tracks = {}
        track_matches = re.findall(r'TRACKS\s+(\w+)\s+([\d\-]+)\s+DO\s+(\d+)\s+STEP\s+(\d+)\s+LAYER\s+(\w+)', content)
        for match in track_matches:
            direction, start, count, step, layer = match
            tracks[layer] = {
                'direction': direction,
                'start': int(start),
                'count': int(count),
                'step': int(step)
            }
        self.data['tracks'] = tracks

    def _parse_components(self, content: str):
        """Parse component information"""
        components = {}

        # Find COMPONENTS section
        comp_section = re.search(r'COMPONENTS\s+(\d+)\s*;(.*?)END\s+COMPONENTS', content, re.DOTALL)
        if not comp_section:
            return

        comp_content = comp_section.group(2)

        # Parse each component
        comp_matches = re.findall(r'-\s+([^\s]+)\s+(\w+)\s+.*?\+\s+PLACED\s+\(\s*([\d\-]+)\s+([\d\-]+)\s*\)\s+(\w+)', comp_content)
        for match in comp_matches:
            comp_name, cell_type, x, y, orientation = match
            components[comp_name] = ComponentInfo(
                name=comp_name,
                cell_type=cell_type,
                x=float(x),
                y=float(y),
                orientation=orientation,
                placement_status='PLACED'
            )

        self.data['components'] = components

    def _parse_pins(self, content: str):
        """Parse pin information"""
        pins = {}

        # Find PINS section
        pin_section = re.search(r'PINS\s+(\d+)\s*;(.*?)END\s+PINS', content, re.DOTALL)
        if not pin_section:
            return

        pin_content = pin_section.group(2)

        # Parse each pin - fix regular expression to match actual format
        pin_matches = re.findall(r'-\s*([\w\[\]_]+)\s*\+\s*NET\s+([\w\[\]_]+)\s*\+\s*DIRECTION\s+(\w+).*?\+\s*LAYER\s+(\w+).*?PLACED\s*\(\s*([\d\-]+)\s+([\d\-]+)\s*\)', pin_content, re.DOTALL)
        for match in pin_matches:
            pin_name, net_name, direction, layer, x, y = match
            pins[pin_name] = PinInfo(
                name=pin_name,
                net=net_name,
                direction=direction,
                layer=layer,
                x=float(x),
                y=float(y)
            )

        self.data['pins'] = pins
    
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

    def _parse_nets_section(self, nets_text: str, net_type: int = 0):
        """Unified method for parsing net sections

        Args:
            nets_text: Text content of net section
            net_type: Net type, 0=normal signal net, 1=special net (power/ground)
        """
        if not hasattr(self, '_nets_initialized'):
            self.data['nets'] = {}
            self.data['internal_pins'] = {}
            self._nets_initialized = True

        nets = self.data['nets']
        internal_pins = self.data['internal_pins']

        if net_type == 0:
            # Normal net format: - net_name ( connections ) [optional + USE SIGNAL ;]
            # Reference B graph generator, use more relaxed matching, support dot characters, don't require USE SIGNAL
            net_pattern = r'-\s+([\w\[\]_$\\.]+)\s+((?:\([^)]+\)\s*)+)'
        else:
            # Special net format: - net_name ( * pin_name ) + USE POWER/GROUND
            net_pattern = r'-\s+([\w\[\]_$\\.]+)\s+\(\s*\*\s+([\w\[\]_$\\.]+)\s*\)\s*\+\s*USE\s+(POWER|GROUND)'

        # Use finditer instead of findall, consistent with B graph generator
        net_matches = re.finditer(net_pattern, nets_text, re.DOTALL)

        for match in net_matches:
            if net_type == 0:
                # Normal net processing
                net_name = match.group(1)
                connections_text = match.group(2)
                connections = []

                # Parse connection information - support two formats: ( component pin ) and ( PIN_name )
                conn_pattern = r'\(\s*([^\s)]+)(?:\s+([^\s)]+))?\s*\)'
                conn_matches = re.findall(conn_pattern, connections_text)
                for match in conn_matches:
                    comp_name, pin_name = match
                    # Clean escape characters
                    comp_name = comp_name.replace('\\', '')

                    if pin_name:  # Two-part format: ( component pin )
                        pin_name = pin_name.replace('\\', '')
                        connections.append(f"{comp_name}.{pin_name}")

                        # If it's a component pin (non-PIN), add to internal pins
                        if comp_name != 'PIN' and comp_name in self.data['components']:
                            pin_id = f"{comp_name}.{pin_name}"
                            comp_info = self.data['components'][comp_name]
                            internal_pins[pin_id] = {
                                'component': comp_name,
                                'name': pin_name,
                                'net': net_name,
                                'x': comp_info.x,  # Use component coordinates
                                'y': comp_info.y
                            }
                    else:  # One-part format: ( PIN_name ) - external pin
                        connections.append(comp_name)  # Use pin name directly

                nets[net_name] = NetInfo(
                    name=net_name,
                    connections=connections
                )
            else:
                # Special net processing (power/ground nets)
                net_name = match.group(1)
                pin_name = match.group(2)
                use_type = match.group(3)
                # Clean escape characters
                net_name = net_name.replace('\\', '')
                pin_name = pin_name.replace('\\', '')

                # Special nets usually connect to corresponding pins of all components
                connections = []
                # Create virtual connections for special nets (connect to all components)
                for comp_name in self.data['components']:
                    connections.append(f"{comp_name}.{pin_name}")

                    # Add to internal pins
                    pin_id = f"{comp_name}.{pin_name}"
                    comp_info = self.data['components'][comp_name]
                    internal_pins[pin_id] = {
                        'component': comp_name,
                        'name': pin_name,
                        'net': net_name,
                        'x': comp_info.x,
                        'y': comp_info.y
                    }

                nets[net_name] = NetInfo(
                    name=net_name,
                    connections=connections
                )

        self.data['nets'] = nets
        self.data['internal_pins'] = internal_pins

# ============================================================================
# ENCODING UTILITIES SECTION - Encoding utilities area
# ============================================================================

class EncodingUtils:
    """Encoding utility class

    Provides encoding methods for various properties to numerical values for feature vector generation.
    References the complete mapping table and encoding scheme from B-graph generator.
    """

    # Orientation mapping - consistent with B-graph
    ORIENTATION_MAPPING = {
        'N': 0, 'S': 1, 'E': 2, 'W': 3,
        'FN': 4, 'FS': 5, 'FE': 6, 'FW': 7
    }

    # Pin direction mapping - consistent with B-graph
    PIN_DIRECTION_MAPPING = {
        'INPUT': 0, 'OUTPUT': 1, 'INOUT': 2, 'FEEDTHRU': 3
    }
    DIRECTION_MAPPING = PIN_DIRECTION_MAPPING  # Naming consistent with B-graph

    # Pin type mapping - use B-graph's complete continuous encoding
    PIN_TYPE_MAPPING = {
        # Basic input pin types (0-4) - continuous encoding
        'A': 0, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0,  # Data input A
        'B': 1, 'B1': 1, 'B2': 1, 'B3': 1, 'B4': 1,  # Data input B
        'C': 2, 'C1': 2, 'C2': 2, 'C3': 2, 'C4': 2,  # Data input C
        'D': 3, 'D1': 3, 'D2': 3, 'D3': 3, 'D4': 3,  # Data input D
        'S': 4, 'S0': 4, 'S1': 4,  # Selection signal

        # Control signal pin types (5-7) - continuous encoding
        'CI': 5, 'CIN': 5,  # Carry input
        'CLK': 6, 'CK': 6,  # Clock input
        'EN': 7, 'G': 7,    # Enable signal

        # Output pin types (8-11)
        'Y': 8, 'CO': 8, 'COUT': 8,  # Combinational logic output
        'Q': 9, 'QN': 9,             # Sequential logic output
        'Z': 10, 'ZN': 10,           # Main output
        'OUT': 11,                   # Other output

        # Power pin types (12-13)
        'VDD': 12, 'VCC': 12, 'VPWR': 12,  # Power
        'VSS': 13, 'GND': 13, 'VGND': 13,  # Ground

        # Unknown pin type
        'UNKNOWN': 14
    }

    # Layer mapping - consistent with B-graph
    LAYER_MAPPING = {
        'metal1': 0, 'metal2': 1, 'metal3': 2, 'metal4': 3, 'metal5': 4, 'metal6': 5,
        'metal7': 6, 'metal8': 7, 'metal9': 8, 'metal10': 9,
        'via1': 10, 'via2': 11, 'via3': 12, 'via4': 13, 'via5': 14,
        'via6': 15, 'via7': 16, 'via8': 17, 'via9': 18,
        'poly': 19, 'contact': 20, 'unknown': 21
    }

    # Net type mapping - consistent with B-graph
    NET_TYPE_MAPPING = {
        'signal': 0, 'power': 1, 'ground': 2, 'clock': 3, 'reset': 4, 'scan': 5
    }

    @classmethod
    def encode_cell_type(cls, cell_type: str) -> int:
        """Encode cell type - use complete mapping table"""
        # First try direct mapping
        if cell_type in COMPLETE_CELL_TYPE_MAPPING:
            return COMPLETE_CELL_TYPE_MAPPING[cell_type]

        # For sky130 library cells, perform standardized mapping
        sky130_mapping = {
            'sky130_fd_sc_hd__nand2_1': 'NAND2_X1',
            'sky130_fd_sc_hd__nor2_1': 'NOR2_X1',
            'sky130_fd_sc_hd__and2_1': 'AND2_X1',
            'sky130_fd_sc_hd__or2_1': 'OR2_X1',
            'sky130_fd_sc_hd__inv_1': 'INV_X1',
            'sky130_fd_sc_hd__buf_1': 'BUF_X1',
            'sky130_fd_sc_hd__mux2_1': 'MUX2_X1',
            'sky130_fd_sc_hd__dfxtp_1': 'DFF_X1',
            'sky130_fd_sc_hd__dlxtp_1': 'DFF_X1',
            'sky130_fd_sc_hd__fill_1': 'FILLCELL_X1',
            'sky130_fd_sc_hd__decap_3': 'FILLCELL_X1',
            'sky130_fd_sc_hd__tapvpwrvgnd_1': 'FILLCELL_X1'
        }

        if cell_type in sky130_mapping:
            standard_type = sky130_mapping[cell_type]
            return COMPLETE_CELL_TYPE_MAPPING[standard_type]

        return COMPLETE_CELL_TYPE_MAPPING['UNKNOWN']  # Unknown type returns 95

    @classmethod
    def encode_orientation(cls, orientation: str) -> int:
        """Encode orientation"""
        return cls.ORIENTATION_MAPPING.get(orientation, 0)

    @classmethod
    def encode_direction(cls, direction: str) -> int:
        """Encode pin direction"""
        return cls.DIRECTION_MAPPING.get(direction, 2)

    @classmethod
    def encode_layer(cls, layer: str) -> int:
        """Encode layer name"""
        return cls.LAYER_MAPPING.get(layer, cls.LAYER_MAPPING['unknown'])

    @classmethod
    def encode_pin_type(cls, pin_name: str) -> int:
        """Encode pin type"""
        return cls.PIN_TYPE_MAPPING.get(pin_name.upper(), cls.PIN_TYPE_MAPPING['UNKNOWN'])

    @classmethod
    def encode_net_type(cls, net_name: str) -> int:
        """Infer net type based on net name and encode"""
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
        """Encode placement status"""
        placement_mapping = {
            'PLACED': 0,   # Placed (movable)
            'FIXED': 1,    # Fixed position (non-movable)
            'COVER': 2,    # Cover placed
            'UNPLACED': 3  # Unplaced
        }
        return placement_mapping.get(placement_status, 0)

# ============================================================================
# AREA AND POWER DEFINITIONS SECTION - Area and power definition area
# ============================================================================

# Complete cell type mapping - consistent with R-graph generator
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
        """Get area in DBU units based on cell type - unify coordinate and area units"""
        # DBU conversion factor: dynamically obtained from DEF file UNITS DISTANCE MICRONS
        if def_data and 'units' in def_data:
            units_info = def_data['units']
            if isinstance(units_info, dict):
                DBU_PER_MICRON = float(units_info.get('dbu_per_micron', 2000))
            else:
                DBU_PER_MICRON = float(units_info)
        else:
            DBU_PER_MICRON = 2000  # Default value
        
        # 1 um = DBU_PER_MICRON DBU, so 1 um² = (DBU_PER_MICRON)² DBU²
        AREA_CONVERSION_FACTOR = DBU_PER_MICRON * DBU_PER_MICRON
        
        # Real process library area data (based on NangateOpenCellLibrary_typical.lib)
        actual_cell_areas = {
            # Basic logic gate series - actual area (um²)
            'INV_X1': 0.532, 'INV_X2': 0.798, 'INV_X4': 1.197, 'INV_X8': 1.995, 'INV_X16': 3.990, 'INV_X32': 7.980,
            'BUF_X1': 0.798, 'BUF_X2': 1.064, 'BUF_X4': 1.596, 'BUF_X8': 2.660, 'BUF_X16': 4.788, 'BUF_X32': 9.044,
            'NAND2_X1': 0.798, 'NAND2_X2': 1.064, 'NAND2_X4': 1.596,
            'NAND3_X1': 1.064, 'NAND3_X2': 1.330, 'NAND3_X4': 1.995,
            'NAND4_X1': 1.330, 'NAND4_X2': 1.729, 'NAND4_X4': 2.594,
            'NOR2_X1': 0.798, 'NOR2_X2': 1.064, 'NOR2_X4': 1.596,
            'NOR3_X1': 1.064, 'NOR3_X2': 1.330, 'NOR3_X4': 1.995,
            'NOR4_X1': 1.330, 'NOR4_X2': 1.729, 'NOR4_X4': 2.594,
            'AND2_X1': 1.064, 'AND2_X2': 1.330, 'AND2_X4': 1.995,
            'AND3_X1': 1.330, 'AND3_X2': 1.729, 'AND3_X4': 2.594,
            'AND4_X1': 1.596, 'AND4_X2': 1.995, 'AND4_X4': 2.992,
            'OR2_X1': 1.064, 'OR2_X2': 1.330, 'OR2_X4': 1.995,
            'OR3_X1': 1.330, 'OR3_X2': 1.729, 'OR3_X4': 2.594,
            'OR4_X1': 1.596, 'OR4_X2': 1.995, 'OR4_X4': 2.992,
            'XOR2_X1': 1.596, 'XOR2_X2': 2.128,
            'XNOR2_X1': 1.596, 'XNOR2_X2': 2.128,
            'AOI21_X1': 1.064, 'AOI21_X2': 1.330, 'AOI21_X4': 1.995,
            'AOI22_X1': 1.330, 'AOI22_X2': 1.729, 'AOI22_X4': 2.594,
            'OAI21_X1': 1.064, 'OAI21_X2': 1.330, 'OAI21_X4': 1.995,
            'OAI22_X1': 1.330, 'OAI22_X2': 1.729, 'OAI22_X4': 2.594,
            'MUX2_X1': 2.660, 'MUX2_X2': 3.458,
            'DFF_X1': 4.522, 'DFF_X2': 5.852,
            'DFFR_X1': 5.320, 'DFFR_X2': 6.650,
            'DFFS_X1': 5.320, 'DFFS_X2': 6.650,
            'FILLCELL_X1': 0.266, 'FILLCELL_X2': 0.532, 'FILLCELL_X4': 1.064, 'FILLCELL_X8': 2.128, 'FILLCELL_X16': 4.256, 'FILLCELL_X32': 8.512,
            'LOGIC0_X1': 0.532, 'LOGIC1_X1': 0.532,
            'ANTENNA_X1': 0.266,
        }

        normalized_cell_type = cell_type.upper()
        actual_area_um2 = actual_cell_areas.get(normalized_cell_type, 0.532)  # Default INV_X1 area

        # Convert to DBU² units, unify with coordinate units
        area_dbu2 = actual_area_um2 * AREA_CONVERSION_FACTOR
        return area_dbu2

    @staticmethod
    def calculate_cell_power(cell_type: str) -> float:
        """Get real power features based on cell type - based on NangateOpenCellLibrary_typical.lib"""
        # Real process library power data (extracted from NangateOpenCellLibrary_typical.lib)
        real_cell_powers = {
            # Inverter series
            'INV_X1': 14.353185, 'INV_X2': 28.706376, 'INV_X4': 57.412850, 'INV_X8': 114.826305, 'INV_X16': 229.651455, 'INV_X32': 459.302800,
            # Buffer series
            'BUF_X1': 21.438247, 'BUF_X2': 43.060820, 'BUF_X4': 86.121805, 'BUF_X8': 172.244545, 'BUF_X16': 344.488100, 'BUF_X32': 688.976200,
            # NAND gate series
            'NAND2_X1': 17.393360, 'NAND2_X2': 34.786630, 'NAND2_X4': 69.573240,
            'NAND3_X1': 18.104768, 'NAND3_X2': 36.209558, 'NAND3_X4': 72.419123,
            'NAND4_X1': 18.126843, 'NAND4_X2': 36.253723, 'NAND4_X4': 72.506878,
            # NOR gate series
            'NOR2_X1': 21.199545, 'NOR2_X2': 42.399074, 'NOR2_X4': 84.798143,
            'NOR3_X1': 26.831667, 'NOR3_X2': 53.663264, 'NOR3_X4': 107.325918,
            'NOR4_X1': 32.601474, 'NOR4_X2': 65.202889, 'NOR4_X4': 130.405147,
            # AND gate series
            'AND2_X1': 25.066064, 'AND2_X2': 50.353160, 'AND2_X4': 100.706457,
            'AND3_X1': 26.481460, 'AND3_X2': 53.190270, 'AND3_X4': 106.380663,
            'AND4_X1': 27.024804, 'AND4_X2': 54.274743, 'AND4_X4': 108.549590,
            # OR gate series
            'OR2_X1': 22.694975, 'OR2_X2': 45.656022, 'OR2_X4': 91.312375,
            'OR3_X1': 24.414625, 'OR3_X2': 49.162437, 'OR3_X4': 98.325150,
            'OR4_X1': 26.733490, 'OR4_X2': 53.869509, 'OR4_X4': 107.739253,
            # XOR gate series
            'XOR2_X1': 36.163718, 'XOR2_X2': 72.593483,
            'XNOR2_X1': 36.441009, 'XNOR2_X2': 73.102975,
            # AOI/OAI complex gate series
            'AOI21_X1': 27.858395, 'AOI21_X2': 55.716720, 'AOI21_X4': 111.433338,
            'AOI22_X1': 32.611944, 'AOI22_X2': 65.223838, 'AOI22_X4': 130.447551,
            'OAI21_X1': 22.619394, 'OAI21_X2': 45.238687, 'OAI21_X4': 90.477187,
            'OAI22_X1': 34.026125, 'OAI22_X2': 68.052131, 'OAI22_X4': 136.103946,
            # Multiplexer series
            'MUX2_X1': 61.229735, 'MUX2_X2': 68.648566,
            # Flip-flop series
            'DFF_X1': 100.684799, 'DFF_X2': 136.676074,
            'DFFR_X1': 105.258197, 'DFFR_X2': 141.961514,
            'DFFS_X1': 107.724855, 'DFFS_X2': 140.592991,
            # Fill cell series
            'FILLCELL_X1': 0.000000, 'FILLCELL_X2': 0.000000, 'FILLCELL_X4': 0.000000, 'FILLCELL_X8': 0.000000, 'FILLCELL_X16': 0.000000, 'FILLCELL_X32': 0.000000,
            # Logic constant cells
            'LOGIC0_X1': 35.928390, 'LOGIC1_X1': 17.885841,
            # Antenna protection cell
            'ANTENNA_X1': 0.000000,
        }

        normalized_cell_type = cell_type.upper()
        return real_cell_powers.get(normalized_cell_type, 14.353185)  # Default INV_X1 power

    @staticmethod
    def calculate_net_weight(net_info, components: Dict) -> float:
        """Calculate network weight - based on connected component power and fanout"""
        if not hasattr(net_info, 'connections') or not net_info.connections:
            return 1.0

        # Extract component names from connections
        connected_components = set()
        for connection in net_info.connections:
            if '.' in connection:
                comp_name = connection.split('.')[0]
                if comp_name != 'PIN':  # Exclude IO pins
                    connected_components.add(comp_name)

        if not connected_components:
            return 1.0

        # Calculate total power of connected components
        total_power = 0.0
        for comp_name in connected_components:
            if comp_name in components:
                comp_info = components[comp_name]
                comp_power = FeatureEngineering.calculate_cell_power(comp_info.cell_type)
                total_power += comp_power

        # Fanout weight
        fanout = len(connected_components)
        fanout_weight = min(fanout / 10.0, 1.0)  # Normalize to [0,1]

        # Comprehensive weight: power weight + fanout weight
        power_weight = min(total_power / 1000.0, 1.0)  # Normalize power weight
        net_weight = 0.7 * power_weight + 0.3 * fanout_weight

        return max(net_weight, 0.1)  # Minimum weight 0.1

# ============================================================================
# HETEROGRAPH BUILDER SECTION - Heterogeneous graph builder area
# ============================================================================

class CHeteroGraphBuilder:
    """C-graph heterogeneous graph builder

    Builds heterogeneous graph with Gate, Net, IO_Pin nodes and Gate-Net, IO_Pin-Net edges.
    Compared to B-graph, removes Pin nodes, Pin-Net and Gate-Pin edges, adds Gate-Net edges.
    """
    
    # ==================== C-graph customized type mapping definition ====================
    # Based on actual usage of C-script in node_edge_analysis_report.md

    @property
    def node_type_to_id(self) -> Dict[str, int]:
        """C-graph node type to ID mapping"""
        return {
            "gate": 0,
            "io_pin": 1,
            "net": 2
        }

    @property
    def edge_type_to_id(self) -> Dict[str, int]:
        """C-graph edge type to ID mapping"""
        return {
            "('gate', 'connects_to', 'net')": 0,
            "('io_pin', 'connects_to', 'net')": 1
        }

    # ========================================================

    def __init__(self, def_data: Dict[str, Any], def_file_path: str):
        """Initialize builder

        Args:
            def_data: Parsed DEF data
            def_file_path: DEF file path
        """
        self.def_data = def_data
        self.def_file_path = def_file_path
        self.hetero_data = HeteroData()

        # Node index mapping
        self.gate_name_to_idx = {}
        self.net_name_to_idx = {}
        self.pin_name_to_idx = {}

        # Get chip area information
        self.die_area = def_data.get('die_area', [0, 0, 1000, 1000])

    def build(self) -> HeteroData:
        """Build C-graph heterogeneous graph

        Returns:
            Built heterogeneous graph data
        """
        print("🔨 Start building C-graph heterogeneous graph...")

        # Add nodes - in standard order Gate -> IO_Pin -> Net
        self._add_gate_nodes()
        self._add_io_pin_nodes()
        self._add_net_nodes()

        # Add edges - in standard order
        self._add_gate_net_edges()    # Gate-Net edge (ID: 1)
        self._add_io_pin_net_edges()  # IO_Pin-Net edge (ID: 4)

        # Add global features
        self._add_global_features()

        print("✅ C-graph heterogeneous graph build completed")
        return self.hetero_data

    def _add_gate_nodes(self):
        """Add Gate nodes"""
        components = self.def_data.get('components', {})
        if not components:
            print("⚠️  Warning: No component information found")
            return

        # Filter out real gate components (exclude fill cells, etc.)
        gate_components = {}
        for comp_name, comp_info in components.items():
            cell_type = comp_info.cell_type
            if (not cell_type.startswith('FILLCELL') and 
                not cell_type.startswith('ANTENNA') and 
                not cell_type.startswith('TAPCELL')):
                gate_components[comp_name] = comp_info

        features = []
        names = []

        for comp_name, comp_info in gate_components.items():
            # Feature vector: [x, y, cell_type_id, orientation_id, area, placement_status, power]
            cell_type_id = EncodingUtils.encode_cell_type(comp_info.cell_type)
            orientation_id = EncodingUtils.encode_orientation(comp_info.orientation)

            # Position features - use original coordinates (consistent with B-graph)
            pos_x = comp_info.x
            pos_y = comp_info.y

            # Area features - use real process library data
            area = FeatureEngineering.calculate_component_size(comp_info.cell_type, self.def_data)

            # Power features - use real process library data
            power = FeatureEngineering.calculate_cell_power(comp_info.cell_type)

            # Placement status features - use encoding function (consistent with B-graph)
            placement_status = EncodingUtils.encode_placement_status(comp_info.placement_status)
            
            features.append([pos_x, pos_y, cell_type_id, orientation_id, area, placement_status, power])
            names.append(comp_name)
        
        self.hetero_data['gate'].x = torch.tensor(features, dtype=torch.float)
        self.hetero_data['gate'].names = names
        self.gate_name_to_idx = {name: i for i, name in enumerate(names)}

        print(f"  ✓ Gate nodes: {len(names)}")

    def _calculate_net_hpwl(self, net_name: str, net_info) -> float:
        """Calculate network HPWL (Half-Perimeter Wire Length)

        Args:
            net_name: Network name
            net_info: Network information object

        Returns:
            float: HPWL value, calculation formula is (x_max - x_min) + (y_max - y_min)
        """
        connections = getattr(net_info, 'connections', [])
        if not connections:
            return 0.0

        # Collect coordinates of all components and pins connected to this network
        x_coords = []
        y_coords = []

        components = self.def_data.get('components', {})
        pins = self.def_data.get('pins', {})

        for connection in connections:
            # Parse connection format "component.pin" or "PIN.pin_name"
            if '.' in connection:
                comp_name, pin_name = connection.split('.', 1)

                # Handle IO pin connections
                if comp_name == 'PIN' and pin_name in pins:
                    pin_info = pins[pin_name]
                    x_coords.append(pin_info.x)
                    y_coords.append(pin_info.y)
                # Handle component pin connections
                elif comp_name in components:
                    comp_info = components[comp_name]
                    x_coords.append(comp_info.x)
                    y_coords.append(comp_info.y)

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
            print("⚠️  Warning: No network information found")
            return

        features = []
        names = []
        hpwl_labels = []  # Added: HPWL label list

        for net_name, net_info in nets.items():
            # Feature vector: [net_type_id, connection_count]
            net_type_id = EncodingUtils.encode_net_type(net_name)
            connection_count = len(net_info.connections)

            # Calculate HPWL as label
            hpwl = self._calculate_net_hpwl(net_name, net_info)
            
            features.append([net_type_id, connection_count])
            names.append(net_name)
            hpwl_labels.append(hpwl)

        self.hetero_data['net'].x = torch.tensor(features, dtype=torch.float)
        self.hetero_data['net'].y = torch.tensor(hpwl_labels, dtype=torch.float)  # Added: HPWL labels
        self.hetero_data['net'].names = names
        self.net_name_to_idx = {name: i for i, name in enumerate(names)}

        # HPWL statistics
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
    
    def _add_io_pin_nodes(self):
        """Add IO_Pin nodes"""
        pins = self.def_data.get('pins', {})
        if not pins:
            print("⚠️  Warning: No IO pin information found")
            return

        features = []
        names = []

        for pin_name, pin_info in pins.items():
            # Feature vector: [x, y, direction_id, layer_id]
            direction_id = EncodingUtils.encode_direction(pin_info.direction)
            layer_id = EncodingUtils.encode_layer(pin_info.layer)

            # Position features - use original coordinates (consistent with B-graph)
            pos_x = pin_info.x
            pos_y = pin_info.y
            
            features.append([pos_x, pos_y, direction_id, layer_id])
            names.append(pin_name)

        self.hetero_data['io_pin'].x = torch.tensor(features, dtype=torch.float)
        self.hetero_data['io_pin'].names = names
        self.pin_name_to_idx = {name: i for i, name in enumerate(names)}

        print(f"  ✓ IO_Pin nodes: {len(names)}")

    def _add_gate_net_edges(self):
        """Add Gate to Net edges"""
        if not self.gate_name_to_idx or not self.net_name_to_idx:
            print("⚠️  Warning: Gate or Net nodes are empty, skipping Gate-Net edges")
            return

        edge_indices = []
        edge_features = []

        internal_pins = self.def_data.get('internal_pins', {})
        for pin_id, pin_info in internal_pins.items():
            comp_name = pin_info.get('component', '')
            net_name = pin_info.get('net', '')

            if comp_name in self.gate_name_to_idx and net_name in self.net_name_to_idx:
                gate_idx = self.gate_name_to_idx[comp_name]
                net_idx = self.net_name_to_idx[net_name]
                edge_indices.append([gate_idx, net_idx])

                # Edge features: [pin_type_id, cell_type_id] - inherit B-graph pin node features
                pin_type_id = EncodingUtils.encode_pin_type(pin_info.get('name', ''))
                comp_info = self.def_data['components'].get(comp_name, {})
                cell_type_id = EncodingUtils.encode_cell_type(comp_info.cell_type if hasattr(comp_info, 'cell_type') else '')
                
                edge_features.append([pin_type_id, cell_type_id])

        if edge_indices:
            self.hetero_data['gate', 'connects_to', 'net'].edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            self.hetero_data['gate', 'connects_to', 'net'].edge_attr = torch.tensor(edge_features, dtype=torch.float)

        print(f"  ✓ Gate-Net edges: {len(edge_indices)}")
    
    def _add_io_pin_net_edges(self):
        """Add IO_Pin to Net edges"""
        if not self.pin_name_to_idx or not self.net_name_to_idx:
            print("⚠️  Warning: IO_Pin or Net nodes are empty, skipping IO_Pin-Net edges")
            return

        edge_indices = []
        edge_features = []

        pins = self.def_data.get('pins', {})
        for pin_name, pin_info in pins.items():
            if pin_name not in self.pin_name_to_idx:
                continue

            pin_idx = self.pin_name_to_idx[pin_name]
            net_name = pin_info.net

            if net_name in self.net_name_to_idx:
                net_idx = self.net_name_to_idx[net_name]
                edge_indices.append([pin_idx, net_idx])

                # Edge features: [pin_direction_id, net_type_id]
                pin_direction_id = EncodingUtils.encode_direction(pin_info.direction)
                net_type_id = EncodingUtils.encode_net_type(net_name)
                
                edge_features.append([pin_direction_id, net_type_id])

        if edge_indices:
            self.hetero_data['io_pin', 'connects_to', 'net'].edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            self.hetero_data['io_pin', 'connects_to', 'net'].edge_attr = torch.tensor(edge_features, dtype=torch.float)

        print(f"  ✓ IO_Pin-Net edges: {len(edge_indices)}")

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

        # Extended global features: [chip width, chip height, chip area, DBU unit, config utilization]
        global_features = torch.tensor([chip_width, chip_height, chip_area, dbu_per_micron, core_utilization_config], dtype=torch.float)
        self.hetero_data.global_features = global_features

        # Chip coordinate information - 2x2 matrix: [[bottom-left x, bottom-left y], [top-right x, top-right y]]
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

        print(f"  ✓ Global features: 5D (width={chip_width:.0f}, height={chip_height:.0f}, area={chip_area:.0f}, DBU={dbu_per_micron:.0f}, config_util={core_utilization_config:.1f}%)")
        print(f"  ✓ Global labels: 3D (actual_util={utilization:.2f}%, HPWL={hpwl:.1f}um, Pin_density={pin_density:.2f}%)")

    def _read_config_utilization(self):
        """Read CORE_UTILIZATION configuration for corresponding design from place_data_extract.csv file"""
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
                    # 4. Handle bracket case, extract name inside brackets for matching
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
        """Read design_utilization and hpwl_after labels for corresponding design from place_data_extract.csv file"""
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
                    # 2. DEF design name in CSV name (e.g., ac97_top not in ac97_ctrl, but aes in systemcaes(aes))
                    elif design_name in csv_design_name:
                        match_found = True
                    # 3. CSV name in DEF design name (e.g., ac97_ctrl not in ac97_top)
                    elif csv_design_name in design_name:
                        match_found = True
                    # 4. Handle bracket case, extract name inside brackets for matching (e.g., vga_lcd(vga_enh_top) -> vga_enh_top)
                    elif '(' in csv_design_name and ')' in csv_design_name:
                        bracket_name = csv_design_name.split('(')[1].split(')')[0]
                        if design_name == bracket_name:
                            match_found = True
                        else:
                            match_found = False
                    # 5. Reverse bracket matching, check if DEF name matches part before brackets in CSV
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

            print(f"⚠️  Warning: Placement label data for design {design_name} not found in CSV file")
            return 0.0, 0.0  # Default value
        except Exception as e:
            print(f"⚠️  Warning: Unable to read place_data_extract.csv file: {e}")
            return 0.0, 0.0  # Default value

# ============================================================================
# GRAPH VALIDATOR SECTION - Graph validator area
# ============================================================================

class GraphValidator:
    """Graph structure validator

    Validates whether the generated heterogeneous graph meets expected structure and constraints.
    """
    
    @staticmethod
    def validate_heterograph(hetero_data: HeteroData) -> bool:
        """Validate heterogeneous graph structure

        Args:
            hetero_data: Heterogeneous graph data to validate

        Returns:
            Whether validation passed
        """
        print("🔍 Start validating C-graph heterogeneous graph structure...")

        try:
            # Check node types
            expected_node_types = {'gate', 'net', 'io_pin'}
            actual_node_types = set(hetero_data.node_types)

            if not expected_node_types.issubset(actual_node_types):
                missing = expected_node_types - actual_node_types
                print(f"❌ Missing node types: {missing}")
                return False

            # Check edge types
            expected_edge_types = {
                ('gate', 'connects_to', 'net'),
                ('io_pin', 'connects_to', 'net')
            }
            actual_edge_types = set(hetero_data.edge_types)

            if not expected_edge_types.issubset(actual_edge_types):
                missing = expected_edge_types - actual_edge_types
                print(f"❌ Missing edge types: {missing}")
                return False

            # Check node feature dimensions
            if 'gate' in hetero_data.node_types:
                gate_features = hetero_data['gate'].x
                if gate_features.shape[1] != 7:
                    print(f"❌ Gate node feature dimension error: expected 7D, actual {gate_features.shape[1]}D")
                    return False

            if 'net' in hetero_data.node_types:
                net_features = hetero_data['net'].x
                if net_features.shape[1] != 2:
                    print(f"❌ Net node feature dimension error: expected 2D, actual {net_features.shape[1]}D")
                    return False

            if 'io_pin' in hetero_data.node_types:
                pin_features = hetero_data['io_pin'].x
                if pin_features.shape[1] != 4:
                    print(f"❌ IO_Pin node feature dimension error: expected 4D, actual {pin_features.shape[1]}D")
                    return False

            # Check edge feature dimensions
            if ('gate', 'connects_to', 'net') in hetero_data.edge_types:
                edge_attr = hetero_data['gate', 'connects_to', 'net'].edge_attr
                if edge_attr.shape[1] != 2:
                    print(f"❌ Gate-Net edge feature dimension error: expected 2D, actual {edge_attr.shape[1]}D")
                    return False

            if ('io_pin', 'connects_to', 'net') in hetero_data.edge_types:
                edge_attr = hetero_data['io_pin', 'connects_to', 'net'].edge_attr
                if edge_attr.shape[1] != 2:
                    print(f"❌ IO_Pin-Net edge feature dimension error: expected 2D, actual {edge_attr.shape[1]}D")
                    return False

            # Check global features
            if hasattr(hetero_data, 'y'):
                global_labels = hetero_data.y
                if global_labels.shape[0] != 3:
                    print(f"❌ Global label dimension error: expected 3D, actual {global_labels.shape[0]}D")
                    return False

            print("✅ C-graph heterogeneous graph structure validation passed")
            return True

        except Exception as e:
            print(f"❌ Error during validation: {e}")
            return False

# ============================================================================
# MAIN CONVERSION FUNCTION SECTION - Main conversion function area
# ============================================================================

def convert_def_to_c_heterograph(def_file_path: str, output_path: Optional[str] = None) -> HeteroData:
    """Convert DEF file to C-graph heterogeneous graph

    Args:
        def_file_path: Input DEF file path
        output_path: Output .pt file path, automatically generated if None

    Returns:
        Converted heterogeneous graph data

    Raises:
        RuntimeError: When error occurs during conversion
    """
    print("🚀 Start DEF to C-graph heterogeneous graph conversion...")
    print(f"📁 Input file: {def_file_path}")
    print()

    # 1. Parse DEF file
    parser = DEFParser(def_file_path)
    def_data = parser.parse()

    print(f"📊 Parsing results:")
    print(f"  - Design name: {def_data.get('design_name', 'unknown')}")
    print(f"  - Component count: {len(def_data.get('components', {}))}")
    print(f"  - Pin count: {len(def_data.get('pins', {}))}")
    print(f"  - Net count: {len(def_data.get('nets', {}))}")
    print(f"  - Layer count: {len(def_data.get('tracks', {}))}")
    print()

    # 2. Build C-graph heterogeneous graph
    builder = CHeteroGraphBuilder(def_data, def_file_path)
    c_graph = builder.build()

    # 3. Validate graph structure
    if not GraphValidator.validate_heterograph(c_graph):
        raise RuntimeError("Graph structure validation failed")

    # 4. Save results
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(def_file_path))[0]
        output_path = f"{base_name}_c_heterograph.pt"

    print(f"💾 Save C-graph heterogeneous graph to: {output_path}")
    torch.save(c_graph, output_path, pickle_protocol=4)

    print("\n🎉 Conversion completed!")
    print(f"📈 Final graph structure statistics:")
    print(f"  - Node types: {len(c_graph.node_types)}")
    print(f"  - Edge types: {len(c_graph.edge_types)}")

    total_nodes = sum(c_graph[nt].x.shape[0] for nt in c_graph.node_types)
    total_edges = sum(c_graph[et].edge_index.shape[1] for et in c_graph.edge_types)
    print(f"  - Total nodes: {total_nodes}")
    print(f"  - Total edges: {total_edges}")

    # Display global feature information
    if hasattr(c_graph, 'global_features'):
        global_feat = c_graph.global_features
        print(f"  - Global features: {global_feat.shape[0]}D")
        print(f"    * Chip width: {global_feat[0]:.0f}")
        print(f"    * Chip height: {global_feat[1]:.0f}")
        print(f"    * Chip area: {global_feat[2]:.0f}")
        print(f"    * DBU unit: {global_feat[3]:.0f}")

    return c_graph

# ============================================================================
# COMMAND LINE INTERFACE SECTION - Command line interface area
# ============================================================================

def main():
    """Main function

    Main entry point for command line interface, handles command line arguments and calls conversion function.
    Supported command line arguments:
    - def_file: Input DEF file path (positional argument)
    - -o/--output: Output .pt file path (optional)

    Returns:
        int: Program exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description='C-graph heterogeneous graph generator - DEF to C-graph heterogeneous graph conversion tool')
    parser.add_argument('def_file', nargs='?', default='/Users/david/Desktop/design_1.0/def/3_5_place_dp.def',
                       help='Input DEF file path')
    parser.add_argument('-o', '--output', default=None,
                       help='Output .pt file path')

    args = parser.parse_args()

    if not os.path.exists(args.def_file):
        print(f"❌ Error: DEF file does not exist: {args.def_file}")
        return 1

    try:
        c_graph = convert_def_to_c_heterograph(
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