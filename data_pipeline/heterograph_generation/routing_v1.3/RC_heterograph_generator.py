#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C[CN][CN][CN][CN]generate[CN] - DEF[CN]C[CN][CN][CN][CN] convert[CN][CN]

Author: EDA for AI Team

[CN][CN][CN][CN]:
-  based onB[CN]generate[CN],[CN][CN]Pinnode
- [CN][CN]Pin-Net[CN]Gate-Pin[CN][CN]edge
-  addGate-Netedge,feature[CN][pin_type_id, cell_type_id]([CN][CN]B[CN]pinnodefeature)
- [CN][CN]Gate,Net,IO_Pin[CN][CN]nodetype
- [CN][CN]IO_Pin-Netedge

[CN][CN][CN]nodetype:
- Gate: [CN][CN] gatenode, contains position,type,[CN][CN][CN]information
- Net: netnode,[CN][CN] connection[CN][CN]
- IO_Pin:  input outputpinnode

[CN][CN][CN]edgetype:
- Gate-Net:  gate[CN]net[CN][CN][CN] connection
- IO_Pin-Net: IOpin[CN]net[CN] connection
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
# DATA STRUCTURES SECTION -  data structure[CN][CN]region
# ============================================================================

@dataclass
class ComponentInfo:
    """ componentinformation data[CN]"""
    name: str
    cell_type: str
    x: float
    y: float
    orientation: str
    placement_status: str

@dataclass
class NetInfo:
    """netinformation data[CN]"""
    name: str
    connections: List[str]
    wire_length: float = 0.0  #  actual[CN][CN](DBU[CN][CN])
    via_count: int = 0        # [CN][CN][CN][CN]
    
@dataclass
class PinInfo:
    """pininformation data[CN]"""
    name: str
    net: str
    direction: str
    layer: str
    x: float
    y: float

@dataclass
class InternalPinInfo:
    """[CN][CN]pininformation data[CN]"""
    pin_id: str
    component: str
    name: str
    net: str
    x: float
    y: float

# ============================================================================
# DEF PARSER SECTION - DEF[CN][CN] parseregion
# ============================================================================

class DEFParser:
    """DEF[CN][CN] parse[CN]
    
    [CN][CN] parseDEF[CN][CN][CN] extract[CN][CN][CN][CN][CN]information,including components,net,pin[CN]。
    """
    
    def __init__(self, def_file_path: str):
        """InitializeDEF parse[CN]
        
        Args:
            def_file_path: DEF[CN][CN][CN][CN]
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
            'internal_pins': {}  # [CN][CN]pininformation
        }
    
    def parse(self) -> Dict[str, Any]:
        """ parseDEF[CN][CN]
        
        Returns:
             contains[CN][CN] parse data[CN] dictionary
        """
        print(f"📖 [CN][CN] parseDEF[CN][CN]: {self.def_file_path}")
        
        with open(self.def_file_path, 'r') as f:
            content = f.read()
        
        #  parse[CN]part
        self._parse_design_name(content)
        self._parse_units(content)
        self._parse_die_area(content)
        self._parse_tracks(content)
        self._parse_components(content)
        self._parse_pins(content)
        self._parse_nets(content)
        
        print(f"[OK] DEF[CN][CN] parsecompleted")
        return self.data
    
    def _parse_design_name(self, content: str):
        """ parse[CN][CN][CN][CN]"""
        match = re.search(r'DESIGN\s+(\w+)', content)
        if match:
            self.data['design_name'] = match.group(1)
    
    def _parse_units(self, content: str):
        """ parse[CN][CN]information"""
        match = re.search(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', content)
        if match:
            self.data['units'] = {'dbu_per_micron': int(match.group(1))}
        else:
            self.data['units'] = {'dbu_per_micron': 1000}  #  default[CN]
    
    def _parse_die_area(self, content: str):
        """ parse[CN][CN]region"""
        match = re.search(r'DIEAREA\s+\(\s*([\d\-]+)\s+([\d\-]+)\s*\)\s+\(\s*([\d\-]+)\s+([\d\-]+)\s*\)', content)
        if match:
            self.data['die_area'] = [int(match.group(1)), int(match.group(2)), 
                                   int(match.group(3)), int(match.group(4))]
    
    def _parse_tracks(self, content: str):
        """ parse[CN][CN]information"""
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
        """ parse componentinformation"""
        components = {}
        
        # [CN][CN]COMPONENTSpart
        comp_section = re.search(r'COMPONENTS\s+(\d+)\s*;(.*?)END\s+COMPONENTS', content, re.DOTALL)
        if not comp_section:
            return
        
        comp_content = comp_section.group(2)
        
        #  parse[CN] component
        # [CN][CN][CN][CN][CN][CN][CN]SOURCE TIMING[CN],ensure[CN][CN][CN] component[CN][CN][CN] parse
        comp_matches = re.findall(r'-\s*([^\s]+)\s+(\w+)\s*.*?\+\s*PLACED\s*\(\s*([\d\-]+)\s+([\d\-]+)\s*\)\s*(\w+)', comp_content)
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
        """ parsepininformation"""
        pins = {}
        
        # [CN][CN]PINSpart
        pin_section = re.search(r'PINS\s+(\d+)\s*;(.*?)END\s+PINS', content, re.DOTALL)
        if not pin_section:
            return
        
        pin_content = pin_section.group(2)
        
        #  parse[CN]pin - [CN][CN][CN][CN][CN][CN][CN][CN][CN][CN] actualformat
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
        """ parsenetinformation"""
        nets = {}
        internal_pins = {}
        
        # [CN][CN]NETSpart
        net_section = re.search(r'NETS\s+(\d+)\s*;(.*?)END\s+NETS', content, re.DOTALL)
        if not net_section:
            return
        
        net_content = net_section.group(2)
        
        #  parse[CN]net - [CN][CN][CN]net parse,[CN][CN][CN][CN][CN]net[CN][CN][CN] connectionformat, contains[CN][CN][CN][CN]
        # [CN][CN]format: - net_name ( comp1 pin1 ) ( comp2 pin2 ) ... + USE SIGNAL ;
        net_pattern = r'-\s+([\w\[\]_$\\.]+)\s+((?:\([^)]+\)\s*)+)(.*?);'
        
        for match in re.finditer(net_pattern, net_content, re.DOTALL):
            net_name = match.group(1)
            connections_text = match.group(2)
            routing_text = match.group(3)  # [CN][CN][CN][CN]informationpart
            connections = []
            
            #  parse connectioninformation - [CN][CN][CN] ( component pin ) format
            conn_pattern = r'\(\s*([^\s]+)\s+([^\s]+)\s*\)'
            conn_matches = re.findall(conn_pattern, connections_text)
            for comp_name, pin_name in conn_matches:
                # [CN][CN][CN][CN][CN][CN]
                comp_name = comp_name.replace('\\', '')
                pin_name = pin_name.replace('\\', '')
                connections.append(f"{comp_name}.{pin_name}")
                
                # [CN][CN][CN] component[CN]pin([CN]PIN), add[CN][CN][CN]pin
                if comp_name != 'PIN' and comp_name in self.data['components']:
                    pin_id = f"{comp_name}.{pin_name}"
                    comp_info = self.data['components'][comp_name]
                    internal_pins[pin_id] = {
                        'component': comp_name,
                        'name': pin_name,
                        'net': net_name,
                        'x': comp_info.x,  # [CN][CN] component[CN][CN]
                        'y': comp_info.y
                    }
            
            #  parse[CN][CN]information, calculate[CN][CN][CN][CN][CN][CN][CN]
            wire_length, via_count = self._parse_routing_info(routing_text)
            
            nets[net_name] = NetInfo(
                name=net_name,
                connections=connections,
                wire_length=wire_length,
                via_count=via_count
            )
        
        self.data['nets'] = nets
        self.data['internal_pins'] = internal_pins
    
    def _parse_routing_info(self, routing_text: str) -> Tuple[float, int]:
        """ parse[CN][CN]information, calculate[CN][CN][CN][CN][CN][CN][CN]"""
        wire_length = 0.0
        via_count = 0
        
        # [CN][CN]ROUTEDpart
        if '+ ROUTED' not in routing_text:
            return wire_length, via_count
        
        #  extract[CN][CN][CN][CN][CN]
        # [CN][CN]format: metal2 ( x1 y1 ) ( x2 y2 ) [CN] metal2 ( x1 y1 ) ( * y2 ) [CN]
        route_pattern = r'(metal\d+)\s+\(\s*([\d\-\*]+)\s+([\d\-\*]+)\s*\)\s+\(\s*([\d\-\*]+)\s+([\d\-\*]+)\s*\)'
        
        for match in re.finditer(route_pattern, routing_text):
            layer = match.group(1)
            x1_str, y1_str = match.group(2), match.group(3)
            x2_str, y2_str = match.group(4), match.group(5)
            
            #  process[CN][CN],*[CN][CN][CN][CN][CN][CN][CN][CN][CN]
            try:
                if x1_str != '*' and y1_str != '*' and x2_str != '*' and y2_str != '*':
                    x1, y1 = int(x1_str), int(y1_str)
                    x2, y2 = int(x2_str), int(y2_str)
                    #  calculate[CN][CN][CN][CN][CN]
                    wire_length += abs(x2 - x1) + abs(y2 - y1)
                elif x1_str != '*' and y1_str != '*':
                    #  processpart*[CN][CN][CN], simplified process
                    if x2_str != '*':
                        x1, x2 = int(x1_str), int(x2_str)
                        wire_length += abs(x2 - x1)
                    if y2_str != '*':
                        y1, y2 = int(y1_str), int(y2_str)
                        wire_length += abs(y2 - y1)
            except ValueError:
                #  skip[CN][CN] parse[CN][CN][CN]
                continue
        
        # statistics[CN][CN][CN][CN]
        via_pattern = r'(via\d+_\d+)'
        via_matches = re.findall(via_pattern, routing_text)
        via_count = len(via_matches)
        
        return wire_length, via_count

# ============================================================================
# ENCODING UTILITIES SECTION -  encoding[CN][CN]region
# ============================================================================

class EncodingUtils:
    """ encoding[CN][CN][CN]
    
    provide various property-to-value encoding methods, used forfeature[CN][CN]generate。
    referenceB[CN]generate[CN][CN][CN][CN] mapping[CN][CN] encoding[CN][CN]。
    """
    
    # [CN][CN] mapping - [CN]B[CN]keep consistent
    ORIENTATION_MAPPING = {
        'N': 0, 'S': 1, 'E': 2, 'W': 3,
        'FN': 4, 'FS': 5, 'FE': 6, 'FW': 7
    }
    
    # pin[CN][CN] mapping - [CN]B[CN]keep consistent
    PIN_DIRECTION_MAPPING = {
        'INPUT': 0, 'OUTPUT': 1, 'INOUT': 2, 'FEEDTHRU': 3
    }
    DIRECTION_MAPPING = PIN_DIRECTION_MAPPING  # [CN]B[CN]keep consistent[CN][CN][CN]
    
    # pintype mapping - [CN][CN]Bgraph's[CN][CN][CN][CN] encoding
    PIN_TYPE_MAPPING = {
        # [CN][CN] inputpintype (0-4) - [CN][CN] encoding
        'A': 0, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0,  #  data inputA
        'B': 1, 'B1': 1, 'B2': 1, 'B3': 1, 'B4': 1,  #  data inputB
        'C': 2, 'C1': 2, 'C2': 2, 'C3': 2, 'C4': 2,  #  data inputC
        'D': 3, 'D1': 3, 'D2': 3, 'D3': 3, 'D4': 3,  #  data inputD
        'S': 4, 'S0': 4, 'S1': 4,  # [CN][CN][CN][CN]
        
        # [CN][CN][CN][CN]pintype (5-7) - [CN][CN] encoding
        'CI': 5, 'CIN': 5,  # [CN][CN] input
        'CLK': 6, 'CK': 6,  # [CN][CN] input
        'EN': 7, 'G': 7,    # [CN][CN][CN][CN]
        
        #  outputpintype (8-11)
        'Y': 8, 'CO': 8, 'COUT': 8,  # [CN][CN][CN][CN] output
        'Q': 9, 'QN': 9,             # [CN][CN][CN][CN] output
        'Z': 10, 'ZN': 10,           # [CN] output
        'OUT': 11,                   # [CN][CN] output
        
        # [CN][CN]pintype (12-13)
        'VDD': 12, 'VCC': 12, 'VPWR': 12,  # [CN][CN]
        'VSS': 13, 'GND': 13, 'VGND': 13,  # [CN]
        
        # [CN][CN]pintype
        'UNKNOWN': 14
    }
    
    # [CN] mapping - [CN]B[CN]keep consistent
    LAYER_MAPPING = {
        'metal1': 0, 'metal2': 1, 'metal3': 2, 'metal4': 3, 'metal5': 4, 'metal6': 5,
        'metal7': 6, 'metal8': 7, 'metal9': 8, 'metal10': 9,
        'via1': 10, 'via2': 11, 'via3': 12, 'via4': 13, 'via5': 14,
        'via6': 15, 'via7': 16, 'via8': 17, 'via9': 18,
        'poly': 19, 'contact': 20, 'unknown': 21
    }
    
    # nettype mapping - [CN]B[CN]keep consistent
    NET_TYPE_MAPPING = {
        'signal': 0, 'power': 1, 'ground': 2, 'clock': 3, 'reset': 4, 'scan': 5
    }
    
    @classmethod
    def encode_cell_type(cls, cell_type: str) -> int:
        """encode cell type - [CN][CN][CN][CN] mapping[CN]"""
        # [CN][CN][CN][CN][CN][CN] mapping
        if cell_type in COMPLETE_CELL_TYPE_MAPPING:
            return COMPLETE_CELL_TYPE_MAPPING[cell_type]
        
        # [CN][CN]sky130[CN][CN] cell,[CN][CN][CN][CN][CN] mapping
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
        
        return COMPLETE_CELL_TYPE_MAPPING['UNKNOWN']  # [CN][CN]type return95
    
    @classmethod
    def encode_orientation(cls, orientation: str) -> int:
        """encoding direction"""
        return cls.ORIENTATION_MAPPING.get(orientation, 0)
    
    @classmethod
    def encode_direction(cls, direction: str) -> int:
        """ encodingpin[CN][CN]"""
        return cls.DIRECTION_MAPPING.get(direction, 2)
    
    @classmethod
    def encode_layer(cls, layer: str) -> int:
        """ encoding[CN][CN][CN]"""
        return cls.LAYER_MAPPING.get(layer, cls.LAYER_MAPPING['unknown'])
    
    @classmethod
    def encode_pin_type(cls, pin_name: str) -> int:
        """ encodingpintype"""
        return cls.PIN_TYPE_MAPPING.get(pin_name.upper(), cls.PIN_TYPE_MAPPING['UNKNOWN'])
    
    @classmethod
    def encode_net_type(cls, net_name: str) -> int:
        """[CN][CN]net[CN][CN][CN][CN]nettype[CN] encoding"""
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
        """ encoding[CN][CN][CN][CN]"""
        placement_mapping = {
            'PLACED': 0,   # [CN][CN][CN]([CN][CN][CN])
            'FIXED': 1,    # [CN][CN] position([CN][CN][CN][CN])
            'COVER': 2,    # [CN][CN][CN][CN]
            'UNPLACED': 3  # [CN][CN][CN]
        }
        return placement_mapping.get(placement_status, 0)

# ============================================================================
# AREA AND POWER DEFINITIONS SECTION -  area[CN] power[CN][CN]region
# ============================================================================

# [CN][CN] celltype mapping - [CN]B[CN]generate[CN]keep consistent
COMPLETE_CELL_TYPE_MAPPING = {
    # [CN][CN][CN][CN] gate[CN][CN]
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
# FEATURE ENGINEERING SECTION - feature[CN][CN][CN][CN][CN]region
# ============================================================================

class FeatureEngineering:
    """feature[CN][CN][CN][CN][CN] - [CN][CN][CN][CN][CN][CN][CN],[CN][CN] calculate[CN][CN][CN]"""
    
    @staticmethod
    def normalize_coordinates(position: Tuple[float, float], die_area: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """[CN][CN][CN][CN][CN][CN][0,1][CN][CN]"""
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
        """[CN][CN] celltype[CN][CN]DBU[CN][CN] area - [CN][CN][CN][CN][CN] area[CN][CN]"""
        # DBU convert[CN][CN]:[CN]DEF[CN][CN]UNITS DISTANCE MICRONS[CN][CN][CN][CN]
        if def_data and 'units' in def_data:
            units_info = def_data['units']
            if isinstance(units_info, dict):
                DBU_PER_MICRON = float(units_info.get('dbu_per_micron', 2000))
            else:
                DBU_PER_MICRON = float(units_info)
        else:
            DBU_PER_MICRON = 2000  #  default[CN]
        
        # 1 um = DBU_PER_MICRON DBU,[CN][CN] 1 um² = (DBU_PER_MICRON)² DBU²
        AREA_CONVERSION_FACTOR = DBU_PER_MICRON * DBU_PER_MICRON
        
        # [CN][CN][CN][CN][CN] area data( based onNangateOpenCellLibrary_typical.lib)- [CN]RB[CN]keep consistent
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
        
        normalized_cell_type = cell_type.upper()
        actual_area_um2 = actual_cell_areas.get(normalized_cell_type, 0.532)  #  defaultINV_X1 area
        
        #  convert[CN]DBU²[CN][CN],unified with coordinate units
        area_dbu2 = actual_area_um2 * AREA_CONVERSION_FACTOR
        return area_dbu2
    
    @staticmethod
    def calculate_cell_power(cell_type: str) -> float:
        """[CN][CN] celltype[CN][CN][CN][CN] powerfeature -  based onNangateOpenCellLibrary_typical.lib"""
        # [CN][CN][CN][CN][CN] power data([CN]NangateOpenCellLibrary_typical.lib extract)- [CN]RB[CN]keep consistent
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
        
        normalized_cell_type = cell_type.upper()
        return real_cell_powers.get(normalized_cell_type, 14.353185)  #  default[CN]INV_X1 power
    


# ============================================================================
# HETEROGRAPH BUILDER SECTION - [CN][CN][CN] build[CN]region
# ============================================================================

class CHeteroGraphBuilder:
    """C[CN][CN][CN][CN] build[CN]
    
     build containsGate,Net,IO_Pinnode[CN]Gate-Net,IO_Pin-Netedge[CN][CN][CN][CN]。
    compared toB[CN],[CN][CN][CN]Pinnode,Pin-Net[CN]Gate-Pinedge, add[CN]Gate-Netedge。
    """
    
    # ==================== RC[CN][CN][CN][CN]type mapping[CN][CN] ====================
    #  based onnode_edge_analysis_report.md[CN]RC[CN][CN][CN] actual[CN][CN][CN][CN]
    
    @property
    def node_type_to_id(self) -> Dict[str, int]:
        """RC[CN]nodetype[CN]ID[CN] mapping"""
        return {
            "gate": 0,
            "io_pin": 1, 
            "net": 2
        }
    
    @property  
    def edge_type_to_id(self) -> Dict[str, int]:
        """RC[CN]edgetype[CN]ID[CN] mapping"""
        return {
            "('gate', 'connects_to', 'net')": 0,
            "('io_pin', 'connects_to', 'net')": 1
        }
    
    # ========================================================
    
    def __init__(self, def_data: Dict[str, Any], def_file_path: str):
        """Initialize build[CN]
        
        Args:
            def_data:  parse[CN][CN]DEF data
            def_file_path: DEF[CN][CN][CN][CN]
        """
        self.def_data = def_data
        self.def_file_path = def_file_path
        self.hetero_data = HeteroData()
        
        # node[CN][CN] mapping
        self.gate_name_to_idx = {}
        self.net_name_to_idx = {}
        self.pin_name_to_idx = {}
        
        # [CN][CN][CN][CN]regioninformation
        self.die_area = def_data.get('die_area', [0, 0, 1000, 1000])
    
    def build(self) -> HeteroData:
        """ buildC[CN][CN][CN][CN]
        
        Returns:
             buildcompleted[CN][CN][CN][CN] data
        """
        print("🔨 [CN][CN] buildC[CN][CN][CN][CN]...")
        
        #  addnode - [CN][CN][CN][CN][CN][CN] Gate -> IO_Pin -> Net
        self._add_gate_nodes()    # Gatenode (ID: 0)
        self._add_io_pin_nodes()  # IO_Pinnode (ID: 1)
        self._add_net_nodes()     # Netnode (ID: 2)
        
        #  addedge - [CN][CN][CN][CN][CN][CN]
        self._add_gate_net_edges()    # Gate-Netedge (ID: 1)
        self._add_io_pin_net_edges()  # IO_Pin-Netedge (ID: 4)
        
        #  addglobalfeature
        self._add_global_features()
        
        print("[OK] C[CN][CN][CN][CN] buildcompleted")
        return self.hetero_data
    
    def _add_gate_nodes(self):
        """ addGatenode"""
        components = self.def_data.get('components', {})
        if not components:
            print("[!]️  warning: [CN][CN][CN][CN] componentinformation")
            return
        
        # [CN][CN][CN][CN][CN][CN]gate component([CN][CN][CN][CN] cell[CN])- [CN][CN][CN]RF[CN][CN][CN][CN][CN][CN][CN][CN]
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
            # feature[CN][CN]: [x, y, cell_type_id, orientation_id, area, placement_status, power]
            cell_type_id = EncodingUtils.encode_cell_type(comp_info.cell_type)
            orientation_id = EncodingUtils.encode_orientation(comp_info.orientation)
            
            #  positionfeature - [CN][CN][CN][CN][CN][CN]([CN]B[CN]keep consistent)
            pos_x = comp_info.x
            pos_y = comp_info.y
            
            #  areafeature - [CN][CN][CN][CN][CN][CN][CN] data
            area = FeatureEngineering.calculate_component_size(comp_info.cell_type, self.def_data)
            
            #  powerfeature - [CN][CN][CN][CN][CN][CN][CN] data
            power = FeatureEngineering.calculate_cell_power(comp_info.cell_type)
            
            # [CN][CN][CN][CN]feature - [CN][CN] encoding[CN][CN]([CN]B[CN]keep consistent)
            placement_status = EncodingUtils.encode_placement_status(comp_info.placement_status)
            
            features.append([pos_x, pos_y, cell_type_id, orientation_id, area, placement_status, power])
            names.append(comp_name)
        
        self.hetero_data['gate'].x = torch.tensor(features, dtype=torch.float)
        self.hetero_data['gate'].names = names
        self.gate_name_to_idx = {name: i for i, name in enumerate(names)}
        
        print(f"  ✓ Gatenode: {len(names)} ")
    
    def _calculate_hpwl(self, net_name: str, net_info) -> float:
        """ calculatenet[CN]HPWL (Half-Perimeter Wire Length)
        
        Args:
            net_name: net[CN][CN]
            net_info: netinformation[CN][CN]
            
        Returns:
            float: HPWL[CN], calculate[CN][CN][CN] (x_max - x_min) + (y_max - y_min)
        """
        connections = getattr(net_info, 'connections', [])
        if not connections:
            return 0.0
        
        # [CN][CN][CN][CN] connection[CN][CN]net[CN] component[CN]pin[CN][CN][CN]
        x_coords = []
        y_coords = []
        
        components = self.def_data.get('components', {})
        pins = self.def_data.get('pins', {})
        
        for connection in connections:
            #  parse connectionformat "component.pin" [CN] "PIN.pin_name"
            if '.' in connection:
                comp_name, pin_name = connection.split('.', 1)
                
                #  processIOpin connection
                if comp_name == 'PIN' and pin_name in pins:
                    pin_info = pins[pin_name]
                    x_coords.append(pin_info.x)
                    y_coords.append(pin_info.y)
                #  process componentpin connection
                elif comp_name in components:
                    comp_info = components[comp_name]
                    x_coords.append(comp_info.x)
                    y_coords.append(comp_info.y)
        
        # [CN][CN][CN][CN][CN][CN][CN][CN], return0
        if not x_coords or not y_coords:
            return 0.0
        
        #  calculateHPWL = (x_max - x_min) + (y_max - y_min)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        hpwl = (x_max - x_min) + (y_max - y_min)
        return float(hpwl)
    
    def _calculate_net_wire_length(self, net_name: str, net_info) -> float:
        """ calculatenet[CN] actual[CN][CN] (routing[CN][CN][CN][CN])"""
        try:
            # [CN][CN] connection[CN] component[CN][CN]
            x_coords = []
            y_coords = []
            components = self.def_data.get('components', {})
            
            for connection in net_info.connections:
                if '.' in connection:
                    comp_name = connection.split('.')[0]
                    if comp_name in components:
                        comp_info = components[comp_name]
                        x_coords.append(comp_info.x)
                        y_coords.append(comp_info.y)
            
            # [CN][CN][CN][CN][CN][CN][CN][CN], return0
            if len(x_coords) < 2:
                return 0.0
            
            #  simplified calculate: based on connection[CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN] actual[CN][CN]
            total_length = 0.0
            for i in range(len(x_coords) - 1):
                manhattan_dist = abs(x_coords[i+1] - x_coords[i]) + abs(y_coords[i+1] - y_coords[i])
                total_length += manhattan_dist
            
            # [CN][CN]routing[CN][CN][CN][CN][CN] ([CN][CN][CN][CN][CN][CN][CN][CN]20-50%)
            routing_factor = 1.3
            actual_length = total_length * routing_factor
            
            return float(actual_length)
        except Exception as e:
            print(f"[!]️  warning:  calculatenet{net_name}[CN][CN][CN][CN][CN]: {e}")
            return 0.0
    
    def _calculate_net_via_count(self, net_name: str, net_info) -> int:
        """ calculatenet[CN][CN][CN][CN][CN] (routing[CN][CN][CN][CN])"""
        try:
            connection_count = len(net_info.connections)
            
            #  simplified calculate: based on connection[CN][CN][CN][CN][CN][CN][CN][CN]
            if connection_count <= 2:
                # [CN][CN][CN][CN][CN] connection,[CN][CN][CN][CN][CN][CN][CN]
                return 0
            elif connection_count <= 4:
                # [CN][CN][CN][CN][CN],[CN][CN][CN][CN][CN][CN]
                return max(1, connection_count - 2)
            else:
                # [CN][CN]net,[CN][CN][CN][CN][CN][CN][CN][CN][CN][CN] connection
                return max(2, int(connection_count * 0.6))
                
        except Exception as e:
            print(f"[!]️  warning:  calculatenet{net_name}[CN][CN][CN][CN][CN][CN]: {e}")
            return 0

    def _add_net_nodes(self):
        """ addnetnode"""
        nets = self.def_data.get('nets', {})
        if not nets:
            print("[!]️  warning: [CN][CN][CN][CN]netinformation")
            return
        
        features = []
        names = []
        labels = []  # [CN][CN]:[CN][CN] label list[ actual[CN][CN], [CN][CN][CN][CN]]
        hpwl_values = []  #  used forstatisticsHPWL
        
        for net_name, net_info in nets.items():
            # feature[CN][CN]: [net_type_id, connection_count, hpwl] -  addHPWL[CN][CN]feature
            net_type_id = EncodingUtils.encode_net_type(net_name)
            connection_count = len(net_info.connections)
            
            #  calculateHPWL[CN][CN]feature(referenceplace[CN][CN][CN] calculate[CN][CN])
            hpwl = self._calculate_hpwl(net_name, net_info)
            hpwl_values.append(hpwl)
            
            # [CN][CN][CN]DEF[CN][CN] parse[CN][CN][CN]information[CN][CN] label
            wire_length = net_info.wire_length
            via_count = net_info.via_count
            
            features.append([net_type_id, connection_count, hpwl])
            names.append(net_name)
            labels.append([wire_length, float(via_count)])
        
        self.hetero_data['net'].x = torch.tensor(features, dtype=torch.float)
        self.hetero_data['net'].y = torch.tensor(labels, dtype=torch.float)  # [CN][CN]:[CN][CN] label
        self.hetero_data['net'].names = names
        self.net_name_to_idx = {name: i for i, name in enumerate(names)}
        
        # statistics[CN][CN],[CN][CN][CN]HPWLinformation
        if labels:
            wire_lengths = [label[0] for label in labels]
            via_counts = [label[1] for label in labels]
            
            wire_stats = {
                'min': min(wire_lengths),
                'max': max(wire_lengths),
                'mean': sum(wire_lengths) / len(wire_lengths),
                'zero_count': sum(1 for w in wire_lengths if w == 0.0)
            }
            
            via_stats = {
                'min': min(via_counts),
                'max': max(via_counts),
                'mean': sum(via_counts) / len(via_counts),
                'zero_count': sum(1 for v in via_counts if v == 0.0)
            }
            
            hpwl_stats = {
                'min': min(hpwl_values),
                'max': max(hpwl_values),
                'mean': sum(hpwl_values) / len(hpwl_values),
                'zero_count': sum(1 for h in hpwl_values if h == 0.0)
            }
            
            print(f"  ✓ Netnode: {len(names)} , 3[CN]feature ( addHPWL), 2[CN] label")
            print(f"  ✓ [CN][CN]statistics: min={wire_stats['min']:.1f}, max={wire_stats['max']:.1f}, mean={wire_stats['mean']:.1f}, zero_count={wire_stats['zero_count']}")
            print(f"  ✓ [CN][CN]statistics: min={via_stats['min']:.0f}, max={via_stats['max']:.0f}, mean={via_stats['mean']:.1f}, zero_count={via_stats['zero_count']}")
            print(f"  ✓ HPWLstatistics: min={hpwl_stats['min']:.1f}, max={hpwl_stats['max']:.1f}, mean={hpwl_stats['mean']:.1f}, zero_count={hpwl_stats['zero_count']}")
        else:
            print(f"  ✓ Netnode: {len(names)} ")
    
    def _add_io_pin_nodes(self):
        """ addIO_Pinnode"""
        pins = self.def_data.get('pins', {})
        if not pins:
            print("[!]️  warning: [CN][CN][CN][CN]IOpininformation")
            return
        
        features = []
        names = []
        
        for pin_name, pin_info in pins.items():
            # feature[CN][CN]: [x, y, direction_id, layer_id]
            direction_id = EncodingUtils.encode_direction(pin_info.direction)
            layer_id = EncodingUtils.encode_layer(pin_info.layer)
            
            #  positionfeature - [CN][CN][CN][CN][CN][CN]([CN]B[CN]keep consistent)
            pos_x = pin_info.x
            pos_y = pin_info.y
            
            features.append([pos_x, pos_y, direction_id, layer_id])
            names.append(pin_name)
        
        self.hetero_data['io_pin'].x = torch.tensor(features, dtype=torch.float)
        self.hetero_data['io_pin'].names = names
        self.pin_name_to_idx = {name: i for i, name in enumerate(names)}
        
        print(f"  ✓ IO_Pinnode: {len(names)} ")
    
    def _add_gate_net_edges(self):
        """ addGate[CN]Net[CN]edge"""
        if not self.gate_name_to_idx or not self.net_name_to_idx:
            print("[!]️  warning: Gate[CN]Netnode[CN][CN], skipGate-Netedge")
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
                
                # edgefeature: [pin_type_id, cell_type_id] - [CN][CN]B[CN]pinnodefeature
                pin_type_id = EncodingUtils.encode_pin_type(pin_info.get('name', ''))
                comp_info = self.def_data['components'].get(comp_name, {})
                cell_type_id = EncodingUtils.encode_cell_type(comp_info.cell_type if hasattr(comp_info, 'cell_type') else '')
                
                edge_features.append([pin_type_id, cell_type_id])
        
        if edge_indices:
            self.hetero_data['gate', 'connects_to', 'net'].edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            self.hetero_data['gate', 'connects_to', 'net'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        print(f"  ✓ Gate-Netedge: {len(edge_indices)} entries")
    
    def _add_io_pin_net_edges(self):
        """ addIO_Pin[CN]Net[CN]edge"""
        if not self.pin_name_to_idx or not self.net_name_to_idx:
            print("[!]️  warning: IO_Pin[CN]Netnode[CN][CN], skipIO_Pin-Netedge")
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
                
                # edgefeature: [pin_direction_id, net_type_id]
                pin_direction_id = EncodingUtils.encode_direction(pin_info.direction)
                net_type_id = EncodingUtils.encode_net_type(net_name)
                
                edge_features.append([pin_direction_id, net_type_id])
        
        if edge_indices:
            self.hetero_data['io_pin', 'connects_to', 'net'].edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            self.hetero_data['io_pin', 'connects_to', 'net'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        print(f"  ✓ IO_Pin-Netedge: {len(edge_indices)} entries")
    
    def _add_global_features(self):
        """ addglobalfeature[CN] label"""
        die_area = self.die_area
        units_info = self.def_data.get('units', {'dbu_per_micron': 1000})
        dbu_per_micron = units_info.get('dbu_per_micron', 1000) if isinstance(units_info, dict) else units_info
        
        # [CN][CN]globalfeature: [[CN][CN][CN][CN], [CN][CN][CN][CN], [CN][CN] area, DBU[CN][CN]]
        chip_width = die_area[2] - die_area[0]
        chip_height = die_area[3] - die_area[1]
        chip_area = chip_width * chip_height
        
        #  readconfig.mk[CN][CN]CORE_UTILIZATION configuration
        core_utilization_config = self._read_config_utilization()
        
        # extended globalfeature: [[CN][CN][CN][CN], [CN][CN][CN][CN], [CN][CN] area, DBU[CN][CN],  configuration[CN][CN][CN]]
        global_features = torch.tensor([chip_width, chip_height, chip_area, dbu_per_micron, core_utilization_config], dtype=torch.float)
        self.hetero_data.global_features = global_features
        
        # [CN][CN][CN][CN]information - 2x2[CN][CN]:[[[CN][CN][CN]x,[CN][CN][CN]y], [[CN][CN][CN]x,[CN][CN][CN]y]]
        die_coordinates = torch.tensor([[die_area[0], die_area[1]], [die_area[2], die_area[3]]], dtype=torch.float)
        self.hetero_data.die_coordinates = die_coordinates
        
        #  readrouting report[CN][CN]global label
        utilization, hpwl, csv_total_wire_length, csv_total_vias = self._read_placement_labels()
        
        #  calculatePin Density ([CN][CN]pin[CN][CN][CN])
        internal_pin_count = len(self.def_data.get('internal_pins', {}))
        total_pin_count = internal_pin_count + len(self.def_data.get('pins', {}))
        pin_density = (internal_pin_count / total_pin_count * 100.0) if total_pin_count > 0 else 0.0
        
        # [CN][CN][CN][CN]CSV[CN][CN] actual[CN],[CN][CN][CN][CN] calculate[CN][CN][CN][CN][CN]
        if csv_total_wire_length > 0:
            total_wire_length = csv_total_wire_length
        else:
            total_wire_length = self._calculate_total_wire_length()
            
        if csv_total_vias > 0:
            total_vias = csv_total_vias
        else:
            total_vias = self._calculate_total_vias()
        
        # global label: [ actual[CN][CN][CN], HPWL, Pin Density, Total Wire Length, Total Vias]
        y = torch.tensor([utilization, hpwl, pin_density, total_wire_length, total_vias], dtype=torch.float)
        self.hetero_data.y = y
        
        print(f"  ✓ globalfeature: 5[CN] ([CN][CN]={chip_width:.0f}, [CN][CN]={chip_height:.0f},  area={chip_area:.0f}, DBU={dbu_per_micron:.0f},  configuration[CN][CN][CN]={core_utilization_config:.1f}%)")
        print(f"  ✓ global label: 5[CN] ( actual[CN][CN][CN]={utilization}%, HPWL={hpwl}um, Pin[CN][CN]={pin_density:.2f}%, [CN][CN][CN]={total_wire_length}, [CN][CN][CN]={total_vias})")
    
    def _read_config_utilization(self):
        """[CN]route_data_extract.csv[CN][CN] read corresponding to[CN][CN][CN]CORE_UTILIZATION configuration"""
        try:
            # [CN]DEFextract design name from file path([CN][CN]_place.def[CN][CN])
            if self.def_file_path:
                import os
                filename = os.path.basename(self.def_file_path)
                #  extract_place[CN][CN][CN]part[CN][CN][CN][CN][CN][CN]
                design_name = filename.replace('_place.def', '')
            else:
                # [CN][CN][CN][CN]:[CN][CN]DEF[CN][CN][CN][CN][CN][CN][CN][CN][CN]
                design_name = self.def_data.get('design_name', '')
            
            # CSV[CN][CN][CN][CN]
            csv_path = 'route_data_extract.csv'
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # [CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN]
                    csv_design_name = row['design_name']
                    
                    # 1. exact match
                    if design_name == csv_design_name:
                        match_found = True
                    # 2. DEF[CN][CN][CN][CN][CN]CSV[CN][CN][CN]
                    elif design_name in csv_design_name:
                        match_found = True
                    # 3. CSV[CN][CN][CN]DEF[CN][CN][CN][CN][CN]
                    elif csv_design_name in design_name:
                        match_found = True
                    # 4.  process[CN][CN][CN][CN], extract[CN][CN][CN][CN][CN][CN][CN][CN][CN][CN]
                    elif '(' in csv_design_name and ')' in csv_design_name:
                        bracket_name = csv_design_name.split('(')[1].split(')')[0]
                        if design_name == bracket_name:
                            match_found = True
                        else:
                            match_found = False
                    # 5. [CN][CN][CN][CN][CN][CN]
                    elif '(' in csv_design_name:
                        prefix_name = csv_design_name.split('(')[0]
                        if design_name == prefix_name:
                            match_found = True
                        else:
                            match_found = False
                    # 6.  process[CN][CN][CN][CN][CN][CN][CN]
                    elif (design_name.replace('_top', '') == csv_design_name or 
                          design_name.replace('_core', '') == csv_design_name or
                          design_name == csv_design_name + '_top' or
                          design_name == csv_design_name + '_core'):
                        match_found = True
                    else:
                        match_found = False
                    
                    if match_found:
                        core_util = row['core_utilization']
                        if core_util and core_util.strip():  # [CN][CN][CN][CN][CN][CN]
                            print(f"[OK] successfully matched design: DEF[CN][CN]'{design_name}' <-> CSV[CN][CN]'{csv_design_name}' (core_utilization={core_util})")
                            return float(core_util)
            
            print(f"[!]️  warning: [CN]CSV[CN][CN][CN][CN][CN][CN][CN][CN] {design_name} [CN]core_utilization data")
            return 0.0  #  default[CN]
        except Exception as e:
            print(f"[!]️  warning: [CN][CN] readroute_data_extract.csv[CN][CN]: {e}")
            return 0.0  #  default[CN]
    
    def _read_placement_labels(self):
        """[CN]route_data_extract.csv[CN][CN] read corresponding to[CN][CN][CN]design_utilization,hpwl_after,total_wire_length[CN]total_vias label"""
        try:
            # [CN]DEFextract design name from file path([CN][CN]_place.def[CN][CN])
            if self.def_file_path:
                import os
                filename = os.path.basename(self.def_file_path)
                #  extract_place[CN][CN][CN]part[CN][CN][CN][CN][CN][CN]
                design_name = filename.replace('_place.def', '')
            else:
                # [CN][CN][CN][CN]:[CN][CN]DEF[CN][CN][CN][CN][CN][CN][CN][CN][CN]
                design_name = self.def_data.get('design_name', '')
            
            # CSV[CN][CN][CN][CN]
            csv_path = 'route_data_extract.csv'
            
            utilization = 0.0
            hpwl = 0.0
            total_wire_length = 0.0
            total_vias = 0.0
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # [CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN]
                    csv_design_name = row['design_name']
                    
                    # 1. exact match
                    if design_name == csv_design_name:
                        match_found = True
                    # 2. DEF[CN][CN][CN][CN][CN]CSV[CN][CN][CN]([CN] ac97_top [CN] ac97_ctrl [CN][CN][CN][CN],[CN] aes [CN] systemcaes(aes) [CN][CN][CN])
                    elif design_name in csv_design_name:
                        match_found = True
                    # 3. CSV[CN][CN][CN]DEF[CN][CN][CN][CN][CN]([CN] ac97_ctrl [CN] ac97_top [CN][CN][CN][CN])
                    elif csv_design_name in design_name:
                        match_found = True
                    # 4.  process[CN][CN][CN][CN], extract[CN][CN][CN][CN][CN][CN][CN][CN][CN][CN]([CN] vga_lcd(vga_enh_top) -> vga_enh_top)
                    elif '(' in csv_design_name and ')' in csv_design_name:
                        bracket_name = csv_design_name.split('(')[1].split(')')[0]
                        if design_name == bracket_name:
                            match_found = True
                        else:
                            match_found = False
                    # 5. [CN][CN][CN][CN][CN][CN],[CN][CN]DEF[CN][CN][CN][CN][CN][CN]CSV[CN][CN][CN][CN][CN]part
                    elif '(' in csv_design_name:
                        prefix_name = csv_design_name.split('(')[0]
                        if design_name == prefix_name:
                            match_found = True
                        else:
                            match_found = False
                    # 6.  process[CN][CN][CN][CN][CN][CN][CN]([CN] tv80_core vs tv80, uart_top vs uart)
                    elif (design_name.replace('_top', '') == csv_design_name or 
                          design_name.replace('_core', '') == csv_design_name or
                          design_name == csv_design_name + '_top' or
                          design_name == csv_design_name + '_core'):
                        match_found = True
                    else:
                        match_found = False
                    
                    if match_found:
                        #  readdesign_utilization
                        design_util = row['design_utilization']
                        if design_util and design_util.strip():
                            utilization = float(design_util)
                        
                        #  readhpwl_after
                        hpwl_after = row['hpwl_after']
                        if hpwl_after and hpwl_after.strip():
                            hpwl = float(hpwl_after)
                        
                        #  readtotal_wire_length
                        wire_length = row['total_wire_length']
                        if wire_length and wire_length.strip():
                            total_wire_length = float(wire_length)
                        
                        #  readtotal_vias
                        vias = row['total_vias']
                        if vias and vias.strip():
                            total_vias = float(vias)
                        
                        print(f"[OK] successfully matched design: DEF[CN][CN]'{design_name}' <-> CSV[CN][CN]'{csv_design_name}'")
                        return utilization, hpwl, total_wire_length, total_vias
            
            print(f"[!]️  warning: [CN]CSV[CN][CN][CN][CN][CN][CN][CN][CN] {design_name} [CN]placement label data")
            return 0.0, 0.0, 0.0, 0.0  #  default[CN]
        except Exception as e:
            print(f"[!]️  warning: [CN][CN] readroute_data_extract.csv[CN][CN]: {e}")
            return 0.0, 0.0, 0.0, 0.0  #  default[CN]
    
    def _calculate_total_wire_length(self):
        """ calculate[CN][CN][CN] (routing[CN][CN][CN][CN])"""
        try:
            # [CN]DEF data[CN] calculate[CN][CN]net[CN] actual[CN][CN][CN][CN]
            total_length = 0.0
            nets = self.def_data.get('nets', {})
            
            for net_name, net_info in nets.items():
                if hasattr(net_info, 'connections') and net_info.connections:
                    #  simplified calculate: based on connection[CN][CN][CN][CN][CN][CN][CN]
                    connection_count = len(net_info.connections)
                    if connection_count > 1:
                        # [CN][CN][CN][CN]: connection[CN][CN][CN],[CN][CN][CN][CN]
                        estimated_length = connection_count * 100  # base unit length
                        total_length += estimated_length
            
            return total_length
        except Exception as e:
            print(f"[!]️  warning:  calculate[CN][CN][CN][CN][CN][CN]: {e}")
            return 0.0
    
    def _calculate_total_vias(self):
        """ calculate[CN][CN][CN][CN] (routing[CN][CN][CN][CN])"""
        try:
            # [CN]DEF data[CN] calculate[CN][CN][CN][CN]
            total_vias = 0
            nets = self.def_data.get('nets', {})
            
            for net_name, net_info in nets.items():
                if hasattr(net_info, 'connections') and net_info.connections:
                    #  simplified calculate: based on connection[CN][CN][CN][CN][CN][CN][CN][CN]
                    connection_count = len(net_info.connections)
                    if connection_count > 2:
                        # [CN][CN]net[CN][CN][CN][CN][CN][CN]
                        estimated_vias = max(1, connection_count - 2)
                        total_vias += estimated_vias
            
            return total_vias
        except Exception as e:
            print(f"[!]️  warning:  calculate[CN][CN][CN][CN][CN][CN][CN]: {e}")
            return 0

# ============================================================================
# GRAPH VALIDATOR SECTION - [CN][CN][CN][CN]region
# ============================================================================

class GraphValidator:
    """[CN] structure[CN][CN][CN]
    
    [CN][CN]generate[CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN] structure[CN][CN][CN]。
    """
    
    @staticmethod
    def validate_heterograph(hetero_data: HeteroData) -> bool:
        """[CN][CN][CN][CN][CN] structure
        
        Args:
            hetero_data: [CN][CN][CN][CN][CN][CN][CN] data
            
        Returns:
            [CN][CN][CN][CN][CN][CN]
        """
        print("[SEARCH] [CN][CN][CN][CN]C[CN][CN][CN][CN] structure...")
        
        try:
            # [CN][CN]nodetype
            expected_node_types = {'gate', 'net', 'io_pin'}
            actual_node_types = set(hetero_data.node_types)
            
            if not expected_node_types.issubset(actual_node_types):
                missing = expected_node_types - actual_node_types
                print(f"[X] [CN][CN]nodetype: {missing}")
                return False
            
            # [CN][CN]edgetype
            expected_edge_types = {
                ('gate', 'connects_to', 'net'),
                ('io_pin', 'connects_to', 'net')
            }
            actual_edge_types = set(hetero_data.edge_types)
            
            if not expected_edge_types.issubset(actual_edge_types):
                missing = expected_edge_types - actual_edge_types
                print(f"[X] [CN][CN]edgetype: {missing}")
                return False
            
            # [CN][CN]nodefeature dimension
            if 'gate' in hetero_data.node_types:
                gate_features = hetero_data['gate'].x
                if gate_features.shape[1] != 7:
                    print(f"[X] Gatenodefeature dimensionerror: [CN][CN]7[CN], actual{gate_features.shape[1]}[CN]")
                    return False
            
            if 'net' in hetero_data.node_types:
                net_features = hetero_data['net'].x
                if net_features.shape[1] != 3:
                    print(f"[X] Netnodefeature dimensionerror: [CN][CN]3[CN], actual{net_features.shape[1]}[CN]")
                    return False
            
            if 'io_pin' in hetero_data.node_types:
                pin_features = hetero_data['io_pin'].x
                if pin_features.shape[1] != 4:
                    print(f"[X] IO_Pinnodefeature dimensionerror: [CN][CN]4[CN], actual{pin_features.shape[1]}[CN]")
                    return False
            
            # [CN][CN]edgefeature dimension
            if ('gate', 'connects_to', 'net') in hetero_data.edge_types:
                edge_attr = hetero_data['gate', 'connects_to', 'net'].edge_attr
                if edge_attr.shape[1] != 2:
                    print(f"[X] Gate-Netedgefeature dimensionerror: [CN][CN]2[CN], actual{edge_attr.shape[1]}[CN]")
                    return False
            
            if ('io_pin', 'connects_to', 'net') in hetero_data.edge_types:
                edge_attr = hetero_data['io_pin', 'connects_to', 'net'].edge_attr
                if edge_attr.shape[1] != 2:
                    print(f"[X] IO_Pin-Netedgefeature dimensionerror: [CN][CN]2[CN], actual{edge_attr.shape[1]}[CN]")
                    return False
            
            # Check for global features
            if hasattr(hetero_data, 'y'):
                global_labels = hetero_data.y
                if global_labels.shape[0] != 5:
                    print(f"[X] global label dimensionerror: [CN][CN]5[CN], actual{global_labels.shape[0]}[CN]")
                    return False
            
            print("[OK] C[CN][CN][CN][CN] structure[CN][CN][CN][CN]")
            return True
            
        except Exception as e:
            print(f"[X] [CN][CN][CN][CN][CN][CN][CN]error: {e}")
            return False

# ============================================================================
# MAIN CONVERSION FUNCTION SECTION - [CN] convert[CN][CN]region
# ============================================================================

def convert_def_to_c_heterograph(def_file_path: str, output_path: Optional[str] = None) -> HeteroData:
    """[CN]DEF[CN][CN] convert[CN]C[CN][CN][CN][CN]
    
    Args:
        def_file_path:  input[CN]DEF[CN][CN][CN][CN]
        output_path:  output[CN].pt[CN][CN][CN][CN],[CN][CN][CN]None[CN][CN][CN]generate
        
    Returns:
         convert[CN][CN][CN][CN][CN] data
        
    Raises:
        RuntimeError: [CN]occurred during conversionerror[CN]
    """
    print("[START] [CN][CN]DEF[CN]C[CN][CN][CN][CN] convert...")
    print(f"[FILE]  input[CN][CN]: {def_file_path}")
    print()
    
    # 1.  parseDEF[CN][CN]
    parser = DEFParser(def_file_path)
    def_data = parser.parse()
    
    print(f"[CHART]  parse[CN][CN]:")
    print(f"  - [CN][CN][CN][CN]: {def_data.get('design_name', 'unknown')}")
    print(f"  -  component[CN][CN]: {len(def_data.get('components', {}))}")
    print(f"  - pin[CN][CN]: {len(def_data.get('pins', {}))}")
    print(f"  - net[CN][CN]: {len(def_data.get('nets', {}))}")
    print(f"  - [CN][CN][CN]:   {len(def_data.get('tracks', {}))}")
    print()
    
    # 2.  buildC[CN][CN][CN][CN]
    builder = CHeteroGraphBuilder(def_data, def_file_path)
    c_graph = builder.build()
    
    # 3. [CN][CN][CN] structure
    if not GraphValidator.validate_heterograph(c_graph):
        raise RuntimeError("[CN] structure[CN][CN]failed")
    
    # 4. Save result
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(def_file_path))[0]
        output_path = f"{base_name}_c_heterograph.pt"
    
    print(f"[SAVE] saveC[CN][CN][CN][CN][CN]: {output_path}")
    torch.save(c_graph, output_path, pickle_protocol=4)
    
    print("\n🎉  convertcompleted！")
    print(f"📈 [CN][CN][CN] structurestatistics:")
    print(f"  - nodetype: {len(c_graph.node_types)}")
    print(f"  - edgetype: {len(c_graph.edge_types)}")
    
    total_nodes = sum(c_graph[nt].x.shape[0] for nt in c_graph.node_types)
    total_edges = sum(c_graph[et].edge_index.shape[1] for et in c_graph.edge_types)
    print(f"  - [CN]node[CN]: {total_nodes}")
    print(f"  - [CN]edge[CN]: {total_edges}")
    
    # [CN][CN]globalfeatureinformation
    if hasattr(c_graph, 'global_features'):
        global_feat = c_graph.global_features
        print(f"  - globalfeature: {global_feat.shape[0]} [CN]")
        print(f"    * [CN][CN][CN][CN]: {global_feat[0]:.0f}")
        print(f"    * [CN][CN][CN][CN]: {global_feat[1]:.0f}")
        print(f"    * [CN][CN] area: {global_feat[2]:.0f}")
        print(f"    * DBU[CN][CN]: {global_feat[3]:.0f}")
    
    return c_graph

# ============================================================================
# COMMAND LINE INTERFACE SECTION - [CN][CN][CN][CN][CN]region
# ============================================================================

def main():
    """Main function
    
    [CN][CN][CN][CN][CN][CN][CN][CN][CN][CN], process[CN][CN][CN] parameter[CN][CN][CN] convert[CN][CN]。
    [CN][CN][CN][CN][CN][CN] parameter:
    - def_file:  input[CN]DEF[CN][CN][CN][CN]( position parameter)
    - -o/--output:  output[CN].pt[CN][CN][CN][CN]([CN][CN])
    
    Returns:
        int: [CN][CN][CN][CN][CN](0[CN][CN]success,1[CN][CN]failed)
    """
    parser = argparse.ArgumentParser(description='C[CN][CN][CN][CN]generate[CN] - DEF[CN]C[CN][CN][CN][CN] convert[CN][CN]')
    parser.add_argument('def_file', nargs='?', default='/Users/david/Desktop/design_1.0/def/3_5_place_dp.def', 
                       help=' input[CN]DEF[CN][CN][CN][CN]')
    parser.add_argument('-o', '--output', default=None,
                       help=' output[CN].pt[CN][CN][CN][CN]')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.def_file):
        print(f"[X] error: DEF[CN][CN][CN][CN][CN]: {args.def_file}")
        return 1
    
    try:
        c_graph = convert_def_to_c_heterograph(
            args.def_file, args.output
        )
        return 0
    except Exception as e:
        print(f"[X] occurred during conversionerror: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())