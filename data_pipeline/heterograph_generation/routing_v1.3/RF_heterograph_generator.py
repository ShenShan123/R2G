#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Routing[CN][CN]F[CN][CN][CN][CN]generate[CN](Gate-Gate connection[CN][CN])

[CN][CN][CN] based onrouting[CN][CN]DEF[CN][CN],[CN][CN]F[CN] convertFeatures:
1. [CN][CN]pinnode,IO_Pin-Pinedge,Pin-Pinedgeandgate-pinedge
2.  build[CN][CN]gate-gateedge,edgefeature[CN]6[CN]:[pin_type_id, cell_type_id] + [net_type_id, connection_count] + [pin_type_id, cell_type_id]
3.  buildIO_Pin-gateedge,edgefeature[CN]4[CN]:[net_type_id, connection_count] + [pin_type_id, cell_type_id]
4. edge label[CN][CN]D[CN][CN][CN]:gate-gateedge label[CN][wire_length, via_count],IO_Pin-gateedge label[CN][wire_length, via_count]
5. global label[CN][CN]routing[CN][CN][CN][CN][CN][CN]:[utilization, hpwl, total_wire_length, total_vias]

 convert[CN][CN]:
- [CN][CN][CN]net,[CN][CN][CN][CN][CN] outputpin([CN]ZN,Z,Q[CN])[CN][CN][CN][CN]
- [CN] outputpin[CN][CN][CN]gate connection[CN][CN]netall inputs inpin[CN][CN][CN]gate
- edgefeature contains[CN]gate[CN]pininformation,netinformation[CN][CN][CN]gate[CN]pininformation
- edge label[CN][CN]routing[CN][CN][CN] actual[CN][CN][CN][CN][CN][CN][CN]

Author: EDA for AI Team
date: 2024

[CN][CN][CN][CN]:
    python F_heterograph_generator.py input_route.def -o output.pt
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
# DATA STRUCTURES SECTION -  data structure[CN][CN]region
# ============================================================================

# [CN][CN] celltype mapping - global[CN][CN]
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

@dataclass
class ComponentInfo:
    """ componentinformation data structure"""
    name: str                                   #  component[CN][CN][CN][CN]
    cell_type: str                              #  celltype([CN]INV_X1, NAND2_X1[CN])
    position: Tuple[float, float]               #  component position[CN][CN](x, y)
    orientation: str = 'N'                      #  component[CN][CN](N, S, E, W, FN, FS, FE, FW)
    pins: Dict[str, Any] = field(default_factory=dict)  #  componentpininformation dictionary
    size: float = 1.0                           #  component[CN][CN][CN][CN]( based on drive strength)
    
@dataclass
class NetInfo:
    """netinformation data structure"""
    name: str                                   # net[CN][CN]
    connections: List[Tuple[str, str]] = field(default_factory=list)  #  connection list:( component[CN], pin[CN])
    routing: List[Dict] = field(default_factory=list)                 # [CN][CN]information list
    net_type: int = 0                          # nettype encoding(0-[CN][CN],1-[CN][CN],2-[CN],3-[CN][CN][CN])
    weight: float = 1.0                        # net[CN][CN]( based on connection[CN][CN][CN])

@dataclass
class PinInfo:
    """pininformation data structure"""
    name: str                                   # pin[CN][CN]
    component: str                              # [CN][CN] component[CN][CN]
    direction: str = 'INOUT'                    # pin[CN][CN](INPUT, OUTPUT, INOUT, FEEDTHRU)
    layer: str = 'metal1'                       # pin[CN][CN][CN][CN][CN]
    position: Tuple[float, float] = (0.0, 0.0)  # pin position[CN][CN]
    net: str = ''                               #  connection[CN]net[CN][CN]

@dataclass
class GateGateEdge:
    """Gate-Gateedgeinformation data structure"""
    source_gate: str                            # [CN]gate[CN][CN]
    target_gate: str                            # [CN][CN]gate[CN][CN]
    source_pin_type: int                        # [CN]pintype encoding
    source_cell_type: int                       # [CN]celltype encoding
    net_type: int                              # nettype encoding
    connection_count: int                       #  connection[CN][CN]
    target_pin_type: int                        # [CN][CN]pintype encoding
    target_cell_type: int                       # [CN][CN]celltype encoding
    net_name: str = ''                         # net[CN][CN]( used for[CN][CN])

@dataclass
class IO_PinGateEdge:
    """IO_Pin-Gateedgeinformation data structure"""
    io_pin_name: str                           # IOpin[CN][CN]
    gate_name: str                             # Gate[CN][CN]
    net_type: int                              # nettype encoding (net_type_id)
    connection_count: int                       #  connection[CN][CN]
    gate_pin_type: int                         # Gatepintype encoding (pin_type_id)
    gate_cell_type: int                        # Gate celltype encoding (cell_type_id)
    net_name: str = ''                         # net[CN][CN]( used for[CN][CN])
    gate_pin_count: int = 1                    # [CN]gate[CN][CN]net[CN][CN]pin[CN][CN]

# ============================================================================
# DEF PARSER SECTION - DEF[CN][CN] parse[CN]region([CN][CN]D[CN][CN][CN][CN] parse[CN])
# ============================================================================

class DEFParser:
    """DEF[CN][CN] parse[CN]
    
    [CN][CN] parseDEF[CN][CN][CN] extract structure[CN] data,[CN][CN]:
    - [CN][CN]information([CN][CN],[CN][CN],[CN][CN]region)
    -  componentinformation( position,type,[CN][CN])
    - pininformation(IOpin[CN]position and orientation)
    - netinformation( connection[CN][CN])
    - [CN]information([CN][CN][CN][CN][CN][CN][CN])
    - [CN]information([CN][CN] cell[CN])
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
            'internal_pins': {},  #  new:[CN][CN]pininformation
            'tracks': {},
            'vias': {},
            'rows': {}
        }
    
    def parse(self) -> Dict:
        """ parseDEF[CN][CN][CN] return structure[CN] data"""
        try:
            with open(self.def_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            #  parse[CN]part
            self._parse_design_info(content)
            self._parse_components(content)
            self._parse_pins(content)
            self._parse_nets(content)
            self._parse_tracks(content)
            self._parse_vias(content)
            self._parse_rows(content)
            
            return self.def_data
            
        except Exception as e:
            print(f"[X] DEF[CN][CN] parsefailed: {e}")
            raise
    
    def _parse_design_info(self, content: str):
        """ parse[CN][CN][CN][CN]information"""
        #  parse[CN][CN][CN][CN]
        design_match = re.search(r'DESIGN\s+(\w+)', content)
        if design_match:
            self.def_data['design_name'] = design_match.group(1)
        
        #  parse[CN][CN]
        units_match = re.search(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', content)
        if units_match:
            self.def_data['units'] = {
                'dbu_per_micron': int(units_match.group(1))
            }
        
        #  parse[CN][CN]region
        diearea_match = re.search(r'DIEAREA\s+\(\s*([\d\-]+)\s+([\d\-]+)\s*\)\s+\(\s*([\d\-]+)\s+([\d\-]+)\s*\)', content)
        if diearea_match:
            self.def_data['die_area'] = tuple(map(int, diearea_match.groups()))
    
    def _parse_components(self, content: str):
        """ parse componentinformation"""
        components_section = re.search(r'COMPONENTS\s+(\d+)\s*;(.*?)END\s+COMPONENTS', content, re.DOTALL)
        if not components_section:
            return
        
        components_text = components_section.group(2)
        #  parsePLACED[CN][CN][CN] component,[CN][CN]FIXED[CN][CN][CN] component([CN]TAPCELL[CN])
        # [CN][CN][CN][CN][CN][CN][CN]SOURCE TIMING[CN]
        component_pattern = r'-\s+([^\s]+)\s+(\w+)\s+.*?\+\s+PLACED\s+\(\s*([\d\-]+)\s+([\d\-]+)\s*\)\s+(\w+)'
        
        for match in re.finditer(component_pattern, components_text):
            instance_name = match.group(1)
            cell_type = match.group(2)
            x = int(match.group(3))
            y = int(match.group(4))
            orientation = match.group(5)
            
            self.def_data['components'][instance_name] = {
                'cell_type': cell_type,
                'position': (x, y),
                'orientation': orientation
            }
    
    def _parse_pins(self, content: str):
        """ parseIOpininformation -  extract[CN][CN][CN] position[CN][CN][CN][CN]information"""
        pins_section = re.search(r'PINS\s+(\d+)\s*;(.*?)END\s+PINS', content, re.DOTALL)
        if not pins_section:
            return
        
        pins_text = pins_section.group(2)
        
        # [CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN]pin[CN][CN]
        # [CN]pin[CN][CN][CN] "- pin_name" [CN][CN],[CN] ";" [CN][CN]
        pin_pattern = r'-\s+([\w\[\]_]+)\s+\+\s+NET\s+([\w\[\]_]+)\s+\+\s+DIRECTION\s+(\w+).*?;'
        
        for match in re.finditer(pin_pattern, pins_text, re.DOTALL):
            pin_name = match.group(1)
            net_name = match.group(2)
            direction = match.group(3)
            pin_def = match.group(0)
            
            #  parsePORTpart[CN][CN]information
            layer_match = re.search(r'\+\s+LAYER\s+(\w+)', pin_def)
            layer = layer_match.group(1) if layer_match else 'metal1'
            
            #  parsePLACED positioninformation
            position_match = re.search(r'\+\s+PLACED\s+\(\s*([\d\-]+)\s+([\d\-]+)\s*\)', pin_def)
            if position_match:
                x_pos = int(position_match.group(1))
                y_pos = int(position_match.group(2))
                position = (x_pos, y_pos)
            else:
                position = (0, 0)  #  default position
            
            #  parseUSEinformation([CN][CN])
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
        """ parsenetinformation[CN][CN][CN]pininformation([CN][CN]NETS[CN]SPECIALNETS)"""
        #  parse[CN][CN]net(NETSpart)
        nets_section = re.search(r'NETS\s+(\d+)\s*;(.*?)END\s+NETS', content, re.DOTALL)
        if nets_section:
            nets_text = nets_section.group(2)
            self._parse_nets_section(nets_text, net_type=0)  # [CN][CN]net
        
        #  parse[CN][CN]net(SPECIALNETSpart)
        specialnets_section = re.search(r'SPECIALNETS\s+(\d+)\s*;(.*?)END\s+SPECIALNETS', content, re.DOTALL)
        if specialnets_section:
            specialnets_text = specialnets_section.group(2)
            self._parse_nets_section(specialnets_text, net_type=1)  # [CN][CN]net([CN][CN]/[CN])

    def _parse_nets_section(self, nets_text: str, net_type: int):
        """ parsenetpart[CN][CN][CN][CN][CN]"""
        # [CN][CN][CN]net parse - [CN][CN][CN][CN][CN]net[CN][CN][CN] connectionformat, contains[CN][CN][CN][CN]
        # [CN][CN]format: - net_name ( comp1 pin1 ) ( comp2 pin2 ) ... + USE SIGNAL ;
        net_pattern = r'-\s+([\w\[\]_$\\.]+)\s+((?:\([^)]+\)\s*)+)(.*?);'
        
        for match in re.finditer(net_pattern, nets_text, re.DOTALL):
            net_name = match.group(1)
            connections_text = match.group(2)
            routing_text = match.group(3)
            
            connections = []
            # [CN][CN][CN] connection: ( component_name pin_name )
            conn_pattern = r'\(\s*([^\s]+)\s+([^\s]+)\s*\)'
            for conn_match in re.finditer(conn_pattern, connections_text):
                comp_name = conn_match.group(1)
                pin_name = conn_match.group(2)
                connections.append((comp_name, pin_name))
                
                #  extract[CN][CN]pininformation
                self._extract_internal_pin(comp_name, pin_name, net_name)
            
            #  parse[CN][CN]information, calculate[CN][CN][CN][CN][CN][CN][CN]
            wire_length, via_count = self._parse_routing_info(routing_text)
            
            self.def_data['nets'][net_name] = {
                'connections': connections,
                'net_type': net_type,  # [CN][CN][CN][CN][CN]nettype
                'weight': 1.0,
                'wire_length': wire_length,
                'via_count': via_count
            }
    
    def _parse_tracks(self, content: str):
        """ parse[CN]information - [CN][CN] processX[CN]Y[CN][CN][CN]TRACKS"""
        track_pattern = r'TRACKS\s+(\w+)\s+([\d\-]+)\s+DO\s+(\d+)\s+STEP\s+(\d+)\s+LAYER\s+(\w+)'
        
        for match in re.finditer(track_pattern, content):
            direction = match.group(1)
            start = int(match.group(2))
            count = int(match.group(3))
            step = int(match.group(4))
            layer = match.group(5)
            
            # Initializelayer data structure([CN][CN][CN][CN][CN])
            if layer not in self.def_data['tracks']:
                self.def_data['tracks'][layer] = {
                    'x_direction': None,
                    'y_direction': None,
                    'count': 0,
                    'step': 0
                }
            
            # [CN][CN][CN][CN][CN][CN] data
            if direction == 'X':
                self.def_data['tracks'][layer]['x_direction'] = {
                    'start': start,
                    'count': count,
                    'step': step
                }
                # [CN][CN]X[CN][CN][CN] data[CN][CN][CN][CN]feature([CN][CN]X[CN][CN][CN][CN][CN])
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
                # [CN][CN][CN][CN][CN][CN][CN][CN][CN]feature,[CN][CN]Y[CN][CN][CN] data
                if self.def_data['tracks'][layer]['direction'] is None:
                    self.def_data['tracks'][layer]['direction'] = direction
                    self.def_data['tracks'][layer]['start'] = start
                    self.def_data['tracks'][layer]['count'] = count
                    self.def_data['tracks'][layer]['step'] = step
    
    def _parse_vias(self, content: str):
        """ parse[CN][CN]information"""
        #  simplified[CN][CN][CN] parse
        via_pattern = r'VIA\s+(\w+)'
        for match in re.finditer(via_pattern, content):
            via_name = match.group(1)
            self.def_data['vias'][via_name] = {
                'layers': ['metal1', 'metal2'],  #  simplified process
                'position': (0, 0)
            }
    
    def _parse_rows(self, content: str):
        """ parse[CN]information"""
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
        """ extract[CN][CN]pininformation"""
        if comp_name == 'PIN':  #  skipIOpin
            return
        
        #  based on component position[CN][CN]pin position
        comp_info = self.def_data['components'].get(comp_name, {})
        comp_pos = comp_info.get('position', (0, 0))
        
        #  create[CN][CN]pin[CN][CN][CN][CN][CN]
        internal_pin_id = f"{comp_name}_{pin_name}"
        
        self.def_data['internal_pins'][internal_pin_id] = {
            'name': pin_name,
            'component': comp_name,
            'pin_type': self._infer_pin_type(pin_name),
            'net': net_name,
            'position': comp_pos  #  simplified:using component positions
        }
    
    def _infer_pin_type(self, pin_name: str) -> str:
        """[CN][CN]pin[CN][CN][CN][CN]pintype"""
        pin_name_upper = pin_name.upper()
        
        #  outputpin[CN][CN]
        output_patterns = ['Z', 'ZN', 'Q', 'QN', 'Y', 'CO', 'S']
        if any(pin_name_upper.startswith(pattern) for pattern in output_patterns):
            return 'OUTPUT'
        
        #  inputpin[CN][CN]
        input_patterns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'CK', 'CLK', 'CP', 'R', 'RN', 'RST', 'EN', 'S', 'SE', 'SI', 'SN']
        if any(pin_name_upper.startswith(pattern) for pattern in input_patterns):
            return 'INPUT'
        
        return 'UNKNOWN'

    def _parse_routing_info(self, routing_text: str) -> tuple:
        """ parserouting[CN][CN][CN][CN][CN]information, calculate actual[CN][CN][CN][CN][CN][CN][CN]"""
        import re
        
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
                # [CN][CN][CN][CN] parse[CN][CN][CN]
                continue
        
        # statistics[CN][CN][CN][CN] - [CN][CN]via[CN][CN][CN]
        via_pattern = r'via\d*|Via\d*|VIA\d*'
        via_count = len(re.findall(via_pattern, routing_text, re.IGNORECASE))
        
        return wire_length, via_count

# ============================================================================
# FEATURE CONFIG SECTION - feature configurationregion([CN][CN]D[CN][CN][CN][CN] configuration)
# ============================================================================

class FeatureConfig:
    """feature configuration[CN] - [CN][CN][CN][CN][CN][CN]feature mapping[CN] parameter"""
    
    # [CN][CN]global mapping[CN]
    # COMPLETE_CELL_TYPE_MAPPING [CN][CN]global[CN][CN]
    
    # [CN][CN] mapping
    ORIENTATION_MAPPING = {
        'N': 0, 'S': 1, 'E': 2, 'W': 3,
        'FN': 4, 'FS': 5, 'FE': 6, 'FW': 7
    }
    
    # pin[CN][CN] mapping
    PIN_DIRECTION_MAPPING = {
        'INPUT': 0, 'OUTPUT': 1, 'INOUT': 2, 'FEEDTHRU': 3
    }
    
    # pintype mapping - [CN][CN][CN][CN] encoding,based on actualDEF[CN][CN][CN][CN][CN][CN]
    PIN_TYPE_MAPPING = {
        # [CN][CN] inputpintype (0-4) - [CN][CN] encoding
        'A': 0, 'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0,  #  data inputA
        'B': 1, 'B1': 1, 'B2': 1, 'B3': 1, 'B4': 1,  #  data inputB
        'C': 2, 'C1': 2, 'C2': 2, 'C3': 2, 'C4': 2,  #  data inputC
        'D': 3, 'D1': 3, 'D2': 3, 'D3': 3, 'D4': 3,  #  data inputD
        'S': 4, 'S0': 4, 'S1': 4,  # [CN][CN][CN][CN]
        
        # [CN][CN][CN][CN]pintype (5-7) - [CN][CN] encoding
        'CI': 5, 'CIN': 5,  # [CN][CN] input
        'CLK': 6, 'CK': 6,  # [CN][CN] input - [CN][CN][CN] encoding6,[CN][CN] actual data[CN][CN][CN][CN]
        'EN': 7, 'G': 7,    # [CN][CN][CN][CN]
        
        #  outputpintype (8-11) - part[CN][CN][CN] encoding[CN][CN][CN][CN][CN] data
        'Y': 8, 'CO': 8, 'COUT': 8,  # [CN][CN][CN][CN] output
        'Q': 9, 'QN': 9,             # [CN][CN][CN][CN] output
        'Z': 10, 'ZN': 10,           # [CN] output - [CN][CN][CN] encoding10, actual data[CN][CN][CN][CN][CN]
        'OUT': 11,                   # [CN][CN] output - [CN][CN][CN] encoding11, actual data[CN][CN][CN][CN]
        
        # [CN][CN]pintype (12-13) - [CN][CN] encoding
        'VDD': 12, 'VCC': 12, 'VPWR': 12,  # [CN][CN]
        'VSS': 13, 'GND': 13, 'VGND': 13,  # [CN]
        
        # [CN][CN]pintype
        'UNKNOWN': 14  # changed to[CN][CN] encoding
    }
    
    # [CN] mapping
    LAYER_MAPPING = {
        'metal1': 0, 'metal2': 1, 'metal3': 2, 'metal4': 3, 'metal5': 4, 'metal6': 5,
        'metal7': 6, 'metal8': 7, 'metal9': 8, 'metal10': 9,
        'via1': 10, 'via2': 11, 'via3': 12, 'via4': 13, 'via5': 14,
        'via6': 15, 'via7': 16, 'via8': 17, 'via9': 18,
        'poly': 19, 'contact': 20, 'unknown': 21
    }
    
    # nettype mapping
    NET_TYPE_MAPPING = {
        'signal': 0, 'power': 1, 'ground': 2, 'clock': 3, 'reset': 4, 'scan': 5
    }
    
    # feature dimension configuration
    FEATURE_DIMS = {
        'component_complete_type': len(COMPLETE_CELL_TYPE_MAPPING),  # [CN][CN] celltype([CN][CN]type+ drive strength)
        'component_orientation': len(ORIENTATION_MAPPING),
        'pin_direction': len(PIN_DIRECTION_MAPPING),
        'pin_type': len(PIN_TYPE_MAPPING),        # pintype dimension
        'net_type': len(NET_TYPE_MAPPING),
        'layer_type': len(LAYER_MAPPING),
        'coordinate_dims': 2,  # x, y ([CN][CN][CN]width, height[CN][CN])
        'area_dim': 1,
        'count_dim': 1         # [CN][CN] dimension
    }

# ============================================================================
# ENCODING UTILS SECTION -  encoding[CN][CN][CN]region([CN][CN]D[CN][CN][CN][CN] encoding[CN][CN])
# ============================================================================

class EncodingUtils:
    """ encoding[CN][CN][CN] - [CN][CN][CN][CN][CN] encoding mapping[CN][CN]"""
    
    # [CN][CN]global mapping configuration
    # COMPLETE_CELL_TYPE_MAPPING [CN][CN]global[CN][CN]
    ORIENTATION_MAPPING = FeatureConfig.ORIENTATION_MAPPING
    DIRECTION_MAPPING = FeatureConfig.PIN_DIRECTION_MAPPING
    PIN_TYPE_MAPPING = FeatureConfig.PIN_TYPE_MAPPING
    LAYER_MAPPING = FeatureConfig.LAYER_MAPPING
    NET_TYPE_MAPPING = FeatureConfig.NET_TYPE_MAPPING
    
    @classmethod
    def encode_cell_type(cls, cell_type: str) -> int:
        """[CN][CN][CN] celltype encoding[CN][CN][CN] - [CN][CN]global mapping[CN]"""
        # [CN][CN][CN][CN]global mapping[CN][CN][CN] encoding
        if cell_type in COMPLETE_CELL_TYPE_MAPPING:
            return COMPLETE_CELL_TYPE_MAPPING[cell_type]
        else:
            return COMPLETE_CELL_TYPE_MAPPING['UNKNOWN']
    
    @classmethod
    def encode_orientation(cls, orientation: str) -> int:
        """[CN][CN][CN] encoding[CN][CN][CN]"""
        return cls.ORIENTATION_MAPPING.get(orientation, 0)
    
    @classmethod
    def encode_pin_direction(cls, direction: str) -> int:
        """[CN]pin[CN][CN] encoding[CN][CN][CN]"""
        return cls.DIRECTION_MAPPING.get(direction, 2)  #  default[CN]INOUT
    
    @classmethod
    def encode_pin_type(cls, pin_name: str) -> int:
        """[CN]pintype encoding[CN][CN][CN] -  based onpin[CN][CN][CN][CN]"""
        pin_name_upper = pin_name.upper()
        
        # [CN][CN][CN][CN][CN][CN]pin[CN][CN]
        if pin_name_upper in cls.PIN_TYPE_MAPPING:
            return cls.PIN_TYPE_MAPPING[pin_name_upper]
        
        # [CN][CN][CN][CN] - [CN][CN][CN][CN][CN][CN]
        for pattern, type_id in cls.PIN_TYPE_MAPPING.items():
            if pin_name_upper.startswith(pattern):
                return type_id
        
        return cls.PIN_TYPE_MAPPING['UNKNOWN']
    
    @classmethod
    def encode_layer(cls, layer: str) -> int:
        """[CN][CN] encoding[CN][CN][CN]"""
        return cls.LAYER_MAPPING.get(layer, cls.LAYER_MAPPING['unknown'])
    
    @classmethod
    def encode_net_type(cls, net_name: str) -> int:
        """[CN][CN]net[CN][CN][CN][CN]nettype[CN] encoding"""
        net_name_lower = net_name.lower()
        
        # [CN][CN]net
        if any(keyword in net_name_lower for keyword in ['vdd', 'vcc', 'vpwr', 'power']):
            return cls.NET_TYPE_MAPPING['power']
        
        # [CN]net
        if any(keyword in net_name_lower for keyword in ['vss', 'gnd', 'vgnd', 'ground']):
            return cls.NET_TYPE_MAPPING['ground']
        
        # [CN][CN]net
        if any(keyword in net_name_lower for keyword in ['clk', 'clock', 'ck']):
            return cls.NET_TYPE_MAPPING['clock']
        
        # [CN][CN]net
        if any(keyword in net_name_lower for keyword in ['rst', 'reset', 'rn']):
            return cls.NET_TYPE_MAPPING['reset']
        
        # [CN][CN]net
        if any(keyword in net_name_lower for keyword in ['scan', 'se', 'si', 'so']):
            return cls.NET_TYPE_MAPPING['scan']
        
        #  default[CN][CN][CN]net
        return cls.NET_TYPE_MAPPING['signal']
    
    @classmethod
    def encode_placement_status(cls, status: str) -> int:
        """[CN][CN][CN][CN][CN] encoding[CN][CN][CN]"""
        status_mapping = {
            'PLACED': 0, 'FIXED': 1, 'COVER': 2, 'UNPLACED': 3
        }
        return status_mapping.get(status, 0)

# ============================================================================
# FEATURE ENGINEERING SECTION - feature[CN][CN]region([CN][CN]D[CN][CN][CN][CN]feature[CN][CN])
# ============================================================================

class FeatureEngineering:
    """feature[CN][CN][CN] - [CN][CN][CN][CN]featurecalculation function"""
    
    @staticmethod
    def calculate_component_size(cell_type: str, def_data: Dict) -> float:
        """ calculate component[CN][CN] area[CN][CN] -  based on[CN][CN][CN] actual area data
        
        [CN][CN][CN][CN]:
        - [CN][CN]NangateOpenCellLibrary_typical.lib[CN][CN][CN] area data
        -  convert[CN]DBU²[CN][CN],[CN][CN][CN][CN][CN][CN][CN]
        - [CN][CN] cell[CN] actual[CN][CN] area,[CN] area[CN][CN][CN][CN][CN][CN][CN][CN]
        
        Args:
            cell_type:  celltype[CN][CN][CN]([CN]'INV_X1', 'NAND2_X4'[CN])
            def_data: DEF data dictionary, used for[CN][CN]DBU convert[CN][CN]
            
        Returns:
            DBU²[CN][CN][CN] area[CN][CN],unified with coordinate units
        """
        # DBU convert[CN][CN]:[CN]DEF[CN][CN]UNITS DISTANCE MICRONS[CN][CN][CN][CN]
        # [CN][CN][CN]def_data[CN][CN][CN],[CN][CN][CN][CN] default[CN]
        if def_data and 'units' in def_data:
            DBU_PER_MICRON = float(def_data['units'].get('dbu_per_micron', 2000))
        else:
            DBU_PER_MICRON = 2000  #  default[CN]
        
        # 1 um = DBU_PER_MICRON DBU,[CN][CN] 1 um² = (DBU_PER_MICRON)² DBU²
        AREA_CONVERSION_FACTOR = DBU_PER_MICRON * DBU_PER_MICRON
        
        # 🔬 [CN][CN][CN] actual area data([CN]NangateOpenCellLibrary_typical.lib[CN][CN] extract)
        # [CHART]  data[CN][CN]:NangateOpenCellLibrary_typical.lib[CN]area[CN][CN]
        # 📐 [CN][CN]:um²([CN][CN][CN][CN])
        # [OK] [CN][CN][CN][CN]:[CN][CN][CN][CN] cell[CN][CN][CN] area, based on[CN][CN][CN][CN][CN]
        actual_cell_areas = {
            # [CN][CN][CN][CN] gate[CN][CN] -  actual area(um²)
            'INV_X1': 0.532,        # [CN][CN][CN][CN][CN] area
            'INV_X2': 0.798,        # 2[CN] drive strength[CN][CN][CN]
            'INV_X4': 1.197,        # 4[CN] drive strength[CN][CN][CN]
            'INV_X8': 1.995,        # 8[CN] drive strength[CN][CN][CN]
            'INV_X16': 3.990,       # 16[CN] drive strength[CN][CN][CN]
            'INV_X32': 7.980,       # 32[CN] drive strength[CN][CN][CN]
            
            # NAND gate[CN][CN] -  actual area(um²)
            'NAND2_X1': 0.798,      # 2 inputNAND gate
            'NAND2_X2': 1.064,      # 2 inputNAND gate,2[CN] drive
            'NAND2_X4': 1.596,      # 2 inputNAND gate,4[CN] drive
            'NAND3_X1': 1.064,      # 3 inputNAND gate
            'NAND3_X2': 1.330,      # 3 inputNAND gate,2[CN] drive
            'NAND3_X4': 1.995,      # 3 inputNAND gate,4[CN] drive
            'NAND4_X1': 1.330,      # 4 inputNAND gate
            'NAND4_X2': 1.729,      # 4 inputNAND gate,2[CN] drive
            'NAND4_X4': 2.594,      # 4 inputNAND gate,4[CN] drive
            
            # NOR gate[CN][CN] -  actual area(um²)
            'NOR2_X1': 0.798,       # 2 inputNOR gate
            'NOR2_X2': 1.064,       # 2 inputNOR gate,2[CN] drive
            'NOR2_X4': 1.596,       # 2 inputNOR gate,4[CN] drive
            'NOR3_X1': 1.064,       # 3 inputNOR gate
            'NOR3_X2': 1.330,       # 3 inputNOR gate,2[CN] drive
            'NOR3_X4': 1.995,       # 3 inputNOR gate,4[CN] drive
            'NOR4_X1': 1.330,       # 4 inputNOR gate
            'NOR4_X2': 1.729,       # 4 inputNOR gate,2[CN] drive
            'NOR4_X4': 2.594,       # 4 inputNOR gate,4[CN] drive
            
            # AND gate[CN][CN] -  actual area(um²)
            'AND2_X1': 1.064,       # 2 inputAND gate
            'AND2_X2': 1.330,       # 2 inputAND gate,2[CN] drive
            'AND2_X4': 1.995,       # 2 inputAND gate,4[CN] drive
            'AND3_X1': 1.330,       # 3 inputAND gate
            'AND3_X2': 1.729,       # 3 inputAND gate,2[CN] drive
            'AND3_X4': 2.594,       # 3 inputAND gate,4[CN] drive
            'AND4_X1': 1.596,       # 4 inputAND gate
            'AND4_X2': 1.995,       # 4 inputAND gate,2[CN] drive
            'AND4_X4': 2.992,       # 4 inputAND gate,4[CN] drive
            
            # OR gate[CN][CN] -  actual area(um²)
            'OR2_X1': 1.064,        # 2 inputOR gate
            'OR2_X2': 1.330,        # 2 inputOR gate,2[CN] drive
            'OR2_X4': 1.995,        # 2 inputOR gate,4[CN] drive
            'OR3_X1': 1.330,        # 3 inputOR gate
            'OR3_X2': 1.729,        # 3 inputOR gate,2[CN] drive
            'OR3_X4': 2.594,        # 3 inputOR gate,4[CN] drive
            'OR4_X1': 1.596,        # 4 inputOR gate
            'OR4_X2': 1.995,        # 4 inputOR gate,2[CN] drive
            'OR4_X4': 2.992,        # 4 inputOR gate,4[CN] drive
            
            # [CN][CN] gate[CN][CN] -  actual area(um²)
            'XOR2_X1': 1.596,       # 2 inputXOR gate
            'XOR2_X2': 2.128,       # 2 inputXOR gate,2[CN] drive
            'XNOR2_X1': 1.596,      # 2 inputXNOR gate
            'XNOR2_X2': 2.128,      # 2 inputXNOR gate,2[CN] drive
            
            # AOI/OAI[CN][CN] gate[CN][CN] -  actual area(um²)
            'AOI21_X1': 1.064,      # AOI21 gate
            'AOI21_X2': 1.330,      # AOI21 gate,2[CN] drive
            'AOI21_X4': 1.995,      # AOI21 gate,4[CN] drive
            'AOI22_X1': 1.330,      # AOI22 gate
            'AOI22_X2': 1.729,      # AOI22 gate,2[CN] drive
            'AOI22_X4': 2.594,      # AOI22 gate,4[CN] drive
            'OAI21_X1': 1.064,      # OAI21 gate
            'OAI21_X2': 1.330,      # OAI21 gate,2[CN] drive
            'OAI21_X4': 1.995,      # OAI21 gate,4[CN] drive
            'OAI22_X1': 1.330,      # OAI22 gate
            'OAI22_X2': 1.729,      # OAI22 gate,2[CN] drive
            'OAI22_X4': 2.594,      # OAI22 gate,4[CN] drive
            
            # [CN][CN][CN][CN][CN] -  actual area(um²)
            'BUF_X1': 0.798,        # [CN][CN][CN]
            'BUF_X2': 1.064,        # [CN][CN][CN],2[CN] drive
            'BUF_X4': 1.596,        # [CN][CN][CN],4[CN] drive
            'BUF_X8': 2.660,        # [CN][CN][CN],8[CN] drive
            'BUF_X16': 4.788,       # [CN][CN][CN],16[CN] drive
            'BUF_X32': 9.044,       # [CN][CN][CN],32[CN] drive
            
            # multiplexer[CN][CN] -  actual area(um²)
            'MUX2_X1': 2.660,       # 2[CN]1multiplexer
            'MUX2_X2': 3.458,       # 2[CN]1multiplexer,2[CN] drive
            
            # [CN][CN][CN][CN][CN] -  actual area(um²)
            'DFF_X1': 4.522,        # D[CN][CN][CN]
            'DFF_X2': 5.852,        # D[CN][CN][CN],2[CN] drive
            'DFFR_X1': 5.320,       # [CN][CN][CN]D[CN][CN][CN]
            'DFFR_X2': 6.650,       # [CN][CN][CN]D[CN][CN][CN],2[CN] drive
            'DFFS_X1': 5.320,       # [CN][CN][CN]D[CN][CN][CN]
            'DFFS_X2': 6.650,       # [CN][CN][CN]D[CN][CN][CN],2[CN] drive
            'DFFRS_X1': 6.118,      # [CN][CN] position[CN]D[CN][CN][CN]
            'DFFRS_X2': 7.448,      # [CN][CN] position[CN]D[CN][CN][CN],2[CN] drive
            
            # [CN][CN][CN][CN][CN] -  actual area(um²)
            'DLH_X1': 2.394,        # D[CN][CN][CN]
            'DLH_X2': 3.192,        # D[CN][CN][CN],2[CN] drive
            'DLHR_X1': 2.926,       # [CN][CN][CN]D[CN][CN][CN]
            'DLHR_X2': 3.724,       # [CN][CN][CN]D[CN][CN][CN],2[CN] drive
            'DLHS_X1': 2.926,       # [CN][CN][CN]D[CN][CN][CN]
            
            # [CN][CN] cell -  actual area(um²)
            'TAPCELL_X1': 0.266,    # substrate contact cell
            'FILLCELL_X1': 0.266,   # [CN][CN] cell
            'FILLCELL_X2': 0.532,   # [CN][CN] cell,2[CN][CN][CN]
            'FILLCELL_X4': 1.064,   # [CN][CN] cell,4[CN][CN][CN]
            'FILLCELL_X8': 2.128,   # [CN][CN] cell,8[CN][CN][CN]
            'FILLCELL_X16': 4.256,  # [CN][CN] cell,16[CN][CN][CN]
            'FILLCELL_X32': 8.512,  # [CN][CN] cell,32[CN][CN][CN]
            
            # [CN][CN][CN][CN] -  actual area(um²)
            'LOGIC0_X1': 0.532,     # [CN][CN]0[CN][CN]
            'LOGIC1_X1': 0.532,     # [CN][CN]1[CN][CN]
            
            # [CN][CN][CN][CN] cell -  actual area(um²)
            'ANTENNA_X1': 0.266,    # [CN][CN][CN][CN] cell
        }
        
        normalized_cell_type = cell_type.upper()
        actual_area_um2 = actual_cell_areas.get(normalized_cell_type, 0.532)  #  defaultINV_X1 area
        
        #  convert[CN]DBU²[CN][CN],unified with coordinate units
        area_dbu2 = actual_area_um2 * AREA_CONVERSION_FACTOR
        return area_dbu2
    
    @staticmethod
    def calculate_cell_power(cell_type: str) -> float:
        """[CN][CN] celltype[CN][CN][CN][CN] powerfeature -  based onNangateOpenCellLibrary_typical.lib
        
        [CN][CN][CN][CN]:
        - [CN][CN][CN][CN][CN][CN][CN]cell_leakage_power[CN][CN][CN][CN][CN] data
        - [CN][CN]:[CN][CN][CN][CN] consistent[CN][CN][CN][CN][CN]([CN][CN][CN][CN] power)
        - [CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN],[CN] power[CN][CN][CN][CN][CN][CN][CN][CN]
        
        Args:
            cell_type:  celltype[CN][CN][CN]([CN]'INV_X1', 'NAND2_X4'[CN])
            
        Returns:
             cell[CN][CN][CN][CN][CN] power[CN]
        """
        # 🔬 [CN][CN][CN][CN][CN] power data([CN]NangateOpenCellLibrary_typical.lib extract)
        # [CHART]  data[CN][CN]:NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power[CN][CN]
        # ⚡ [CN][CN]:[CN][CN][CN][CN] consistent[CN][CN][CN][CN][CN]// cell[CN][CN][CN][CN][CN][CN][CN] power[CN][CN],[CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN]
        real_cell_powers = {
            # [CN][CN][CN][CN][CN] - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'INV_X1': 14.353185,    # [CN][CN][CN],1[CN] drive
            'INV_X2': 28.706376,    # [CN][CN][CN],2[CN] drive
            'INV_X4': 57.412850,    # [CN][CN][CN],4[CN] drive
            'INV_X8': 114.826305,   # [CN][CN][CN],8[CN] drive
            'INV_X16': 229.651455,  # [CN][CN][CN],16[CN] drive
            'INV_X32': 459.302800,  # [CN][CN][CN],32[CN] drive
            
            # [CN][CN][CN][CN][CN] - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'BUF_X1': 21.438247,    # [CN][CN][CN],1[CN] drive
            'BUF_X2': 43.060820,    # [CN][CN][CN],2[CN] drive
            'BUF_X4': 86.121805,    # [CN][CN][CN],4[CN] drive
            'BUF_X8': 172.244545,   # [CN][CN][CN],8[CN] drive
            'BUF_X16': 344.488100,  # [CN][CN][CN],16[CN] drive
            'BUF_X32': 688.976200,  # [CN][CN][CN],32[CN] drive
            
            # [CN][CN] gate[CN][CN] - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'NAND2_X1': 17.393360,  # 2 input[CN][CN] gate,1[CN] drive
            'NAND2_X2': 34.786630,  # 2 input[CN][CN] gate,2[CN] drive
            'NAND2_X4': 69.573240,  # 2 input[CN][CN] gate,4[CN] drive
            'NAND3_X1': 18.104768,  # 3 input[CN][CN] gate,1[CN] drive
            'NAND3_X2': 36.209558,  # 3 input[CN][CN] gate,2[CN] drive
            'NAND3_X4': 72.419123,  # 3 input[CN][CN] gate,4[CN] drive
            'NAND4_X1': 18.126843,  # 4 input[CN][CN] gate,1[CN] drive
            'NAND4_X2': 36.253723,  # 4 input[CN][CN] gate,2[CN] drive
            'NAND4_X4': 72.506878,  # 4 input[CN][CN] gate,4[CN] drive
            
            # [CN][CN] gate[CN][CN] - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'NOR2_X1': 21.199545,   # 2 input[CN][CN] gate,1[CN] drive
            'NOR2_X2': 42.399074,   # 2 input[CN][CN] gate,2[CN] drive
            'NOR2_X4': 84.798143,   # 2 input[CN][CN] gate,4[CN] drive
            'NOR3_X1': 26.831667,   # 3 input[CN][CN] gate,1[CN] drive
            'NOR3_X2': 53.663264,   # 3 input[CN][CN] gate,2[CN] drive
            'NOR3_X4': 107.325918,  # 3 input[CN][CN] gate,4[CN] drive
            'NOR4_X1': 32.601474,   # 4 input[CN][CN] gate,1[CN] drive
            'NOR4_X2': 65.202889,   # 4 input[CN][CN] gate,2[CN] drive
            'NOR4_X4': 130.405147,  # 4 input[CN][CN] gate,4[CN] drive
            
            # [CN] gate[CN][CN] - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'AND2_X1': 25.066064,   # 2 input[CN] gate,1[CN] drive
            'AND2_X2': 50.353160,   # 2 input[CN] gate,2[CN] drive
            'AND2_X4': 100.706457,  # 2 input[CN] gate,4[CN] drive
            'AND3_X1': 26.481460,   # 3 input[CN] gate,1[CN] drive
            'AND3_X2': 53.190270,   # 3 input[CN] gate,2[CN] drive
            'AND3_X4': 106.380663,  # 3 input[CN] gate,4[CN] drive
            'AND4_X1': 27.024804,   # 4 input[CN] gate,1[CN] drive
            'AND4_X2': 54.274743,   # 4 input[CN] gate,2[CN] drive
            'AND4_X4': 108.549590,  # 4 input[CN] gate,4[CN] drive
            
            # OR gate[CN][CN] - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'OR2_X1': 22.694975,    # 2 input[CN] gate,1[CN] drive
            'OR2_X2': 45.656022,    # 2 input[CN] gate,2[CN] drive
            'OR2_X4': 91.312375,    # 2 input[CN] gate,4[CN] drive
            'OR3_X1': 24.414625,    # 3 input[CN] gate,1[CN] drive
            'OR3_X2': 49.162437,    # 3 input[CN] gate,2[CN] drive
            'OR3_X4': 98.325150,    # 3 input[CN] gate,4[CN] drive
            'OR4_X1': 26.733490,    # 4 input[CN] gate,1[CN] drive
            'OR4_X2': 53.869509,    # 4 input[CN] gate,2[CN] drive
            'OR4_X4': 107.739253,   # 4 input[CN] gate,4[CN] drive
            
            # [CN][CN] gate[CN][CN] - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'XOR2_X1': 36.163718,   # 2 input[CN][CN] gate,1[CN] drive
            'XOR2_X2': 72.593483,   # 2 input[CN][CN] gate,2[CN] drive
            'XNOR2_X1': 36.441009,  # 2 input[CN][CN][CN] gate,1[CN] drive
            'XNOR2_X2': 73.102975,  # 2 input[CN][CN][CN] gate,2[CN] drive
            
            # AOI/OAI[CN][CN] gate[CN][CN] - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'AOI21_X1': 27.858395,  # AOI21 gate,1[CN] drive
            'AOI21_X2': 55.716720,  # AOI21 gate,2[CN] drive
            'AOI21_X4': 111.433338, # AOI21 gate,4[CN] drive
            'AOI22_X1': 32.611944,  # AOI22 gate,1[CN] drive
            'AOI22_X2': 65.223838,  # AOI22 gate,2[CN] drive
            'AOI22_X4': 130.447551, # AOI22 gate,4[CN] drive
            'OAI21_X1': 22.619394,  # OAI21 gate,1[CN] drive
            'OAI21_X2': 45.238687,  # OAI21 gate,2[CN] drive
            'OAI21_X4': 90.477187,  # OAI21 gate,4[CN] drive
            'OAI22_X1': 34.026125,  # OAI22 gate,1[CN] drive
            'OAI22_X2': 68.052131,  # OAI22 gate,2[CN] drive
            'OAI22_X4': 136.103946, # OAI22 gate,4[CN] drive
            
            # multiplexer[CN][CN] - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'MUX2_X1': 61.229735,   # 2[CN]1multiplexer,1[CN] drive
            'MUX2_X2': 68.648566,   # 2[CN]1multiplexer,2[CN] drive
            
            # [CN][CN][CN][CN][CN] - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'DFF_X1': 100.684799,   # D[CN][CN][CN],1[CN] drive
            'DFF_X2': 136.676074,   # D[CN][CN][CN],2[CN] drive
            'DFFR_X1': 105.258197,  # [CN][CN][CN]D[CN][CN][CN],1[CN] drive
            'DFFR_X2': 141.961514,  # [CN][CN][CN]D[CN][CN][CN],2[CN] drive
            'DFFS_X1': 107.724855,  # [CN][CN][CN]D[CN][CN][CN],1[CN] drive
            'DFFS_X2': 140.592991,  # [CN][CN][CN]D[CN][CN][CN],2[CN] drive
            'DFFRS_X1': 100.161505, # [CN][CN] position[CN]D[CN][CN][CN],1[CN] drive
            'DFFRS_X2': 142.302832, # [CN][CN] position[CN]D[CN][CN][CN],2[CN] drive
            
            # [CN][CN][CN][CN][CN] - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'DLH_X1': 40.863240,    # D[CN][CN][CN],1[CN] drive
            'DLH_X2': 57.430452,    # D[CN][CN][CN],2[CN] drive
            'DLHR_X1': 40.863416,   # [CN][CN][CN]D[CN][CN][CN],1[CN] drive
            'DLHR_X2': 57.430445,   # [CN][CN][CN]D[CN][CN][CN],2[CN] drive
            'DLHS_X1': 75.762253,   # [CN][CN][CN]D[CN][CN][CN],1[CN] drive
            
            # [CN][CN] cell[CN][CN] -  based on[CN][CN][CN][CN][CN]([CN][CN] cell[CN][CN] power[CN][CN])
            'FILLCELL_X1': 0.000000, # [CN][CN] cell,1[CN](ANTENNA_X1[CN][CN])
            'FILLCELL_X2': 0.000000, # [CN][CN] cell,2[CN]
            'FILLCELL_X4': 0.000000, # [CN][CN] cell,4[CN]
            'FILLCELL_X8': 0.000000, # [CN][CN] cell,8[CN]
            'FILLCELL_X16': 0.000000,# [CN][CN] cell,16[CN]
            'FILLCELL_X32': 0.000000,# [CN][CN] cell,32[CN]
            
            # [CN][CN][CN][CN] cell - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'LOGIC0_X1': 35.928390, # [CN][CN]0 cell
            'LOGIC1_X1': 17.885841, # [CN][CN]1 cell
            
            # [CN][CN][CN][CN] cell - [CN][CN][CN][CN][CN]NangateOpenCellLibrary_typical.lib[CN]cell_leakage_power
            'ANTENNA_X1': 0.000000, # [CN][CN][CN][CN] cell
        }
        
        normalized_cell_type = cell_type.upper()
        return real_cell_powers.get(normalized_cell_type, 14.353185)  #  defaultINV_X1 power

# ============================================================================
# F GRAPH GENERATOR SECTION - F[CN]generate[CN]region
# ============================================================================

class FHeterographGenerator:
    """F[CN][CN][CN][CN]generate[CN]
    
     based onDEFfile generationF[CN][CN][CN][CN], contains:
    1. Gatenode:[CN][CN][CN]gate component
    2. IO_Pinnode:IOpinnode
    3. Gate-Gateedge:6[CN]feature [pin_type_id, cell_type_id] + [net_type_id, connection_count] + [pin_type_id, cell_type_id]
    4. IO_Pin-Gateedge:4[CN]feature [net_type_id, connection_count] + [pin_type_id, cell_type_id]
    """
    
    # ==================== [CN][CN]type mapping[CN][CN] ====================
    # [CN]node_edge_analysis_report.mdkeep consistent[CN] mapping[CN][CN]
    
    @property
    def node_type_to_id(self) -> Dict[str, int]:
        """nodetype[CN]ID[CN] mapping - [CN][CN][CN][CN]"""
        return {
            "gate": 0,
            "io_pin": 1, 
            "net": 2,
            "pin": 3
        }
    
    @property  
    def edge_type_to_id(self) -> Dict[str, int]:
        """edgetype[CN]ID[CN] mapping - [CN][CN][CN][CN]"""
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
        
        # statisticsinformation
        self.stats = {
            'total_gates': 0,
            'total_io_pins': 0,
            'total_gate_gate_edges': 0,
            'total_io_pin_gate_edges': 0,
            'nets_processed': 0
        }
    
    def generate(self) -> HeteroData:
        """generateF[CN][CN][CN][CN]"""
        print("[START] [CN][CN]generateF[CN][CN][CN][CN]...")
        
        try:
            # 1.  buildgate-gateedge[CN]IO_Pin-gateedge
            self._build_edges()
            
            # 2.  create[CN][CN][CN] data structure
            hetero_data = self._create_hetero_data()
            
            # 3.  outputstatisticsinformation
            self._print_statistics()
            
            return hetero_data
            
        except Exception as e:
            print(f"[X] F[CN]generatefailed: {e}")
            traceback.print_exc()
            raise
    
    def _build_edges(self):
        """ buildgate-gateedge[CN]IO_Pin-gateedge"""
        print("[CHART]  buildedge connection...")
        nets = self.def_data.get('nets', {})
        components = self.def_data.get('components', {})
        pins = self.def_data.get('pins', {})
        
        print(f"[CHART]  datastatistics: {len(nets)}net, {len(components)} component, {len(pins)}pin")
        
        #  addPIN connectionstatistics
        pin_networks_count = 0
        pin_connections_count = 0
        single_pin_networks = 0
        multi_pin_networks = 0
        
        for net_name, net_info in nets.items():
            self.stats['nets_processed'] += 1
            connections = net_info.get('connections', [])
            
            # [CN][CN][CN][CN] containsPIN connection
            has_pin = any(comp_name == 'PIN' for comp_name, pin_name in connections)
            if has_pin:
                pin_networks_count += 1
                pin_connections_count += sum(1 for comp_name, pin_name in connections if comp_name == 'PIN')
                
                if len(connections) < 2:
                    single_pin_networks += 1
                    if single_pin_networks <= 5:
                        print(f"  [CN] connectionPINnet {net_name}: {connections}")
                else:
                    multi_pin_networks += 1
                    if multi_pin_networks <= 5:
                        print(f"  [CN] connectionPINnet {net_name}: {len(connections)} connection")

            if len(connections) < 2:
                continue  #  skip[CN][CN][CN] connection[CN]net

            # [CN][CN]IOpin[CN]gatepin
            io_pins = []
            gate_pins = []

            for comp_name, pin_name in connections:
                # [CN][CN] output:[CN][CN] connectioninformation
                if has_pin and self.stats['nets_processed'] <= 10:  # [CN][CN][CN] containsPIN[CN][CN]10net
                    print(f"  net {net_name}:  connection ({comp_name}, {pin_name})")
                
                if comp_name == 'PIN':
                    # [CN][CN]IOpin
                    io_pins.append((comp_name, pin_name))
                    if has_pin and self.stats['nets_processed'] <= 10:
                        print(f"    -> identified asIOpin: {pin_name}")
                elif comp_name in components:
                    # [CN][CN]gatepin
                    gate_pins.append((comp_name, pin_name))
                    if has_pin and self.stats['nets_processed'] <= 10:
                        print(f"    -> identified asGatepin: {comp_name}.{pin_name}")
                else:
                    # [CN][CN][CN][CN] connectiontype
                    if has_pin and self.stats['nets_processed'] <= 10:
                        print(f"    -> [CN][CN][CN][CN] connection: {comp_name}.{pin_name} ([CN][CN]components dictionary[CN])")
                        # [CN][CN][CN][CN][CN][CN][CN][CN] component[CN]
                        if comp_name.startswith('output') or comp_name.startswith('input'):
                            print(f"       [CN][CN][CN][CN]IO[CN][CN][CN] component: {comp_name}")
            
            # [CN][CN][CN]PIN connection,[CN][CN][CN][CN][CN][CN]
            if has_pin and self.stats['nets_processed'] <= 10:
                print(f"    [CN][CN][CN][CN]: IOpin={len(io_pins)}, Gatepin={len(gate_pins)}")
            
            #  buildgate-gateedge
            self._build_gate_gate_edges(net_name, net_info, gate_pins, components)
            
            #  buildIO_Pin-gateedge
            self._build_io_pin_gate_edges(net_name, net_info, io_pins, gate_pins, components)
        
        print(f"[CHART] PIN connectionstatistics: {pin_networks_count}net containsPIN, total{pin_connections_count}PIN connection")
        print(f"   - [CN] connectionPINnet: {single_pin_networks} ([CN][CN]PIN,[CN]gate connection)")
        print(f"   - [CN] connectionPINnet: {multi_pin_networks} (PIN connection[CN]gate)")
    
    def _build_gate_gate_edges(self, net_name: str, net_info: Dict, gate_pins: List[Tuple[str, str]], components: Dict):
        """ buildgate-gateedge"""
        if len(gate_pins) < 2:
            return
        
        # [CN][CN] outputpin[CN] inputpin
        output_pins = []
        input_pins = []
        
        for comp_name, pin_name in gate_pins:
            pin_type = self._infer_pin_type(pin_name)
            if pin_type == 'OUTPUT':
                output_pins.append((comp_name, pin_name))
            else:
                input_pins.append((comp_name, pin_name))
        
        # [CN][CN][CN][CN][CN][CN][CN] outputpin,[CN][CN][CN][CN]pin[CN][CN] output
        if not output_pins and gate_pins:
            output_pins = [gate_pins[0]]
            input_pins = gate_pins[1:]
        
        # [CN][CN] outputpin connection[CN][CN][CN] inputpin
        for source_comp, source_pin in output_pins:
            for target_comp, target_pin in input_pins:
                if source_comp == target_comp:
                    continue  #  skip[CN] connection
                
                #  creategate-gateedge
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
        """ buildIO_Pin-gateedge
        
        [CN][CN][CN][CN]:IOPin -> net -> gate [CN][CN][CN][CN]entriesedge,[CN][CN]gate[CN][CN][CN]pin connection[CN][CN]net
        """
        if not io_pins or not gate_pins:
            if io_pins and not gate_pins:
                # [CN][CN]:[CN][CN][CN][CN]IOpin[CN][CN]gatepin[CN][CN][CN]
                if self.stats.get('debug_io_only_count', 0) < 5:
                    print(f"    [CN][CN]: net {net_name} [CN][CN]IOpin,[CN]gatepin: io_pins={io_pins}")
                    self.stats['debug_io_only_count'] = self.stats.get('debug_io_only_count', 0) + 1
            return
        
        # [CN][CN]net[CN][CN][CN][CN]gate list([CN][CN])
        unique_gates = list(set(gate_comp for gate_comp, gate_pin in gate_pins))
        
        # [CN][CN]:[CN][CN]success[CN]IO_Pin-Gateedge build
        if self.stats.get('debug_io_gate_count', 0) < 10:
            print(f"    [CN][CN]:  buildIO_Pin-Gateedge - net {net_name}: {len(io_pins)}IOpin -> {len(unique_gates)}[CN][CN]gate")
            # [CN][CN]gate[CN]pin[CN][CN]statistics
            gate_pin_count = {}
            for gate_comp, gate_pin in gate_pins:
                gate_pin_count[gate_comp] = gate_pin_count.get(gate_comp, 0) + 1
            multi_pin_gates = {gate: count for gate, count in gate_pin_count.items() if count > 1}
            if multi_pin_gates:
                print(f"      [CN]pin[CN]gate: {multi_pin_gates}")
            self.stats['debug_io_gate_count'] = self.stats.get('debug_io_gate_count', 0) + 1
        
        # [CN][CN]IOpin connection[CN][CN][CN][CN][CN]gate([CN]gate[CN] create[CN]entriesedge)
        for io_comp, io_pin in io_pins:
            for gate_comp in unique_gates:
                # [CN][CN][CN]gate[CN][CN]net[CN][CN][CN][CN]pin, used forfeature calculate
                gate_pins_for_this_gate = [pin for comp, pin in gate_pins if comp == gate_comp]
                
                # [CN][CN][CN][CN][CN][CN]pin used forfeature encoding([CN][CN][CN][CN][CN][CN][CN] outputpin)
                representative_pin = self._select_representative_pin(gate_pins_for_this_gate)
                
                #  createIO_Pin-gateedge
                edge = IO_PinGateEdge(
                    io_pin_name=io_pin,
                    gate_name=gate_comp,
                    net_type=EncodingUtils.encode_net_type(net_name),
                    connection_count=len(unique_gates) + len(io_pins),  # use uniquegate[CN][CN]
                    gate_pin_type=EncodingUtils.encode_pin_type(representative_pin),
                    gate_cell_type=EncodingUtils.encode_cell_type(components[gate_comp]['cell_type']),
                    net_name=net_name,
                    gate_pin_count=len(gate_pins_for_this_gate)  # [CN][CN][CN]gate[CN][CN]net[CN][CN]pin[CN][CN]
                )
                
                self.io_pin_gate_edges.append(edge)
                self.stats['total_io_pin_gate_edges'] += 1
    
    def _select_representative_pin(self, gate_pins: List[str]) -> str:
        """[CN]gate[CN][CN]pin[CN][CN][CN][CN][CN][CN][CN]pin used forfeature encoding
        
        [CN][CN][CN]: outputpin > [CN][CN]pin
        """
        if not gate_pins:
            return "UNKNOWN"
        
        # [CN][CN] outputpin
        for pin in gate_pins:
            if self._infer_pin_type(pin) == 'OUTPUT':
                return pin
        
        # [CN][CN][CN][CN][CN][CN][CN] outputpin, return[CN][CN]pin
        return gate_pins[0]
    
    def _infer_pin_type(self, pin_name: str) -> str:
        """[CN][CN]pin[CN][CN][CN][CN]pintype"""
        pin_name_upper = pin_name.upper()
        
        #  outputpin[CN][CN]
        output_patterns = ['Z', 'ZN', 'Q', 'QN', 'Y', 'CO', 'S']
        if any(pin_name_upper.startswith(pattern) for pattern in output_patterns):
            return 'OUTPUT'
        
        #  inputpin[CN][CN]
        input_patterns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'CK', 'CLK', 'CP', 'R', 'RN', 'RST', 'EN', 'S', 'SE', 'SI', 'SN']
        if any(pin_name_upper.startswith(pattern) for pattern in input_patterns):
            return 'INPUT'
        
        return 'UNKNOWN'
    
    def _create_hetero_data(self) -> HeteroData:
        """ create[CN][CN][CN] data structure"""
        print("🏗️  create[CN][CN][CN] data structure...")
        
        hetero_data = HeteroData()
        
        # 1.  creategatenode
        self._create_gate_nodes(hetero_data)
        
        # 2.  createIO_Pinnode
        self._create_io_pin_nodes(hetero_data)
        
        # 3.  createedge - [CN][CN][CN][CN][CN][CN]
        self._create_gate_gate_edges(hetero_data)     # Gate-Gateedge (ID: 0)
        self._create_io_pin_gate_edges(hetero_data)   # IO_Pin-Gateedge (ID: 3)
        
        # 5.  addglobalfeature[CN] label([CN]D[CN]keep consistent)
        self._add_global_features(hetero_data)
        
        return hetero_data
    
    def _create_gate_nodes(self, hetero_data: HeteroData):
        """ creategatenode([CN]D[CN]keep consistent[CN]featureformat[CN][CN][CN])"""
        components = self.def_data.get('components', {})
        
        # [CN][CN][CN][CN][CN][CN]gate component([CN][CN][CN][CN] cell[CN])
        gate_components = {}
        for comp_name, comp_info in components.items():
            cell_type = comp_info['cell_type']
            if (not cell_type.startswith('FILLCELL') and 
                not cell_type.startswith('ANTENNA') and 
                not cell_type.startswith('TAPCELL')):
                gate_components[comp_name] = comp_info
        
        self.stats['total_gates'] = len(gate_components)
        
        if not gate_components:
            print("[!]️ warning:[CN][CN][CN][CN]gate component")
            hetero_data['gate'].x = torch.empty((0, 7), dtype=torch.float)
            return
        
        #  creategate[CN][CN][CN][CN] mapping
        self.gate_to_idx = {name: idx for idx, name in enumerate(gate_components.keys())}
        
        #  buildgatenodefeature([CN]D[CN]keep consistent[CN]feature[CN][CN])
        gate_features = []
        names = []
        for comp_name, comp_info in gate_components.items():
            # Gatenodefeature:[x, y, cell_type_id, orientation_id, area, placement_status, power] - [CN]D[CN]keep consistent
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
        print(f"[OK]  create[CN] {len(gate_components)} gatenode([CN]D[CN]featureformat consistent)")
    
    def _create_io_pin_nodes(self, hetero_data: HeteroData):
        """ createIO_Pinnode([CN]D[CN]keep consistent[CN]featureformat)"""
        pins = self.def_data.get('pins', {})
        self.stats['total_io_pins'] = len(pins)
        
        if not pins:
            print("[!]️ warning:[CN][CN][CN][CN]IOpin")
            hetero_data['io_pin'].x = torch.empty((0, 4), dtype=torch.float)
            return
        
        #  createIO_Pin[CN][CN][CN][CN] mapping
        self.io_pin_to_idx = {name: idx for idx, name in enumerate(pins.keys())}
        
        #  buildIO_Pinnodefeature([CN]D[CN]keep consistent)
        io_pin_features = []
        names = []
        for pin_name, pin_info in pins.items():
            # IO_Pinnodefeature:[x, y, direction_id, layer_id] - [CN]D[CN]keep consistent
            pos = pin_info.get('position', (0, 0))
            direction_id = EncodingUtils.encode_pin_direction(pin_info.get('direction', 'INOUT'))
            layer_id = EncodingUtils.encode_layer(pin_info.get('layer', 'metal1'))
            
            features = [pos[0], pos[1], direction_id, layer_id]
            io_pin_features.append(features)
            names.append(pin_name)
        
        hetero_data['io_pin'].x = torch.tensor(io_pin_features, dtype=torch.float)
        hetero_data['io_pin'].names = names
        print(f"[OK]  create[CN] {len(pins)} IO_Pinnode([CN]D[CN]featureformat consistent)")
    
    def _create_gate_gate_edges(self, hetero_data: HeteroData):
        """ creategate-gateedge( add[CN][CN]D[CN]pin_pinedge[CN]HPWL label)"""
        if not self.gate_gate_edges:
            print("[!]️ warning:[CN][CN]gate-gateedge")
            hetero_data['gate', 'connects_to', 'gate'].edge_index = torch.empty((2, 0), dtype=torch.long)
            hetero_data['gate', 'connects_to', 'gate'].edge_attr = torch.empty((0, 7), dtype=torch.float)
            hetero_data['gate', 'connects_to', 'gate'].edge_label = torch.empty((0, 2), dtype=torch.float)
            return
        
        #  buildedge[CN][CN],edgefeature[CN]edge label
        edge_indices = []
        edge_features = []
        edge_labels = []
        
        for edge in self.gate_gate_edges:
            if edge.source_gate in self.gate_to_idx and edge.target_gate in self.gate_to_idx:
                source_idx = self.gate_to_idx[edge.source_gate]
                target_idx = self.gate_to_idx[edge.target_gate]
                
                edge_indices.append([source_idx, target_idx])
                
                #  calculate[CN]net[CN]HPWL
                hpwl = self._calculate_hpwl(edge.net_name)
                
                # 7[CN]edgefeature:[source_pin_type, source_cell_type, net_type, connection_count, target_pin_type, target_cell_type, hpwl]
                features = [
                    edge.source_pin_type,
                    edge.source_cell_type,
                    edge.net_type,
                    edge.connection_count,
                    edge.target_pin_type,
                    edge.target_cell_type,
                    hpwl
                ]
                edge_features.append(features)
                
                #  calculate[CN]net[CN]routinginformation[CN][CN]edge label([CN][CN]D[CN]pin_pinedge[CN] label[CN][CN])
                wire_length, via_count = self._calculate_net_routing_info(edge.net_name)
                edge_labels.append([wire_length, float(via_count)])
        
        if edge_indices:
            hetero_data['gate', 'connects_to', 'gate'].edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            hetero_data['gate', 'connects_to', 'gate'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
            hetero_data['gate', 'connects_to', 'gate'].edge_label = torch.tensor(edge_labels, dtype=torch.float)
            print(f"[OK]  create[CN] {len(edge_indices)} entriesgate-gateedge( contains[wire_length, via_count] label[CN]HPWLfeature)")
        else:
            hetero_data['gate', 'connects_to', 'gate'].edge_index = torch.empty((2, 0), dtype=torch.long)
            hetero_data['gate', 'connects_to', 'gate'].edge_attr = torch.empty((0, 7), dtype=torch.float)
            hetero_data['gate', 'connects_to', 'gate'].edge_label = torch.empty((0, 2), dtype=torch.float)
    
    def _create_io_pin_gate_edges(self, hetero_data: HeteroData):
        """ createIO_Pin-gateedge"""
        if not self.io_pin_gate_edges:
            print("[!]️ warning:[CN][CN]IO_Pin-gateedge")
            hetero_data['io_pin', 'connects_to', 'gate'].edge_index = torch.empty((2, 0), dtype=torch.long)
            hetero_data['io_pin', 'connects_to', 'gate'].edge_attr = torch.empty((0, 4), dtype=torch.float)
            hetero_data['io_pin', 'connects_to', 'gate'].edge_label = torch.empty((0, 2), dtype=torch.float)
            return
        
        #  buildedge[CN][CN],edgefeature[CN]edge label
        edge_indices = []
        edge_features = []
        edge_labels = []
        
        for edge in self.io_pin_gate_edges:
            if edge.io_pin_name in self.io_pin_to_idx and edge.gate_name in self.gate_to_idx:
                io_pin_idx = self.io_pin_to_idx[edge.io_pin_name]
                gate_idx = self.gate_to_idx[edge.gate_name]
                
                edge_indices.append([io_pin_idx, gate_idx])
                
                #  calculate[CN]net[CN]HPWL[CN][CN]feature
                hpwl = self._calculate_hpwl(edge.net_name)
                
                # 5[CN]edgefeature:[net_type_id, connection_count, pin_type_id, cell_type_id, hpwl]
                features = [
                    edge.net_type,
                    edge.connection_count,
                    edge.gate_pin_type,
                    edge.gate_cell_type,
                    hpwl
                ]
                edge_features.append(features)
                
                #  calculate[CN]net[CN]routinginformation[CN][CN]edge label([CN][CN]D[CN]iopin-pinedge[CN] label[CN][CN])
                wire_length, via_count = self._calculate_net_routing_info(edge.net_name)
                edge_labels.append([wire_length, float(via_count)])
        
        if edge_indices:
            hetero_data['io_pin', 'connects_to', 'gate'].edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            hetero_data['io_pin', 'connects_to', 'gate'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
            hetero_data['io_pin', 'connects_to', 'gate'].edge_label = torch.tensor(edge_labels, dtype=torch.float)
            print(f"[OK]  create[CN] {len(edge_indices)} entriesIO_Pin-gateedge( contains[wire_length, via_count] label[CN]HPWLfeature)")
        else:
            hetero_data['io_pin', 'connects_to', 'gate'].edge_index = torch.empty((2, 0), dtype=torch.long)
            hetero_data['io_pin', 'connects_to', 'gate'].edge_attr = torch.empty((0, 5), dtype=torch.float)
            hetero_data['io_pin', 'connects_to', 'gate'].edge_label = torch.empty((0, 2), dtype=torch.float)
    
    def _add_global_features(self, hetero_data: HeteroData):
        """ addglobalfeature[CN] label([CN]D[CN][CN][CN] consistent)"""
        print("[GLOBE]  addglobalfeature[CN] label...")
        
        # [CN][CN][CN][CN][CN][CN]information
        die_area = self.def_data.get('die_area', (0, 0, 100000, 100000))
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
        hetero_data.global_features = global_features
        
        # [CN][CN][CN][CN]information - 2x2[CN][CN]:[[[CN][CN][CN]x,[CN][CN][CN]y], [[CN][CN][CN]x,[CN][CN][CN]y]]
        die_coordinates = torch.tensor([[die_area[0], die_area[1]], [die_area[2], die_area[3]]], dtype=torch.float)
        hetero_data.die_coordinates = die_coordinates
        
        #  readrouting report[CN][CN]global label
        utilization, hpwl, total_wire_length, total_vias = self._read_routing_labels()
        
        #  calculatePin Density ([CN][CN]pin[CN][CN][CN])
        internal_pin_count = len(self.def_data.get('internal_pins', {}))
        total_pin_count = internal_pin_count + len(self.def_data.get('pins', {}))
        pin_density = (internal_pin_count / total_pin_count * 100.0) if total_pin_count > 0 else 0.0
        
        # global label: [ actual[CN][CN][CN], HPWL, Pin Density, Total Wire Length, Total Vias]
        y = torch.tensor([utilization, hpwl, pin_density, total_wire_length, total_vias], dtype=torch.float)
        hetero_data.y = y
        
        print(f"  ✓ globalfeature: 5[CN] ([CN][CN]={chip_width}, [CN][CN]={chip_height},  area={chip_area}, DBU={dbu_per_micron},  configuration[CN][CN][CN]={core_utilization_config})")
        print(f"  ✓ global label: 5[CN] ( actual[CN][CN][CN]={utilization}%, HPWL={hpwl}um, Pin[CN][CN]={pin_density:.2f}%, [CN][CN][CN]={total_wire_length}, [CN][CN][CN]={total_vias})")
    
    def _calculate_total_hpwl(self) -> float:
        """ calculate[CN][CN][CN][CN][CN][CN][CN](HPWL)"""
        nets = self.def_data.get('nets', {})
        components = self.def_data.get('components', {})
        pins = self.def_data.get('pins', {})
        total_hpwl = 0.0
        
        for net_name, net_info in nets.items():
            connections = net_info.get('connections', [])
            if len(connections) < 2:
                continue
            
            # [CN][CN][CN][CN] connection[CN][CN][CN][CN]
            x_coords = []
            y_coords = []
            
            for comp_name, pin_name in connections:
                if comp_name == 'PIN' and pin_name in pins:
                    # IOpin
                    pos = pins[pin_name].get('position', (0, 0))
                    x_coords.append(pos[0])
                    y_coords.append(pos[1])
                elif comp_name in components:
                    #  componentpin
                    pos = components[comp_name].get('position', (0, 0))
                    x_coords.append(pos[0])
                    y_coords.append(pos[1])
            
            if len(x_coords) >= 2:
                #  calculateHPWL = (max_x - min_x) + (max_y - min_y)
                hpwl = (max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords))
                total_hpwl += hpwl
        
        return total_hpwl
    
    def _calculate_hpwl(self, net_name: str) -> float:
        """ calculate[CN]net[CN]HPWL( used forgate_gateedge label)"""
        nets = self.def_data.get('nets', {})
        components = self.def_data.get('components', {})
        pins = self.def_data.get('pins', {})
        
        if net_name not in nets:
            return 0.0
            
        net_info = nets[net_name]
        connections = net_info.get('connections', [])
        if len(connections) < 2:
            return 0.0
            
        # [CN][CN][CN][CN] connection[CN][CN][CN][CN]
        x_coords = []
        y_coords = []
        
        for comp_name, pin_name in connections:
            if comp_name == 'PIN' and pin_name in pins:
                # IOpin
                pos = pins[pin_name].get('position', (0, 0))
                x_coords.append(pos[0])
                y_coords.append(pos[1])
            elif comp_name in components:
                #  componentpin
                pos = components[comp_name].get('position', (0, 0))
                x_coords.append(pos[0])
                y_coords.append(pos[1])
        
        if len(x_coords) >= 2:
            #  calculateHPWL = (max_x - min_x) + (max_y - min_y)
            return (max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords))
        
        return 0.0
    
    def _parse_routing_info(self, routing_text: str) -> tuple:
        """ parserouting[CN][CN][CN][CN][CN]information, calculate actual[CN][CN][CN][CN][CN][CN][CN]
        
        [CN][CN][CN][CN] gate processrouting[CN][CN]DEF[CN][CN][CN][CN]ROUTEDpart, extract:
        1. wire_length: [CN][CN] parse[CN][CN][CN][CN][CN] calculate[CN][CN][CN][CN][CN][CN][CN] actual[CN][CN]
        2. via_count: statistics[CN][CN][CN][CN][CN][CN][CN][CN]
        
        [CN][CN] data[CN][CN][CN]gate-gateedge[CN] label,[CN][CN]D[CN]pin-pinedge label[CN][CN]
        
        Args:
            routing_text: net[CN][CN][CN][CN][CN][CN][CN][CN]part
            
        Returns:
            tuple: (wire_length, via_count) -  actual[CN][CN][CN][CN][CN][CN][CN]
        """
        import re
        
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
    
    def _calculate_net_routing_info(self, net_name: str) -> tuple:
        """ calculate[CN]net[CN]routinginformation( used forgate_gateedge label)
        
        Args:
            net_name: net[CN][CN]
            
        Returns:
            tuple: (wire_length, via_count) -  actual[CN][CN][CN][CN][CN][CN][CN]
        """
        nets = self.def_data.get('nets', {})
        
        if net_name not in nets:
            return 0.0, 0
            
        net_info = nets[net_name]
        
        # [CN][CN][CN][CN][CN] parse[CN][CN][CN]information
        wire_length = net_info.get('wire_length', 0.0)
        via_count = net_info.get('via_count', 0)
        
        return wire_length, via_count
    
    def _read_config_utilization(self):
        """[CN]place_data_extract.csv[CN][CN] read corresponding to[CN][CN][CN]CORE_UTILIZATION configuration"""
        try:
            import csv
            import os
            
            # [CN]DEFextract design name from file path([CN][CN]_place.def[CN][CN])
            if hasattr(self, 'def_file_path') and self.def_file_path:
                import os
                filename = os.path.basename(self.def_file_path)
                #  extract_place[CN][CN][CN]part[CN][CN][CN][CN][CN][CN]
                design_name = filename.replace('_place.def', '')
            else:
                # [CN][CN][CN][CN]:[CN][CN]DEF[CN][CN][CN][CN][CN][CN][CN][CN][CN]
                design_name = self.def_data.get('design_name', '')
            
            # CSV[CN][CN][CN][CN] - [CN][CN][CN][CN][CN][CN][CN][CN][CN]
            csv_path = 'route_data_extract.csv'
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # [CN][CN][CN][CN][CN][CN][CN][CN][CN][CN][CN]
                    csv_design_name = row['design_name']
                    match_found = False  # Initialize[CN][CN][CN][CN]
                    
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
            print(f"[!]️  warning: [CN][CN] readplace_data_extract.csv[CN][CN]: {e}")
            return 0.0  #  default[CN]
    
    def _read_routing_labels(self):
        """[CN]route_data_extract.csv[CN][CN] read corresponding to[CN][CN][CN]routing[CN][CN]global label:utilization, hpwl, total_wire_length, total_vias"""
        try:
            import csv
            import os
            
            # [CN]DEFextract design name from file path([CN][CN]_route.def[CN][CN])
            if hasattr(self, 'def_file_path') and self.def_file_path:
                import os
                filename = os.path.basename(self.def_file_path)
                #  extract_route[CN][CN][CN]part[CN][CN][CN][CN][CN][CN]
                design_name = filename.replace('_route.def', '').replace('_place.def', '')
            else:
                # [CN][CN][CN][CN]:[CN][CN]DEF[CN][CN][CN][CN][CN][CN][CN][CN][CN]
                design_name = self.def_data.get('design_name', '')
            
            # CSV[CN][CN][CN][CN] - [CN][CN][CN][CN][CN][CN][CN][CN][CN]
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
                        #  readdesign_utilization ([CN][CN][CN][CN][CN])
                        util_val = row['design_utilization']
                        if util_val and util_val.strip():
                            utilization = float(util_val)
                        
                        #  readhpwl_after ([CN][CN][CN][CN][CN])
                        hpwl_val = row['hpwl_after']
                        if hpwl_val and hpwl_val.strip():
                            hpwl = float(hpwl_val)
                        
                        #  readtotal_wire_length
                        wire_length_val = row['total_wire_length']
                        if wire_length_val and wire_length_val.strip():
                            total_wire_length = float(wire_length_val)
                        
                        #  readtotal_vias
                        vias_val = row['total_vias']
                        if vias_val and vias_val.strip():
                            total_vias = float(vias_val)
                        
                        print(f"[OK] successfully matched design: DEF[CN][CN]'{design_name}' <-> CSV[CN][CN]'{csv_design_name}'")
                        return utilization, hpwl, total_wire_length, total_vias
            
            print(f"[!]️  warning: [CN]CSV[CN][CN][CN][CN][CN][CN][CN][CN] {design_name} [CN]routing label data")
            return 0.0, 0.0, 0.0, 0.0  #  default[CN]
        except Exception as e:
            print(f"[!]️  warning: [CN][CN] readroute_data_extract.csv[CN][CN]: {e}")
            return 0.0, 0.0, 0.0, 0.0  #  default[CN]
    
    def _calculate_component_area(self, cell_type: str) -> float:
        """[CN][CN] celltype calculate component area"""
        #  simplified[CN] area calculate -  based on drive strength
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
            return 1.0  #  default area
    
    def _print_statistics(self):
        """ outputstatisticsinformation"""
        print("\n[CHART] F[CN]generatestatisticsinformation:")
        print(f"   [CN]gate[CN][CN]: {self.stats['total_gates']}")
        print(f"   [CN]IO_Pin[CN][CN]: {self.stats['total_io_pins']}")
        print(f"   Gate-Gateedge[CN][CN]: {self.stats['total_gate_gate_edges']}")
        print(f"   IO_Pin-Gateedge[CN][CN]: {self.stats['total_io_pin_gate_edges']}")
        print(f"    process[CN]net[CN][CN]: {self.stats['nets_processed']}")

# ============================================================================
# MAIN FUNCTION SECTION - Main functionregion
# ============================================================================

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='F[CN][CN][CN][CN]generate[CN]')
    parser.add_argument('def_file', help=' input[CN]DEF[CN][CN][CN][CN]')
    parser.add_argument('-o', '--output', help=' output[CN].pt[CN][CN][CN][CN]', default=None)

    
    args = parser.parse_args()
    
    # [CN][CN] input[CN][CN]
    if not os.path.exists(args.def_file):
        print(f"[X] error:DEF[CN][CN][CN][CN][CN]: {args.def_file}")
        return
    
    # set output file path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.def_file))[0]
        args.output = f"{base_name}_f_graph.pt"
    
    try:
        print(f"[SEARCH]  parseDEF[CN][CN]: {args.def_file}")
        
        # 1.  parseDEF[CN][CN]
        parser = DEFParser(args.def_file)
        def_data = parser.parse()
        
        # 2. generateF[CN]
        generator = FHeterographGenerator(def_data, args.def_file)
        hetero_data = generator.generate()
        
        # 3. Save result
        print(f"[SAVE] saveF[CN][CN]: {args.output}")
        torch.save(hetero_data, args.output)
        

        
        print("[OK] F[CN]generatecompleted!")
        
    except Exception as e:
        print(f"[X]  processfailed: {e}")
        traceback.print_exc()



if __name__ == "__main__":
    main()