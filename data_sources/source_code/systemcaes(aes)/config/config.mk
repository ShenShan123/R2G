export DESIGN_NAME = aes
export PLATFORM    = nangate45

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/aes.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/byte_mixcolum.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/keysched.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/mixcolum.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/sbox.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/subbytes.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/timescale.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/word_mixcolum.v

export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NAME)/constraint.sdc
export ABC_AREA      = 1

# Ultra-simple floorplan configuration
export FP_ASPECT_RATIO = 1
export FP_CORE_UTIL    = 10
export FP_IO_MIN_SPACING = 0.0
export FP_PDN_HORIZONTAL_HALO = 10.0
export FP_PDN_VERTICAL_HALO = 10.0

# Disable problematic features for smooth execution
export SKIP_INCREMENTAL_REPAIR = 1
export ENABLE_DPO = 0
export DETAILED_ROUTE_END_ITERATION = 10
export OR_SEED = 1
export OR_K = 4

# Multiplier circuit optimizations
export MULT_MAP_FILE :=

export CORE_UTILIZATION ?= 15
export PLACE_DENSITY_LB_ADDON = 0.01
export TNS_END_PERCENT        = 1

# Skip complex optimizations for simple GDS generation
export SKIP_GATE_CLONING = 1
export SKIP_BUFFER_REMOVAL = 1
export SKIP_PIN_SWAP = 1
export ENABLE_RECOVERY = 0
export SKIP_CTS_REPAIR = 1
export SKIP_TIMING_REPAIR = 1
export SKIP_LAST_GASP = 1
export CTS_BUF_DISTANCE = 100
export SETUP_SLACK_MARGIN = 0.0
export HOLD_SLACK_MARGIN = 0.0

# Conservative routing settings to avoid detail_route errors
export RT_CLOCK_MIN_LAYER = 2
export RT_CLOCK_MAX_LAYER = 4
export ROUTING_CORES = 1
export DRT_MIN_LAYER = 2
export DRT_MAX_LAYER = 4

# Simplify flow for GDS generation
export ENABLE_ANTENNA_REPAIR = 0
export ENABLE_FILL_INSERTION = 0
export MACRO_PLACE_HALO = 1 1
export MACRO_PLACE_CHANNEL = 2 2

# Emergency bypass settings to avoid map::at error
export SKIP_DETAILED_ROUTE = 1
export USE_FILL = 0
export DETAILED_ROUTE_ARGS = 
export GLOBAL_ROUTE_ARGS = -allow_congestion -overflow_iterations 50
export MIN_ROUTING_LAYER = 1
export MAX_ROUTING_LAYER = 10
export ENABLE_SIMPLE_ROUTE = 1
export FASTROUTE_TCL = $(PLATFORM_DIR)/fastroute.tcl
export REMOVE_CELLS_FOR_EQY   = TAPCELL*
