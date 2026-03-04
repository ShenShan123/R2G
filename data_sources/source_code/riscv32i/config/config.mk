export DESIGN_NICKNAME ?= riscv32i
export DESIGN_NAME = riscv_top
export PLATFORM    = nangate45

export SYNTH_HIERARCHICAL ?= 1

export RTLMP_MIN_INST = 1000
export RTLMP_MAX_INST = 3500
export RTLMP_MIN_MACRO = 1
export RTLMP_MAX_MACRO = 5

export SYNTH_MINIMUM_KEEP_SIZE ?= 10000

export VERILOG_FILES = $(sort $(wildcard $(DESIGN_HOME)/src/riscv32i/*.v))
export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/riscv32i/constraint.sdc


export ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef/fakeram45_256x32.lef \
                         $(PLATFORM_DIR)/lef/fakeram45_64x64.lef

export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib/fakeram45_256x32.lib \
                         $(PLATFORM_DIR)/lib/fakeram45_64x64.lib

export IO_PLACER_H = metal5
export IO_PLACER_V = metal6

#export DIE_AREA = 0 0 80 90
#export CORE_AREA = 5 5 75 85 
export CORE_UTILIZATION = 40
export CORE_ASPECT_RATIO = 1

export PLACE_DENSITY_LB_ADDON = 0.10

export IO_CONSTRAINTS     = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/io.tcl
export MACRO_PLACE_HALO    = 2 2

export TNS_END_PERCENT   = 100

# nano ./designs/nangate45/riscv32i/constraint.sdc
# make DESIGN_CONFIG=./designs/nangate45/riscv32i/config.mk
