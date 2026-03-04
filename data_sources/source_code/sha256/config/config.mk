export DESIGN_NAME = sha256
export PLATFORM = nangate45

export VERILOG_FILES = $(sort $(wildcard ./designs/src/$(DESIGN_NICKNAME)/*.v))
export SDC_FILE = ./designs/$(PLATFORM)/$(DESIGN_NICKNAME)/constraint.sdc

export CORE_UTILIZATION = 40
export PLACE_DENSITY = 0.6
export TNS_END_PERCENT = 100
