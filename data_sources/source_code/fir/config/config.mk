export PLATFORM = nangate45

export DESIGN_NAME      = FIR_filter
export DESIGN_NICKNAME = fir

export VERILOG_FILES    = $(sort $(wildcard ./designs/src/$(DESIGN_NICKNAME)/*.v))
export SDC_FILE         = ./designs/$(PLATFORM)/$(DESIGN_NICKNAME)/constraint.sdc

export CORE_UTILIZATION = 40
export PLACE_DENSITY    = 0.60

export TNS_END_PERCENT  = 100
