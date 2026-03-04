
export DESIGN_NICKNAME = ac97_ctrl

export DESIGN_NAME = ac97_top

export PLATFORM    = nangate45

ABC_AREA = 1

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/*.v

export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/constraint.sdc

# export ADDER_MAP_FILE :=

export CORE_UTILIZATION = 40
export PLACE_DENSITY_LB_ADDON = 0.10
export TNS_END_PERCENT = 100
