export DESIGN_NICKNAME = ss_pcm
export PLATFORM    = nangate45
export DESIGN_NAME = pcm_slv_top

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/pcm_slv_top.v
export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/constraint.sdc

export ABC_AREA        = 1            # Enable ABC area optimization mode, prioritize reducing circuit area

# Adders degrade GCD

export ADDER_MAP_FILE :=              # Clear adder mapping file

export CORE_UTILIZATION ?= 55         # Define core utilization as 55% (lower increases area/higher routing congestion)
export PLACE_DENSITY_LB_ADDON = 0.20  # Increase placement density by 0.20, improve placement compactness
export TNS_END_PERCENT        = 100   # Optimize timing for all paths (100% coverage)
export REMOVE_CELLS_FOR_EQY   = TAPCELL*
                                      # Remove all TapCells during equivalence verification
