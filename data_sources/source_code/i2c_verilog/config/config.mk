export DESIGN_NAME = i2c_master_top
export DESIGN_NICKNAME = i2c_verilog
export PLATFORM = nangate45

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/i2c_master_top.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/i2c_master_bit_ctrl.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/i2c_master_byte_ctrl.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/i2c_master_defines.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/timescale.v 
export SDC_FILE = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/constraint.sdc
export ABC_AREA = 1
export CORE_UTILIZATION ?= 45
export PLACE_DENSITY_LB_ADDON = 0.20
export TNS_END_PERCENT        = 100


