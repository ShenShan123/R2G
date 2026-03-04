export DESIGN_NICKNAME = tinyRocket
export DESIGN_NAME = RocketTile
export PLATFORM    = nangate45

export SYNTH_HIERARCHICAL = 1
export SYNTH_MINIMUM_KEEP_SIZE ?= 5000

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/AsyncResetReg.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/ClockDivider2.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/ClockDivider3.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/plusarg_reader.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/freechips.rocketchip.system.TinyConfig.v \
                       $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/freechips.rocketchip.system.TinyConfig.v

export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/constraint.sdc

export ADDITIONAL_LEFS = $(sort $(wildcard $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/*.lef))
export ADDITIONAL_LIBS = $(sort $(wildcard $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/*.lib))


# These values must be multiples of placement site
# x=0.19 y=1.4
#export DIE_AREA    = 0 0 424.92 499.4
#export CORE_AREA   = 10.07 9.8 414.85 489.6
#export TNS_END_PERCENT        = 100

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

