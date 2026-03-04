export PLATFORM         = nangate45
export DESIGN_NAME      = pci_bridge32
export DESIGN_NICKNAME  = pci
# File settings
export SDC_FILE      = ./designs/$(PLATFORM)/$(DESIGN_NICKNAME)/constraint.sdc

export VERILOG_FILES = $(sort $(wildcard $(DESIGN_HOME)/src/pci/*.v))

export CORE_UTILIZATION = 40
export PLACE_DENSITY    = 0.60
export TNS_END_PERCENT  = 100


#nano ./designs/nangate45/pci/constraint.sdc
#make DESIGN_CONFIG=./designs/nangate45/pci/config.mk
