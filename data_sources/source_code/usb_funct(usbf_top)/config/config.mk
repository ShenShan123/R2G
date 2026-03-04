export DESIGN_NAME = usbf_top
export PLATFORM    = nangate45

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_top.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_crc5.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_crc16.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_defines.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_ep_rf.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_ep_rf_dummy.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_idma.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_mem_arb.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_pa.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_pd.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_pe.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_pl.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_rf.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_utmi_if.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_utmi_ls.v \
                       $(DESIGN_HOME)/src/$(DESIGN_NAME)/rtl/usbf_wb.v

export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NAME)/constraint.sdc
export ABC_AREA      = 1

export CORE_UTILIZATION = 40
export PLACE_DENSITY    = 0.60

export TNS_END_PERCENT  = 100
