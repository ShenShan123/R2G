current_design pci

set clk_name1 web_clk
set clk_port_name1 web_clk_i
set clk_period1 2.0
set clk_io_pct1 0.2

set clk_port1 [get_ports $clk_port_name1]

create_clock -name $clk_name1 -period $clk_period1 $clk_port1

set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] $clk_port1]

set_input_delay  [expr $clk_period1 * $clk_io_pct1] -clock $clk_name1 $non_clock_inputs 
set_output_delay [expr $clk_period1 * $clk_io_pct1] -clock $clk_name1 [all_outputs]



set clk_name2 pci_clk
set clk_port_name2 pci_clk_i
set clk_period2 2.0
set clk_io_pct2 0.2

set clk_port2 [get_ports $clk_port_name2]

create_clock -name $clk_name2 -period $clk_period2 $clk_port2

set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] $clk_port2]

set_input_delay  [expr $clk_period2 * $clk_io_pct2] -clock $clk_name2 $non_clock_inputs 
set_output_delay [expr $clk_period2 * $clk_io_pct2] -clock $clk_name2 [all_outputs]
