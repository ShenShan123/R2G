# Specify current design top-level module (can be omitted in OpenROAD)
current_design aes128_core

# Define main clock and its period
set clk_name       core_clock
set clk_port_name  i_clk
set clk_period     10.0    ;# 10ns = 100MHz
set clk_io_pct     0.2     ;# 20% input/output delay percentage

# Get clock port
set clk_port [get_ports $clk_port_name]

# Create main clock
create_clock -name $clk_name -period $clk_period $clk_port

# Get all non-clock input ports
set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] $clk_port]

# Set input delay and output delay (0.2 * 10ns = 2.0ns)
set_input_delay  [expr $clk_period * $clk_io_pct] -clock $clk_name $non_clock_inputs
set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_name [all_outputs]

