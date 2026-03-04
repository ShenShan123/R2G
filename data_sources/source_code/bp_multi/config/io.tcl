# exclude_io_pin_region -region left:100-1100 -region right:100-1100 -region top:*

# Restrict IO pins to bottom edge only
 set_io_pin_constraint -region bottom:*
#set_io_pin_constraint -region bottom:0.5  # Half on bottom edge
#set_io_pin_constraint -region left:0.25   # Quarter on left side
#set_io_pin_constraint -region right:0.25  # Quarter on right side

# place_pin -exclude left:* -exclude right:* -exclude top:*

puts "Starting IO pin placement..."

# Method 1: Try using place_pins command to place all IO pins directly
# Get all ports
set all_ports [get_ports]
set port_count [llength $all_ports]
puts "Found $port_count IO ports"

# If you only want bottom edge IO pins, you can set it like this
# But first we ensure all pins are placed
set IO_PLACER_H  metal5
set IO_PLACER_V  metal6

place_pins -hor_layers $IO_PLACER_H -ver_layers $IO_PLACER_V \
            -corner_avoidance 0 -min_distance 2.0

puts "IO pin placement completed - using place_pins"
