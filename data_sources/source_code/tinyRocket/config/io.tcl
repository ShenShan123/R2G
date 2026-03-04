# Restrict IO pins to bottom edge only
set_io_pin_constraint -region bottom:*
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
