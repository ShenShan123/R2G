//---------------------------------------------------------------------------------------
// uart top level module  
//
//---------------------------------------------------------------------------------------

module uart_top 
(
	// global signals 
	clock, reset,
	// uart serial signals 
	ser_in, ser_out,
	// transmit and receive internal interface signals 
	rx_data, rx_busy,
	new_rx_data, 
	tx_data, tx_busy, 
	// baud rate configuration register - see baud_gen.v for details 
	ce_1,
	baud_freq, baud_limit, 
	baud_clk, tx_begin
);
//---------------------------------------------------------------------------------------
// modules inputs and outputs 
input 			clock;			// global clock input 
input 			reset;			// global reset input 
input			ser_in;			// serial data input      
output			ser_out;		// serial data output 
input	[7:0]	tx_data;		// data byte to transmit 
output 			tx_busy;		// signs that transmitter is busy 
output	[7:0]	rx_data;		// data byte received 
output			rx_busy;
output 			new_rx_data;	// signs that a new byte was received      
input	[11:0]	baud_freq;	// baud rate setting registers - see header description 
input	[15:0]	baud_limit;
output			baud_clk;
output         ce_1;
input			tx_begin;

// internal wires 
wire ce_16;		// clock enable at bit rate 

assign baud_clk = ce_16;
//---------------------------------------------------------------------------------------
// module implementation 
// baud rate generator module 
baud_gen baud_gen_1
(
	.clock(clock), .reset(reset), 
	.ce_16(ce_16), .baud_freq(baud_freq), .baud_limit(baud_limit)
);

// uart receiver 
uart_rx uart_rx_1 
(
	.clock(clock), .reset(reset), 
	.ce_16(ce_16), .ce_1(ce_1),
	.ser_in(ser_in), 
	.rx_data(rx_data), .rx_busy(rx_busy),
	.new_rx_data(new_rx_data) 
);

// uart transmitter 
uart_tx  uart_tx_1
(
	.clock(clock), .reset(reset), 
	.ce_16(ce_16), .tx_data(tx_data),
	.ser_out(ser_out), .tx_busy(tx_busy),
	.tx_begin(tx_begin)
);

endmodule
//---------------------------------------------------------------------------------------
//						That is all !
//---------------------------------------------------------------------------------------
