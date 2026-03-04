//---------------------------------------------------------------------------------------
// Simple Uart Core
//
//---------------------------------------------------------------------------------------

module uart2bus_top 
(
	// global signals 
	clock, reset,
	new_rx_data,ce_1,
	// uart serial signals 
	ser_out,ser_in
	// internal bus to register file 
	//int_address, int_wr_data, int_write,
	//int_rd_data, int_read, 
	//int_req, int_gnt 
);
//parameter T1ms =23'd50000;
//---------------------------------------------------------------------------------------
// modules inputs and outputs 
input 			clock;			// global clock input 
input 			reset;			// global reset input 
output			ser_out;		// serial data output 
input          ser_in;
//output 			rx_data;
output			new_rx_data;
output			ce_1;

// baud rate configuration, see baud_gen.v for more details.
// baud rate generator parameters for 115200 baud on 50MHz clock 
`define D_BAUD_FREQ			12'd576
`define D_BAUD_LIMIT		16'd15049
// baud rate generator parameters for 115200 baud on 40MHz clock 
//`define D_BAUD_FREQ			12'h90
//`define D_BAUD_LIMIT		16'h0ba5
// baud rate generator parameters for 115200 baud on 44MHz clock 
// `define D_BAUD_FREQ			12'd23
// `define D_BAUD_LIMIT		16'd527
// baud rate generator parameters for 9600 baud on 66MHz clock 
//`define D_BAUD_FREQ		12'h10
//`define D_BAUD_LIMIT		16'h1ACB

// internal wires 
reg	[7:0]	tx_data;		// data byte to transmit 
reg			tx_begin;
wire			rx_busy;
wire 			tx_busy;		// signs that transmitter is busy 
wire	[7:0]	rx_data;		// data byte received 
wire 			new_rx_data;	// signs that a new byte was received 
wire	[11:0]	baud_freq;
wire	[15:0]	baud_limit;
wire			baud_clk;
wire  		ce_1;
reg	[1:0]  i;
//reg	[22:0] Count1;

//always @ (posedge clock or negedge reset)
//	if (reset == 1'b0)
//		Count1 <= 23'd0;
//	else if(Count1 == T1ms)
//		Count1 <= 23'd0;
//	else
//		Count1 <= Count1 + 1'b1;

//always @ (posedge clock or negedge reset)        //test for UART sending
//begin
//	if (reset == 1'b0) 
//		begin
//			i <= 1'b0;
//		end
//	else
//		begin
//			case(i)
//			0:
//				begin
//					if(~tx_busy)
//						i <= i + 1'b1;
//					else
//						tx_data <= 8'd5;
//				end
//			1:
//				begin
//					if(~tx_busy)
//						i <= i - 1'b1;
//					else
//						tx_data <= 8'd15;
//				end
//			default:
//				begin
//					i <= 1'b0;
//				end
//			endcase
//		end					
//end

always @ (posedge clock or negedge reset)      //test for UART sending and Receiving
	if (reset == 1'b0) 
		begin
			i <= 1'b0;
		end
	else
		begin
			case(i)
			0:
				begin
					if(new_rx_data)
						begin
							tx_data <= rx_data;
							tx_begin <= 1'b1;
							//i <= i + 1'b1;
						end
					else
						begin
							i <= 1'b0;
							tx_begin <= 1'b0;
						end	
				end
//			1:
//				begin
//					if(Count1 == T1ms)
//						i <= i - 1'b1;
//					else if(tx_begin)
//						tx_begin <= 1'b0;
//					else
//						i <= 1'b1;
//				end
			default:
				tx_begin <= 1'b0;
			endcase
		end
					
				
			


//---------------------------------------------------------------------------------------
// module implementation 
// uart top module instance 
uart_top uart1
(
	.clock(clock), .reset(reset),
	.ser_in(ser_in), .ser_out(ser_out),
	.rx_data(rx_data), .rx_busy(rx_busy),
	.new_rx_data(new_rx_data), 
	.tx_data(tx_data),  .tx_busy(tx_busy), 
	.ce_1(ce_1),
	.baud_freq(baud_freq), .baud_limit(baud_limit),
	.baud_clk(baud_clk), .tx_begin(tx_begin)
);

// assign baud rate default values 
assign baud_freq = `D_BAUD_FREQ;
assign baud_limit = `D_BAUD_LIMIT;

// uart parser instance 
//uart_parser #(16) uart_parser1
//(
//	.clock(clock), .reset(reset),
//	.rx_data(rx_data), .new_rx_data(new_rx_data), 
//	.tx_data(tx_data), .new_tx_data(new_tx_data), .tx_busy(tx_busy), 
//	.int_address(int_address), .int_wr_data(int_wr_data), .int_write(int_write),
//	.int_rd_data(int_rd_data), .int_read(int_read), 
//	.int_req(int_req), .int_gnt(int_gnt) 
//);

endmodule
//---------------------------------------------------------------------------------------
//						 That is all
//---------------------------------------------------------------------------------------
