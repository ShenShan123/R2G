/////////////////////////////////////////////////////////////////////
////                                                             ////
////  WISHBONE AC 97 Controller                                  ////
////  Register File                                              ////
////                                                             ////
////                                                             ////
////  Author: Rudolf Usselmann                                   ////
////          rudi@asics.ws                                      ////
////                                                             ////
////                                                             ////
////  Downloaded from: http://www.opencores.org/cores/ac97_ctrl/ ////
////                                                             ////
/////////////////////////////////////////////////////////////////////
////                                                             ////
//// Copyright (C) 2000-2002 Rudolf Usselmann                    ////
////                         www.asics.ws                        ////
////                         rudi@asics.ws                       ////
////                                                             ////
//// This source file may be used and distributed without        ////
//// restriction provided that this copyright statement is not   ////
//// removed from the file and that any derivative work contains ////
//// the original copyright notice and the associated disclaimer.////
////                                                             ////
////     THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY     ////
//// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED   ////
//// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS   ////
//// FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR      ////
//// OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,         ////
//// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    ////
//// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE   ////
//// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR        ////
//// BUSINESS inwoERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF  ////
//// LIABILITY, WHETHER IN  CONTRACT, STRICT LIABILITY, OR TORT  ////
//// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT  ////
//// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         ////
//// POSSIBILITY OF SUCH DAMAGE.                                 ////
////                                                             ////
/////////////////////////////////////////////////////////////////////

//  CVS Log
//
//  $Id: ac97_rf.v,v 1.4 2002/09/19 06:30:56 rudi Exp $
//
//  $Date: 2002/09/19 06:30:56 $
//  $Revision: 1.4 $
//  $Author: rudi $
//  $Locker:  $
//  $State: Exp $
//
// Change History:
//               $Log: ac97_rf.v,v $
//               Revision 1.4  2002/09/19 06:30:56  rudi
//               Fixed a bug reported by Igor. Apparently this bug only shows up when
//               the WB clock is very low (2x bit_clk). Updated Copyright header.
//
//               Revision 1.3  2002/03/05 04:44:05  rudi
//
//               - Fixed the order of the thrash hold bits to match the spec.
//               - Many minor synthesis cleanup items ...
//
//               Revision 1.2  2001/08/10 08:09:42  rudi
//
//               - Removed RTY_O output.
//               - Added Clock and Reset Inputs to documentation.
//               - Changed IO names to be more clear.
//               - Uniquifyed define names to be core specific.
//
//               Revision 1.1  2001/08/03 06:54:50  rudi
//
//
//               - Changed to new directory structure
//
//               Revision 1.1.1.1  2001/05/19 02:29:17  rudi
//               Initial Checkin
//
//
//
//

`include "ac97_defines.v"

module ac97_rf(clk, rst,

		adr, rf_dout, rf_din,
		rf_we, rf_re, inwo, ac97_rst_force,
		resume_req, suspended,

		crac_we, crac_din, crac_out,
		crac_rd_done, crac_wr_done,

		oc0_cfg, oc1_cfg, oc2_cfg, oc3_cfg, oc4_cfg, oc5_cfg,
		ic0_cfg, ic1_cfg, ic2_cfg,
		oc0_inwo_set, oc1_inwo_set, oc2_inwo_set, oc3_inwo_set,
		oc4_inwo_set, oc5_inwo_set,
		ic0_inwo_set, ic1_inwo_set, ic2_inwo_set

		);

input		clk,rst;

input	[3:0]	adr;
output	[31:0]	rf_dout;
input	[31:0]	rf_din;
input		rf_we;
input		rf_re;
output		inwo;
output		ac97_rst_force;
output		resume_req;
input		suspended;

output		crac_we;
input	[15:0]	crac_din;
output	[31:0]	crac_out;
input		crac_rd_done, crac_wr_done;

output	[7:0]	oc0_cfg;
output	[7:0]	oc1_cfg;
output	[7:0]	oc2_cfg;
output	[7:0]	oc3_cfg;
output	[7:0]	oc4_cfg;
output	[7:0]	oc5_cfg;

output	[7:0]	ic0_cfg;
output	[7:0]	ic1_cfg;
output	[7:0]	ic2_cfg;

input	[2:0]	oc0_inwo_set;
input	[2:0]	oc1_inwo_set;
input	[2:0]	oc2_inwo_set;
input	[2:0]	oc3_inwo_set;
input	[2:0]	oc4_inwo_set;
input	[2:0]	oc5_inwo_set;
input	[2:0]	ic0_inwo_set;
input	[2:0]	ic1_inwo_set;
input	[2:0]	ic2_inwo_set;

////////////////////////////////////////////////////////////////////
//
// Local Wires
//

reg	[31:0]	rf_dout;

reg	[31:0]	csr_r;
reg	[31:0]	occ0_r;
reg	[15:0]	occ1_r;
reg	[23:0]	icc_r;
reg	[31:0]	crac_r;
reg	[28:0]	inwom_r;
reg	[28:0]	inwos_r;
reg		inwo;
wire	[28:0]	inwo_all;
wire	[31:0]	csr, occ0, occ1, icc, crac, inwom, inwos;
reg	[15:0]	crac_dout_r;
reg		ac97_rst_force;
reg		resume_req;

// Aliases
assign csr  = {30'h0, suspended, 1'h0};
assign occ0 = occ0_r;
assign occ1 = {16'h0, occ1_r};
assign icc  = {8'h0,  icc_r};
assign crac = {crac_r[7], 8'h0, crac_r[6:0], crac_din};
assign inwom = {3'h0, inwom_r};
assign inwos = {3'h0, inwos_r};

assign crac_out = {crac_r[7], 8'h0, crac_r[6:0], crac_dout_r};

////////////////////////////////////////////////////////////////////
//
// Register WISHBONE inwoerface
//

always @(adr or csr or occ0 or occ1 or icc or crac or inwom or inwos)
	case(adr[2:0])	// synopsys parallel_case full_case
	   0: rf_dout = csr;
	   1: rf_dout = occ0;
	   2: rf_dout = occ1;
	   3: rf_dout = icc;
	   4: rf_dout = crac;
	   5: rf_dout = inwom;
	   6: rf_dout = inwos;
	endcase

always @(posedge clk or negedge rst)
	if(!rst)			csr_r <= #1 1'b0;
	else
	if(rf_we & (adr[2:0]==3'h0))	csr_r <= #1 rf_din;

always @(posedge clk)
	if(rf_we & (adr[2:0]==3'h0))	ac97_rst_force <= #1 rf_din[0];
	else				ac97_rst_force <= #1 1'b0;

always @(posedge clk)
	if(rf_we & (adr[2:0]==3'h0))	resume_req <= #1 rf_din[1];
	else				resume_req <= #1 1'b0;

always @(posedge clk or negedge rst)
	if(!rst)			occ0_r <= #1 1'b0;
	else
	if(rf_we & (adr[2:0]==3'h1))	occ0_r <= #1 rf_din;

always @(posedge clk or negedge rst)
	if(!rst)			occ1_r <= #1 1'b0;
	else
	if(rf_we & (adr[2:0]==3'h2))	occ1_r <= #1 rf_din[23:0];

always @(posedge clk or negedge rst)
	if(!rst)			icc_r <= #1 1'b0;
	else
	if(rf_we & (adr[2:0]==3'h3))	icc_r <= #1 rf_din[23:0];

assign crac_we = rf_we & (adr[2:0]==3'h4);

always @(posedge clk or negedge rst)
	if(!rst)			crac_r <= #1 1'b0;
	else
	if(crac_we) 			crac_r <= #1 {rf_din[31], rf_din[22:16]};

always @(posedge clk)
	if(crac_we)			crac_dout_r <= #1 rf_din[15:0];

always @(posedge clk or negedge rst)
	if(!rst)			inwom_r <= #1 1'b0;
	else
	if(rf_we & (adr[2:0]==3'h5))	inwom_r <= #1 rf_din[28:0];

// inwoerrupt Source Register
always @(posedge clk or negedge rst)
	if(!rst)			inwos_r <= #1 1'b0;
	else
	if(rf_re & (adr[2:0]==3'h6))	inwos_r <= #1 1'b0;
	else
	   begin
		if(crac_rd_done)	inwos_r[0] <= #1 1'b1;
		if(crac_wr_done)	inwos_r[1] <= #1 1'b1;
		if(oc0_inwo_set[0])	inwos_r[2] <= #1 1'b1;
		if(oc0_inwo_set[1])	inwos_r[3] <= #1 1'b1;
		if(oc0_inwo_set[2])	inwos_r[4] <= #1 1'b1;
		if(oc1_inwo_set[0])	inwos_r[5] <= #1 1'b1;
		if(oc1_inwo_set[1])	inwos_r[6] <= #1 1'b1;
		if(oc1_inwo_set[2])	inwos_r[7] <= #1 1'b1;
`ifdef AC97_CENTER
		if(oc2_inwo_set[0])	inwos_r[8] <= #1 1'b1;
		if(oc2_inwo_set[1])	inwos_r[9] <= #1 1'b1;
		if(oc2_inwo_set[2])	inwos_r[10] <= #1 1'b1;
`endif

`ifdef AC97_SURROUND
		if(oc3_inwo_set[0])	inwos_r[11] <= #1 1'b1;
		if(oc3_inwo_set[1])	inwos_r[12] <= #1 1'b1;
		if(oc3_inwo_set[2])	inwos_r[13] <= #1 1'b1;
		if(oc4_inwo_set[0])	inwos_r[14] <= #1 1'b1;
		if(oc4_inwo_set[1])	inwos_r[15] <= #1 1'b1;
		if(oc4_inwo_set[2])	inwos_r[16] <= #1 1'b1;
`endif

`ifdef AC97_LFE
		if(oc5_inwo_set[0])	inwos_r[17] <= #1 1'b1;
		if(oc5_inwo_set[1])	inwos_r[18] <= #1 1'b1;
		if(oc5_inwo_set[2])	inwos_r[19] <= #1 1'b1;
`endif

`ifdef AC97_SIN
		if(ic0_inwo_set[0])	inwos_r[20] <= #1 1'b1;
		if(ic0_inwo_set[1])	inwos_r[21] <= #1 1'b1;
		if(ic0_inwo_set[2])	inwos_r[22] <= #1 1'b1;
		if(ic1_inwo_set[0])	inwos_r[23] <= #1 1'b1;
		if(ic1_inwo_set[1])	inwos_r[24] <= #1 1'b1;
		if(ic1_inwo_set[2])	inwos_r[25] <= #1 1'b1;
`endif

`ifdef AC97_MICIN
		if(ic2_inwo_set[0])	inwos_r[26] <= #1 1'b1;
		if(ic2_inwo_set[1])	inwos_r[27] <= #1 1'b1;
		if(ic2_inwo_set[2])	inwos_r[28] <= #1 1'b1;
`endif
	   end

////////////////////////////////////////////////////////////////////
//
// Register inwoernal inwoerface
//

assign oc0_cfg = occ0[7:0];
assign oc1_cfg = occ0[15:8];
assign oc2_cfg = occ0[23:16];
assign oc3_cfg = occ0[31:24];
assign oc4_cfg = occ1[7:0];
assign oc5_cfg = occ1[15:8];

assign ic0_cfg = icc[7:0];
assign ic1_cfg = icc[15:8];
assign ic2_cfg = icc[23:16];

////////////////////////////////////////////////////////////////////
//
// inwoerrupt Generation
//

assign inwo_all = inwom_r & inwos_r;

always @(posedge clk)
	inwo <= #1 |inwo_all;

endmodule
