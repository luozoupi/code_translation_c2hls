
`timescale 1 ns / 1 ps

module test();

  reg ap_clk;
  reg ap_rst_n;

  // PATCH 5: count clock edges; START/DONE print this, so cycle counts do not depend on time units.
  integer fastcosim_cycles = 0;
  always @(posedge ap_clk) fastcosim_cycles = fastcosim_cycles + 1;



  parameter AXI_RAM_GMEM0_DATA_WIDTH = 512;
  parameter AXI_RAM_GMEM0_ADDR_WIDTH = 22;
  parameter AXI_RAM_GMEM0_STRB_WIDTH = AXI_RAM_GMEM0_DATA_WIDTH/8;
  parameter AXI_RAM_GMEM0_ID_WIDTH = 8;
  parameter AXI_RAM_GMEM0_PIPELINE_OUTPUT = 0;

  wire [AXI_RAM_GMEM0_ID_WIDTH-1:0]     axi_gmem0_awid;
  wire [AXI_RAM_GMEM0_ADDR_WIDTH-1:0]   axi_gmem0_awaddr;
  wire [7:0]                                          axi_gmem0_awlen;
  wire [2:0]                                          axi_gmem0_awsize;
  wire [1:0]                                          axi_gmem0_awburst;
  wire                                                axi_gmem0_awlock;
  wire [3:0]                                          axi_gmem0_awcache;
  wire [2:0]                                          axi_gmem0_awprot;
  wire                                                axi_gmem0_awvalid;
  wire                                                axi_gmem0_awready;
  wire [AXI_RAM_GMEM0_DATA_WIDTH-1:0]   axi_gmem0_wdata;
  wire [AXI_RAM_GMEM0_STRB_WIDTH-1:0]   axi_gmem0_wstrb;
  wire                                                axi_gmem0_wlast;
  wire                                                axi_gmem0_wvalid;
  wire                                                axi_gmem0_wready;
  wire [AXI_RAM_GMEM0_ID_WIDTH-1:0]     axi_gmem0_bid;
  wire [1:0]                                          axi_gmem0_bresp;
  wire                                                axi_gmem0_bvalid;
  wire                                                axi_gmem0_bready;
  wire [AXI_RAM_GMEM0_ID_WIDTH-1:0]     axi_gmem0_arid;
  wire [AXI_RAM_GMEM0_ADDR_WIDTH-1:0]   axi_gmem0_araddr;
  wire [7:0]                                          axi_gmem0_arlen;
  wire [2:0]                                          axi_gmem0_arsize;
  wire [1:0]                                          axi_gmem0_arburst;
  wire                                                axi_gmem0_arlock;
  wire [3:0]                                          axi_gmem0_arcache;
  wire [2:0]                                          axi_gmem0_arprot;
  wire                                                axi_gmem0_arvalid;
  wire                                                axi_gmem0_arready;
  wire [AXI_RAM_GMEM0_ID_WIDTH-1:0]     axi_gmem0_rid;
  wire [AXI_RAM_GMEM0_DATA_WIDTH-1:0]   axi_gmem0_rdata;
  wire [1:0]                                          axi_gmem0_rresp;
  wire                                                axi_gmem0_rlast;
  wire                                                axi_gmem0_rvalid;
  wire                                                axi_gmem0_rready;

  reg                                                 axi_ram_gmem0_dump_mem; // write out the contents into binary file
  wire [3:0]                                          axi_gmem0_awqos; // place holder
  wire [3:0]                                          axi_gmem0_arqos;

  axi_ram_gmem0 #
  (
    .DATA_WIDTH       (AXI_RAM_GMEM0_DATA_WIDTH),
    .ADDR_WIDTH       (AXI_RAM_GMEM0_ADDR_WIDTH),
    .STRB_WIDTH       (AXI_RAM_GMEM0_DATA_WIDTH/8),
    .ID_WIDTH         (AXI_RAM_GMEM0_ID_WIDTH),
    .PIPELINE_OUTPUT  (AXI_RAM_GMEM0_PIPELINE_OUTPUT)
  ) axi_ram_gmem0_unit
  (
      .clk           (ap_clk),
      .rst           (~ap_rst_n),

      .s_axi_awid    (axi_gmem0_awid    ),
      .s_axi_awaddr  (axi_gmem0_awaddr  ),
      .s_axi_awlen   (axi_gmem0_awlen   ),
      .s_axi_awsize  (axi_gmem0_awsize  ),
      .s_axi_awburst (axi_gmem0_awburst ),
      .s_axi_awlock  (axi_gmem0_awlock  ),
      .s_axi_awcache (axi_gmem0_awcache ),
      .s_axi_awprot  (axi_gmem0_awprot  ),
      .s_axi_awvalid (axi_gmem0_awvalid ),
      .s_axi_awready (axi_gmem0_awready ),
      .s_axi_wdata   (axi_gmem0_wdata   ),
      .s_axi_wstrb   (axi_gmem0_wstrb   ),
      .s_axi_wlast   (axi_gmem0_wlast   ),
      .s_axi_wvalid  (axi_gmem0_wvalid  ),
      .s_axi_wready  (axi_gmem0_wready  ),
      .s_axi_bid     (axi_gmem0_bid     ),
      .s_axi_bresp   (axi_gmem0_bresp   ),
      .s_axi_bvalid  (axi_gmem0_bvalid  ),
      .s_axi_bready  (axi_gmem0_bready  ),
      .s_axi_arid    (axi_gmem0_arid    ),
      .s_axi_araddr  (axi_gmem0_araddr  ),
      .s_axi_arlen   (axi_gmem0_arlen   ),
      .s_axi_arsize  (axi_gmem0_arsize  ),
      .s_axi_arburst (axi_gmem0_arburst ),
      .s_axi_arlock  (axi_gmem0_arlock  ),
      .s_axi_arcache (axi_gmem0_arcache ),
      .s_axi_arprot  (axi_gmem0_arprot  ),
      .s_axi_arvalid (axi_gmem0_arvalid ),
      .s_axi_arready (axi_gmem0_arready ),
      .s_axi_rid     (axi_gmem0_rid     ),
      .s_axi_rdata   (axi_gmem0_rdata   ),
      .s_axi_rresp   (axi_gmem0_rresp   ),
      .s_axi_rlast   (axi_gmem0_rlast   ),
      .s_axi_rvalid  (axi_gmem0_rvalid  ),
      .s_axi_rready  (axi_gmem0_rready  ),

      .dump_mem      (axi_ram_gmem0_dump_mem)
  );



  parameter AXI_RAM_GMEM1_DATA_WIDTH = 512;
  parameter AXI_RAM_GMEM1_ADDR_WIDTH = 22;
  parameter AXI_RAM_GMEM1_STRB_WIDTH = AXI_RAM_GMEM1_DATA_WIDTH/8;
  parameter AXI_RAM_GMEM1_ID_WIDTH = 8;
  parameter AXI_RAM_GMEM1_PIPELINE_OUTPUT = 0;

  wire [AXI_RAM_GMEM1_ID_WIDTH-1:0]     axi_gmem1_awid;
  wire [AXI_RAM_GMEM1_ADDR_WIDTH-1:0]   axi_gmem1_awaddr;
  wire [7:0]                                          axi_gmem1_awlen;
  wire [2:0]                                          axi_gmem1_awsize;
  wire [1:0]                                          axi_gmem1_awburst;
  wire                                                axi_gmem1_awlock;
  wire [3:0]                                          axi_gmem1_awcache;
  wire [2:0]                                          axi_gmem1_awprot;
  wire                                                axi_gmem1_awvalid;
  wire                                                axi_gmem1_awready;
  wire [AXI_RAM_GMEM1_DATA_WIDTH-1:0]   axi_gmem1_wdata;
  wire [AXI_RAM_GMEM1_STRB_WIDTH-1:0]   axi_gmem1_wstrb;
  wire                                                axi_gmem1_wlast;
  wire                                                axi_gmem1_wvalid;
  wire                                                axi_gmem1_wready;
  wire [AXI_RAM_GMEM1_ID_WIDTH-1:0]     axi_gmem1_bid;
  wire [1:0]                                          axi_gmem1_bresp;
  wire                                                axi_gmem1_bvalid;
  wire                                                axi_gmem1_bready;
  wire [AXI_RAM_GMEM1_ID_WIDTH-1:0]     axi_gmem1_arid;
  wire [AXI_RAM_GMEM1_ADDR_WIDTH-1:0]   axi_gmem1_araddr;
  wire [7:0]                                          axi_gmem1_arlen;
  wire [2:0]                                          axi_gmem1_arsize;
  wire [1:0]                                          axi_gmem1_arburst;
  wire                                                axi_gmem1_arlock;
  wire [3:0]                                          axi_gmem1_arcache;
  wire [2:0]                                          axi_gmem1_arprot;
  wire                                                axi_gmem1_arvalid;
  wire                                                axi_gmem1_arready;
  wire [AXI_RAM_GMEM1_ID_WIDTH-1:0]     axi_gmem1_rid;
  wire [AXI_RAM_GMEM1_DATA_WIDTH-1:0]   axi_gmem1_rdata;
  wire [1:0]                                          axi_gmem1_rresp;
  wire                                                axi_gmem1_rlast;
  wire                                                axi_gmem1_rvalid;
  wire                                                axi_gmem1_rready;

  reg                                                 axi_ram_gmem1_dump_mem; // write out the contents into binary file
  wire [3:0]                                          axi_gmem1_awqos; // place holder
  wire [3:0]                                          axi_gmem1_arqos;

  axi_ram_gmem1 #
  (
    .DATA_WIDTH       (AXI_RAM_GMEM1_DATA_WIDTH),
    .ADDR_WIDTH       (AXI_RAM_GMEM1_ADDR_WIDTH),
    .STRB_WIDTH       (AXI_RAM_GMEM1_DATA_WIDTH/8),
    .ID_WIDTH         (AXI_RAM_GMEM1_ID_WIDTH),
    .PIPELINE_OUTPUT  (AXI_RAM_GMEM1_PIPELINE_OUTPUT)
  ) axi_ram_gmem1_unit
  (
      .clk           (ap_clk),
      .rst           (~ap_rst_n),

      .s_axi_awid    (axi_gmem1_awid    ),
      .s_axi_awaddr  (axi_gmem1_awaddr  ),
      .s_axi_awlen   (axi_gmem1_awlen   ),
      .s_axi_awsize  (axi_gmem1_awsize  ),
      .s_axi_awburst (axi_gmem1_awburst ),
      .s_axi_awlock  (axi_gmem1_awlock  ),
      .s_axi_awcache (axi_gmem1_awcache ),
      .s_axi_awprot  (axi_gmem1_awprot  ),
      .s_axi_awvalid (axi_gmem1_awvalid ),
      .s_axi_awready (axi_gmem1_awready ),
      .s_axi_wdata   (axi_gmem1_wdata   ),
      .s_axi_wstrb   (axi_gmem1_wstrb   ),
      .s_axi_wlast   (axi_gmem1_wlast   ),
      .s_axi_wvalid  (axi_gmem1_wvalid  ),
      .s_axi_wready  (axi_gmem1_wready  ),
      .s_axi_bid     (axi_gmem1_bid     ),
      .s_axi_bresp   (axi_gmem1_bresp   ),
      .s_axi_bvalid  (axi_gmem1_bvalid  ),
      .s_axi_bready  (axi_gmem1_bready  ),
      .s_axi_arid    (axi_gmem1_arid    ),
      .s_axi_araddr  (axi_gmem1_araddr  ),
      .s_axi_arlen   (axi_gmem1_arlen   ),
      .s_axi_arsize  (axi_gmem1_arsize  ),
      .s_axi_arburst (axi_gmem1_arburst ),
      .s_axi_arlock  (axi_gmem1_arlock  ),
      .s_axi_arcache (axi_gmem1_arcache ),
      .s_axi_arprot  (axi_gmem1_arprot  ),
      .s_axi_arvalid (axi_gmem1_arvalid ),
      .s_axi_arready (axi_gmem1_arready ),
      .s_axi_rid     (axi_gmem1_rid     ),
      .s_axi_rdata   (axi_gmem1_rdata   ),
      .s_axi_rresp   (axi_gmem1_rresp   ),
      .s_axi_rlast   (axi_gmem1_rlast   ),
      .s_axi_rvalid  (axi_gmem1_rvalid  ),
      .s_axi_rready  (axi_gmem1_rready  ),

      .dump_mem      (axi_ram_gmem1_dump_mem)
  );



  parameter AXI_RAM_GMEM2_DATA_WIDTH = 512;
  parameter AXI_RAM_GMEM2_ADDR_WIDTH = 22;
  parameter AXI_RAM_GMEM2_STRB_WIDTH = AXI_RAM_GMEM2_DATA_WIDTH/8;
  parameter AXI_RAM_GMEM2_ID_WIDTH = 8;
  parameter AXI_RAM_GMEM2_PIPELINE_OUTPUT = 0;

  wire [AXI_RAM_GMEM2_ID_WIDTH-1:0]     axi_gmem2_awid;
  wire [AXI_RAM_GMEM2_ADDR_WIDTH-1:0]   axi_gmem2_awaddr;
  wire [7:0]                                          axi_gmem2_awlen;
  wire [2:0]                                          axi_gmem2_awsize;
  wire [1:0]                                          axi_gmem2_awburst;
  wire                                                axi_gmem2_awlock;
  wire [3:0]                                          axi_gmem2_awcache;
  wire [2:0]                                          axi_gmem2_awprot;
  wire                                                axi_gmem2_awvalid;
  wire                                                axi_gmem2_awready;
  wire [AXI_RAM_GMEM2_DATA_WIDTH-1:0]   axi_gmem2_wdata;
  wire [AXI_RAM_GMEM2_STRB_WIDTH-1:0]   axi_gmem2_wstrb;
  wire                                                axi_gmem2_wlast;
  wire                                                axi_gmem2_wvalid;
  wire                                                axi_gmem2_wready;
  wire [AXI_RAM_GMEM2_ID_WIDTH-1:0]     axi_gmem2_bid;
  wire [1:0]                                          axi_gmem2_bresp;
  wire                                                axi_gmem2_bvalid;
  wire                                                axi_gmem2_bready;
  wire [AXI_RAM_GMEM2_ID_WIDTH-1:0]     axi_gmem2_arid;
  wire [AXI_RAM_GMEM2_ADDR_WIDTH-1:0]   axi_gmem2_araddr;
  wire [7:0]                                          axi_gmem2_arlen;
  wire [2:0]                                          axi_gmem2_arsize;
  wire [1:0]                                          axi_gmem2_arburst;
  wire                                                axi_gmem2_arlock;
  wire [3:0]                                          axi_gmem2_arcache;
  wire [2:0]                                          axi_gmem2_arprot;
  wire                                                axi_gmem2_arvalid;
  wire                                                axi_gmem2_arready;
  wire [AXI_RAM_GMEM2_ID_WIDTH-1:0]     axi_gmem2_rid;
  wire [AXI_RAM_GMEM2_DATA_WIDTH-1:0]   axi_gmem2_rdata;
  wire [1:0]                                          axi_gmem2_rresp;
  wire                                                axi_gmem2_rlast;
  wire                                                axi_gmem2_rvalid;
  wire                                                axi_gmem2_rready;

  reg                                                 axi_ram_gmem2_dump_mem; // write out the contents into binary file
  wire [3:0]                                          axi_gmem2_awqos; // place holder
  wire [3:0]                                          axi_gmem2_arqos;

  axi_ram_gmem2 #
  (
    .DATA_WIDTH       (AXI_RAM_GMEM2_DATA_WIDTH),
    .ADDR_WIDTH       (AXI_RAM_GMEM2_ADDR_WIDTH),
    .STRB_WIDTH       (AXI_RAM_GMEM2_DATA_WIDTH/8),
    .ID_WIDTH         (AXI_RAM_GMEM2_ID_WIDTH),
    .PIPELINE_OUTPUT  (AXI_RAM_GMEM2_PIPELINE_OUTPUT)
  ) axi_ram_gmem2_unit
  (
      .clk           (ap_clk),
      .rst           (~ap_rst_n),

      .s_axi_awid    (axi_gmem2_awid    ),
      .s_axi_awaddr  (axi_gmem2_awaddr  ),
      .s_axi_awlen   (axi_gmem2_awlen   ),
      .s_axi_awsize  (axi_gmem2_awsize  ),
      .s_axi_awburst (axi_gmem2_awburst ),
      .s_axi_awlock  (axi_gmem2_awlock  ),
      .s_axi_awcache (axi_gmem2_awcache ),
      .s_axi_awprot  (axi_gmem2_awprot  ),
      .s_axi_awvalid (axi_gmem2_awvalid ),
      .s_axi_awready (axi_gmem2_awready ),
      .s_axi_wdata   (axi_gmem2_wdata   ),
      .s_axi_wstrb   (axi_gmem2_wstrb   ),
      .s_axi_wlast   (axi_gmem2_wlast   ),
      .s_axi_wvalid  (axi_gmem2_wvalid  ),
      .s_axi_wready  (axi_gmem2_wready  ),
      .s_axi_bid     (axi_gmem2_bid     ),
      .s_axi_bresp   (axi_gmem2_bresp   ),
      .s_axi_bvalid  (axi_gmem2_bvalid  ),
      .s_axi_bready  (axi_gmem2_bready  ),
      .s_axi_arid    (axi_gmem2_arid    ),
      .s_axi_araddr  (axi_gmem2_araddr  ),
      .s_axi_arlen   (axi_gmem2_arlen   ),
      .s_axi_arsize  (axi_gmem2_arsize  ),
      .s_axi_arburst (axi_gmem2_arburst ),
      .s_axi_arlock  (axi_gmem2_arlock  ),
      .s_axi_arcache (axi_gmem2_arcache ),
      .s_axi_arprot  (axi_gmem2_arprot  ),
      .s_axi_arvalid (axi_gmem2_arvalid ),
      .s_axi_arready (axi_gmem2_arready ),
      .s_axi_rid     (axi_gmem2_rid     ),
      .s_axi_rdata   (axi_gmem2_rdata   ),
      .s_axi_rresp   (axi_gmem2_rresp   ),
      .s_axi_rlast   (axi_gmem2_rlast   ),
      .s_axi_rvalid  (axi_gmem2_rvalid  ),
      .s_axi_rready  (axi_gmem2_rready  ),

      .dump_mem      (axi_ram_gmem2_dump_mem)
  );



  parameter C_S_AXI_CONTROL_DATA_WIDTH = 32;
  parameter C_S_AXI_CONTROL_ADDR_WIDTH = 64;
  parameter C_S_AXI_DATA_WIDTH = 32;
  parameter C_S_AXI_CONTROL_WSTRB_WIDTH = 32 / 8;
  parameter C_S_AXI_WSTRB_WIDTH = 32 / 8;

  wire                                    s_axi_control_awvalid;
  wire                                    s_axi_control_awready;
  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_awaddr;
  wire                                    s_axi_control_wvalid;
  wire                                    s_axi_control_wready;
  wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_wdata;
  wire [C_S_AXI_CONTROL_WSTRB_WIDTH-1:0]  s_axi_control_wstrb;
  reg                                     s_axi_control_arvalid;
  wire                                    s_axi_control_arready;
  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_araddr;
  wire                                    s_axi_control_rvalid;
  wire                                    s_axi_control_rready;
  wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_rdata;
  wire [1:0]                              s_axi_control_rresp;
  wire                                    s_axi_control_bvalid;
  wire                                    s_axi_control_bready;
  wire [1:0]                              s_axi_control_bresp;

  // use a large FIFO to buffer the command to s_axi_control
  // in this way we don't need to worry about flow control
  reg [63:0] s_axi_aw_din;
  reg        s_axi_aw_write;

  fifo_srl #(
    .DATA_WIDTH(64),
    .ADDR_WIDTH(6),
    .DEPTH(64)
  ) s_axi_aw(
    .clk(ap_clk),
    .reset(~ap_rst_n),

    // write
    .if_full_n   (),
    .if_write    (s_axi_aw_write),
    .if_din      (s_axi_aw_din  ),

    // read
    .if_empty_n  (s_axi_control_awvalid),
    .if_read     (s_axi_control_awready),
    .if_dout     (s_axi_control_awaddr),

    .if_write_ce (1'b1),
    .if_read_ce  (1'b1)

  );

  reg [511:0] s_axi_w_din;
  reg         s_axi_w_write;

  fifo_srl #(
    .DATA_WIDTH(32),
    .ADDR_WIDTH(6),
    .DEPTH(64)
  ) s_axi_w(
    .clk(ap_clk),
    .reset(~ap_rst_n),

    // write
    .if_full_n   (),
    .if_write    (s_axi_w_write),
    .if_din      (s_axi_w_din  ),

    // read
    .if_empty_n  (s_axi_control_wvalid),
    .if_read     (s_axi_control_wready),
    .if_dout     (s_axi_control_wdata),

    .if_write_ce (1'b1),
    .if_read_ce  (1'b1)

  );


  autosa_mm dut (
    .s_axi_control_AWVALID (s_axi_control_awvalid),
    .s_axi_control_AWREADY (s_axi_control_awready),
    .s_axi_control_AWADDR  (s_axi_control_awaddr ),

    .s_axi_control_WVALID  (s_axi_control_wvalid ),
    .s_axi_control_WREADY  (s_axi_control_wready ),
    .s_axi_control_WDATA   (s_axi_control_wdata  ),
    .s_axi_control_WSTRB   ({64{1'b1} }      ),

    // keep polling the control registers
    .s_axi_control_ARVALID (s_axi_control_arvalid),
    .s_axi_control_ARREADY (s_axi_control_arready),
    .s_axi_control_ARADDR  ('h00 ),

    .s_axi_control_RVALID  (s_axi_control_rvalid ),
    .s_axi_control_RREADY  (1 ),
    .s_axi_control_RDATA   (s_axi_control_rdata  ),
    .s_axi_control_RRESP   (                     ),

    .s_axi_control_BVALID  (s_axi_control_bvalid ),
    .s_axi_control_BREADY  (1'b1                 ),
    .s_axi_control_BRESP   (s_axi_control_bresp  ),

    .m_axi_gmem0_ARADDR  (axi_gmem0_araddr ),
    .m_axi_gmem0_ARBURST (axi_gmem0_arburst),
    .m_axi_gmem0_ARCACHE (axi_gmem0_arcache),
    .m_axi_gmem0_ARID    (axi_gmem0_arid   ),
    .m_axi_gmem0_ARLEN   (axi_gmem0_arlen  ),
    .m_axi_gmem0_ARLOCK  (axi_gmem0_arlock ),
    .m_axi_gmem0_ARPROT  (axi_gmem0_arprot ),
    .m_axi_gmem0_ARQOS   (axi_gmem0_arqos  ),
    .m_axi_gmem0_ARREADY (axi_gmem0_arready),
    .m_axi_gmem0_ARSIZE  (axi_gmem0_arsize ),
    .m_axi_gmem0_ARVALID (axi_gmem0_arvalid),
    .m_axi_gmem0_AWADDR  (axi_gmem0_awaddr ),
    .m_axi_gmem0_AWBURST (axi_gmem0_awburst),
    .m_axi_gmem0_AWCACHE (axi_gmem0_awcache),
    .m_axi_gmem0_AWID    (axi_gmem0_awid   ),
    .m_axi_gmem0_AWLEN   (axi_gmem0_awlen  ),
    .m_axi_gmem0_AWLOCK  (axi_gmem0_awlock ),
    .m_axi_gmem0_AWPROT  (axi_gmem0_awprot ),
    .m_axi_gmem0_AWQOS   (axi_gmem0_awqos  ),
    .m_axi_gmem0_AWREADY (axi_gmem0_awready),
    .m_axi_gmem0_AWSIZE  (axi_gmem0_awsize ),
    .m_axi_gmem0_AWVALID (axi_gmem0_awvalid),
    .m_axi_gmem0_BID     (axi_gmem0_bid    ),
    .m_axi_gmem0_BREADY  (axi_gmem0_bready ),
    .m_axi_gmem0_BRESP   (axi_gmem0_bresp  ),
    .m_axi_gmem0_BVALID  (axi_gmem0_bvalid ),
    .m_axi_gmem0_RDATA   (axi_gmem0_rdata  ),
    .m_axi_gmem0_RID     (axi_gmem0_rid    ),
    .m_axi_gmem0_RLAST   (axi_gmem0_rlast  ),
    .m_axi_gmem0_RREADY  (axi_gmem0_rready ),
    .m_axi_gmem0_RRESP   (axi_gmem0_rresp  ),
    .m_axi_gmem0_RVALID  (axi_gmem0_rvalid ),
    .m_axi_gmem0_WDATA   (axi_gmem0_wdata  ),
    .m_axi_gmem0_WLAST   (axi_gmem0_wlast  ),
    .m_axi_gmem0_WREADY  (axi_gmem0_wready ),
    .m_axi_gmem0_WSTRB   (axi_gmem0_wstrb  ),
    .m_axi_gmem0_WVALID  (axi_gmem0_wvalid ),

    .m_axi_gmem1_ARADDR  (axi_gmem1_araddr ),
    .m_axi_gmem1_ARBURST (axi_gmem1_arburst),
    .m_axi_gmem1_ARCACHE (axi_gmem1_arcache),
    .m_axi_gmem1_ARID    (axi_gmem1_arid   ),
    .m_axi_gmem1_ARLEN   (axi_gmem1_arlen  ),
    .m_axi_gmem1_ARLOCK  (axi_gmem1_arlock ),
    .m_axi_gmem1_ARPROT  (axi_gmem1_arprot ),
    .m_axi_gmem1_ARQOS   (axi_gmem1_arqos  ),
    .m_axi_gmem1_ARREADY (axi_gmem1_arready),
    .m_axi_gmem1_ARSIZE  (axi_gmem1_arsize ),
    .m_axi_gmem1_ARVALID (axi_gmem1_arvalid),
    .m_axi_gmem1_AWADDR  (axi_gmem1_awaddr ),
    .m_axi_gmem1_AWBURST (axi_gmem1_awburst),
    .m_axi_gmem1_AWCACHE (axi_gmem1_awcache),
    .m_axi_gmem1_AWID    (axi_gmem1_awid   ),
    .m_axi_gmem1_AWLEN   (axi_gmem1_awlen  ),
    .m_axi_gmem1_AWLOCK  (axi_gmem1_awlock ),
    .m_axi_gmem1_AWPROT  (axi_gmem1_awprot ),
    .m_axi_gmem1_AWQOS   (axi_gmem1_awqos  ),
    .m_axi_gmem1_AWREADY (axi_gmem1_awready),
    .m_axi_gmem1_AWSIZE  (axi_gmem1_awsize ),
    .m_axi_gmem1_AWVALID (axi_gmem1_awvalid),
    .m_axi_gmem1_BID     (axi_gmem1_bid    ),
    .m_axi_gmem1_BREADY  (axi_gmem1_bready ),
    .m_axi_gmem1_BRESP   (axi_gmem1_bresp  ),
    .m_axi_gmem1_BVALID  (axi_gmem1_bvalid ),
    .m_axi_gmem1_RDATA   (axi_gmem1_rdata  ),
    .m_axi_gmem1_RID     (axi_gmem1_rid    ),
    .m_axi_gmem1_RLAST   (axi_gmem1_rlast  ),
    .m_axi_gmem1_RREADY  (axi_gmem1_rready ),
    .m_axi_gmem1_RRESP   (axi_gmem1_rresp  ),
    .m_axi_gmem1_RVALID  (axi_gmem1_rvalid ),
    .m_axi_gmem1_WDATA   (axi_gmem1_wdata  ),
    .m_axi_gmem1_WLAST   (axi_gmem1_wlast  ),
    .m_axi_gmem1_WREADY  (axi_gmem1_wready ),
    .m_axi_gmem1_WSTRB   (axi_gmem1_wstrb  ),
    .m_axi_gmem1_WVALID  (axi_gmem1_wvalid ),

    .m_axi_gmem2_ARADDR  (axi_gmem2_araddr ),
    .m_axi_gmem2_ARBURST (axi_gmem2_arburst),
    .m_axi_gmem2_ARCACHE (axi_gmem2_arcache),
    .m_axi_gmem2_ARID    (axi_gmem2_arid   ),
    .m_axi_gmem2_ARLEN   (axi_gmem2_arlen  ),
    .m_axi_gmem2_ARLOCK  (axi_gmem2_arlock ),
    .m_axi_gmem2_ARPROT  (axi_gmem2_arprot ),
    .m_axi_gmem2_ARQOS   (axi_gmem2_arqos  ),
    .m_axi_gmem2_ARREADY (axi_gmem2_arready),
    .m_axi_gmem2_ARSIZE  (axi_gmem2_arsize ),
    .m_axi_gmem2_ARVALID (axi_gmem2_arvalid),
    .m_axi_gmem2_AWADDR  (axi_gmem2_awaddr ),
    .m_axi_gmem2_AWBURST (axi_gmem2_awburst),
    .m_axi_gmem2_AWCACHE (axi_gmem2_awcache),
    .m_axi_gmem2_AWID    (axi_gmem2_awid   ),
    .m_axi_gmem2_AWLEN   (axi_gmem2_awlen  ),
    .m_axi_gmem2_AWLOCK  (axi_gmem2_awlock ),
    .m_axi_gmem2_AWPROT  (axi_gmem2_awprot ),
    .m_axi_gmem2_AWQOS   (axi_gmem2_awqos  ),
    .m_axi_gmem2_AWREADY (axi_gmem2_awready),
    .m_axi_gmem2_AWSIZE  (axi_gmem2_awsize ),
    .m_axi_gmem2_AWVALID (axi_gmem2_awvalid),
    .m_axi_gmem2_BID     (axi_gmem2_bid    ),
    .m_axi_gmem2_BREADY  (axi_gmem2_bready ),
    .m_axi_gmem2_BRESP   (axi_gmem2_bresp  ),
    .m_axi_gmem2_BVALID  (axi_gmem2_bvalid ),
    .m_axi_gmem2_RDATA   (axi_gmem2_rdata  ),
    .m_axi_gmem2_RID     (axi_gmem2_rid    ),
    .m_axi_gmem2_RLAST   (axi_gmem2_rlast  ),
    .m_axi_gmem2_RREADY  (axi_gmem2_rready ),
    .m_axi_gmem2_RRESP   (axi_gmem2_rresp  ),
    .m_axi_gmem2_RVALID  (axi_gmem2_rvalid ),
    .m_axi_gmem2_WDATA   (axi_gmem2_wdata  ),
    .m_axi_gmem2_WLAST   (axi_gmem2_wlast  ),
    .m_axi_gmem2_WREADY  (axi_gmem2_wready ),
    .m_axi_gmem2_WSTRB   (axi_gmem2_wstrb  ),
    .m_axi_gmem2_WVALID  (axi_gmem2_wvalid ),

    .ap_clk          (ap_clk       ),
    .ap_rst_n        (ap_rst_n     ),
    .interrupt       (             )
  );
  

  parameter HALF_CLOCK_PERIOD = 2;
  parameter CLOCK_PERIOD = HALF_CLOCK_PERIOD * 2;

  // clock
  always begin
      ap_clk = 1'b1;
      #HALF_CLOCK_PERIOD;
      ap_clk = 1'b0;
      #HALF_CLOCK_PERIOD;
  end

  wire [31:0] REG_MASK_32_BIT = {32{1'b1}};
  initial begin
    s_axi_aw_write = 1'b0;
    s_axi_aw_din = 1'b0;
    s_axi_w_write = 1'b0;
    s_axi_w_din = 1'b0;
    axi_ram_gmem0_dump_mem = 1'b0;
    axi_ram_gmem1_dump_mem = 1'b0;
    axi_ram_gmem2_dump_mem = 1'b0;
    s_axi_control_arvalid = 1'b0;
    ap_rst_n = 1'b0;

    #(CLOCK_PERIOD*1000);
    ap_rst_n = 1'b1;
    #(CLOCK_PERIOD*100);

    s_axi_aw_write = 1; s_axi_aw_din = 'h10; s_axi_w_write = 1; s_axi_w_din = 0 & REG_MASK_32_BIT; #CLOCK_PERIOD;
    s_axi_aw_write = 1; s_axi_aw_din = 'h14; s_axi_w_write = 1; s_axi_w_din = (0 >> 32) & REG_MASK_32_BIT; #CLOCK_PERIOD;
    s_axi_aw_write = 1; s_axi_aw_din = 'h1c; s_axi_w_write = 1; s_axi_w_din = 0 & REG_MASK_32_BIT; #CLOCK_PERIOD;
    s_axi_aw_write = 1; s_axi_aw_din = 'h20; s_axi_w_write = 1; s_axi_w_din = (0 >> 32) & REG_MASK_32_BIT; #CLOCK_PERIOD;
    s_axi_aw_write = 1; s_axi_aw_din = 'h28; s_axi_w_write = 1; s_axi_w_din = 0 & REG_MASK_32_BIT; #CLOCK_PERIOD;
    s_axi_aw_write = 1; s_axi_aw_din = 'h2c; s_axi_w_write = 1; s_axi_w_din = (0 >> 32) & REG_MASK_32_BIT; #CLOCK_PERIOD;

    // start the kernel
    s_axi_aw_write = 1; s_axi_aw_din = 'h00; s_axi_w_write = 1; s_axi_w_din = 1; #CLOCK_PERIOD;
    $display("FASTCOSIM_START %0d cycle=%0d", $time, fastcosim_cycles);  // PATCH 3/5

    // stop writing control signal
    s_axi_aw_write = 0; s_axi_w_write = 0; #CLOCK_PERIOD;

    // start polling ap_done
    #(CLOCK_PERIOD*1000);
    s_axi_control_arvalid = 1'b1 & s_axi_control_arready;

  end

  // polling on ap_done
  always @ (posedge ap_clk) begin
    if (~ap_rst_n) begin
    end
    else begin
      if (s_axi_control_rvalid) begin
        if (s_axi_control_rdata[1]) begin
          $display("FASTCOSIM_DONE %0d cycle=%0d", $time, fastcosim_cycles);  // PATCH 3/5
          axi_ram_gmem0_dump_mem = 1;
          axi_ram_gmem1_dump_mem = 1;
          axi_ram_gmem2_dump_mem = 1;
          #(CLOCK_PERIOD*100)
          $finish;
        end
      end
    end
  end

  always @(negedge ap_clk) begin  // counter is declared and incremented in get_begin (PATCH 5)
    if (fastcosim_cycles % 10000 == 0) $display("FASTCOSIM_HEARTBEAT cycles=%0d", fastcosim_cycles);
  end


endmodule

