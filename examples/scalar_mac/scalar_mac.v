// SPDX-License-Identifier: MIT
// Copyright (c) 2026 luozoupi
// Hand-authored fixture; not a synthesis result for a C++ candidate.
module scalar_mac (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [31:0] x,
    input wire [31:0] y,
    input wire [31:0] bias,
    output reg done,
    output reg [31:0] result
);
    wire [63:0] product = x * y;

    always @(posedge clk) begin
        if (!rst_n) begin
            done <= 1'b0;
            result <= 32'd0;
        end else begin
            done <= start;
            if (start)
                result <= product[31:0] + bias;
        end
    end
endmodule
