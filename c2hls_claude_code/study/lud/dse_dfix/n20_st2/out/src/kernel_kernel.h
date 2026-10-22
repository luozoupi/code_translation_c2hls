#include <ap_int.h>
#include <hls_stream.h>

#define min(x,y) ((x < y) ? x : y)
#define max(x,y) ((x > y) ? x : y)

/* Data Type */
typedef float prev_V_t1;
typedef float V_t1;
typedef float U_tmp_t1;
typedef float L_tmp_t1;
typedef float A_t1;
typedef float L_t1;
typedef float U_t1;
typedef ap_uint<128> prev_V_t4;
typedef ap_uint<128> U_tmp_t4;
typedef ap_uint<128> A_t4;
/* Data Type */

void kernel0(A_t4 *A, L_t1 *L, U_t1 *U);
template<int p0>
void A_IO_L1_in_intra_trans(A_t4 local_A[1][5], hls::stream<float> &fifo_A_local_out);
template<int p0>
void A_IO_L1_in_inter_trans(A_t4 local_A[1][5], hls::stream<A_t4> &fifo_A_in, hls::stream<A_t4> &fifo_A_out);
template<int p0>
void A_IO_L1_in_inter_trans_boundary(A_t4 local_A[1][5], hls::stream<A_t4> &fifo_A_in);
template<int p0>
void A_IO_L1_in_wrapper(hls::stream<A_t4> &fifo_A_in, hls::stream<A_t4> &fifo_A_out, hls::stream<float> &fifo_A_local_out);
template<int p0>
void A_IO_L1_in_boundary_wrapper(hls::stream<A_t4> &fifo_A_in, hls::stream<float> &fifo_A_local_out);
template<int p0>
void PE_wrapper(hls::stream<float> &fifo_L_tmp_0_in, hls::stream<float> &fifo_L_tmp_0_out, hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_U_drain_out);
template<int p0>
void L_drain_IO_L1_out_intra_trans(float local_L[1][19], hls::stream<float> &fifo_L_drain_local_in);
template<int p0>
void L_drain_IO_L1_out_inter_trans(float local_L[1][19], hls::stream<float> &fifo_L_drain_in, hls::stream<float> &fifo_L_drain_out);
template<int p0>
void L_drain_IO_L1_out_inter_trans_boundary(float local_L[1][19], hls::stream<float> &fifo_L_drain_out);
template<int p0>
void L_drain_IO_L1_out_wrapper(hls::stream<float> &fifo_L_drain_in, hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_L_drain_local_in);
template<int p0>
void L_drain_IO_L1_out_boundary_wrapper(hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_L_drain_local_in);
template<int p0>
void U_drain_IO_L1_out_intra_trans(float local_U[20][1], hls::stream<float> &fifo_U_drain_local_in);
template<int p0>
void U_drain_IO_L1_out_inter_trans(float local_U[20][1], hls::stream<float> &fifo_U_drain_in, hls::stream<float> &fifo_U_drain_out);
template<int p0>
void U_drain_IO_L1_out_inter_trans_boundary(float local_U[20][1], hls::stream<float> &fifo_U_drain_out);
template<int p0>
void U_drain_IO_L1_out_wrapper(hls::stream<float> &fifo_U_drain_in, hls::stream<float> &fifo_U_drain_out, hls::stream<float> &fifo_U_drain_local_in);
template<int p0>
void U_drain_IO_L1_out_boundary_wrapper(hls::stream<float> &fifo_U_drain_out, hls::stream<float> &fifo_U_drain_local_in);
