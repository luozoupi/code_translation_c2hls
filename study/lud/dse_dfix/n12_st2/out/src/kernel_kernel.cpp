#include <cstdio>
#include <cstdlib>
#include "kernel_kernel.h"
/* Module Definition */
void A_IO_L2_in(A_t4 *A, hls::stream<A_t4> &fifo_A_local_out) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  /* Variable Declaration */

  // array
  // io_L2
  for (ap_uint<5> c0 = 0; c0 <= 11; c0 += 1) {
    // io_L1
    // access_coalesce
    for (ap_uint<3> c2 = 0; c2 <= 2; c2 += 1) {
    #pragma HLS PIPELINE II=1
      {
        A_t4 in_data;
        A_t4 out_data;
        in_data = A[3*c0 + c2];
        out_data = in_data;
        fifo_A_local_out.write(out_data);
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0>
void A_IO_L1_in_intra_trans(A_t4 local_A[1][3], hls::stream<float> &fifo_A_local_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  ap_uint<32> data_split[4];
  #pragma HLS ARRAY_PARTITION variable=data_split complete
  /* Variable Declaration */


  // array
  // io_L2
  // io_L1
  // pe
  for (ap_uint<5> c2 = 0; c2 <= 11; c2 += 1) {
  #pragma HLS PIPELINE II=1
    // hls_pipeline
    {
      A_t4 in_data;
      A_t1 out_data;
      in_data = local_A[0][c2 / 4];
      for (ap_uint<3> n = 0; n < 4; n++) {
      #pragma HLS UNROLL
        data_split[n] = in_data(31, 0);
        in_data = in_data >> 32;
      }
      int split_idx = (c2) % 4;
      union {unsigned int ui; float ut;} u;
      u.ui = (unsigned int)data_split[split_idx];
      out_data = u.ut;
      fifo_A_local_out.write(out_data);
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0>
void A_IO_L1_in_inter_trans(A_t4 local_A[1][3], hls::stream<A_t4> &fifo_A_in, hls::stream<A_t4> &fifo_A_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */

  for (ap_uint<5> c0 = p0; c0 <= 11; c0 += 1) {
    // io_L1
    if (c0 == p0) {
      // access_coalesce
      for (ap_uint<3> c2 = 0; c2 <= 2; c2 += 1) {
      #pragma HLS PIPELINE II=1
        {
          A_t4 in_data;
          A_t4 out_data;
          in_data = fifo_A_in.read();
          out_data = in_data;
          local_A[0][c2] = out_data;
        }
      }
    } else {
      // access_coalesce
      for (ap_uint<3> c2 = 0; c2 <= 2; c2 += 1) {
      #pragma HLS PIPELINE II=1
        {
          A_t4 in_data;
          A_t4 out_data;
          in_data = fifo_A_in.read();
          out_data = in_data;
          fifo_A_out.write(out_data);
        }
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0>
void A_IO_L1_in_inter_trans_boundary(A_t4 local_A[1][3], hls::stream<A_t4> &fifo_A_in) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */

  for (ap_uint<5> c0 = p0; c0 <= 11; c0 += 1)
    if (c0 == p0) {
      // io_L1
      // access_coalesce
      for (ap_uint<3> c2 = 0; c2 <= 2; c2 += 1) {
      #pragma HLS PIPELINE II=1
        {
          A_t4 in_data;
          A_t4 out_data;
          in_data = fifo_A_in.read();
          out_data = in_data;
          local_A[0][c2] = out_data;
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
template<int p0>
void A_IO_L1_in(hls::stream<A_t4> &fifo_A_in, hls::stream<A_t4> &fifo_A_out, hls::stream<float> &fifo_A_local_out) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  A_t4 local_A[1][3];
  #pragma HLS RESOURCE variable=local_A core=RAM_2P_BRAM
  /* Variable Declaration */

  // array
  // io_L2
  A_IO_L1_in_inter_trans<p0>(
    /* array */ local_A, 
    /* fifo */ fifo_A_in, 
    /* fifo */ fifo_A_out
  );
  A_IO_L1_in_intra_trans<p0>(
    /* array */ local_A, 
    /* fifo */ fifo_A_local_out
  );
}
/* Module Definition */

/* Module Definition */
template<int p0>
void A_IO_L1_in_wrapper(hls::stream<A_t4> &fifo_A_in, hls::stream<A_t4> &fifo_A_out, hls::stream<float> &fifo_A_local_out)
 {
  A_IO_L1_in<p0>(
    /* fifo */ fifo_A_in, 
    /* fifo */ fifo_A_out, 
    /* fifo */ fifo_A_local_out);
}
/* Module Definition */

/* Module Definition */
template<int p0>
void A_IO_L1_in_boundary(hls::stream<A_t4> &fifo_A_in, hls::stream<float> &fifo_A_local_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  A_t4 local_A[1][3];
  #pragma HLS RESOURCE variable=local_A core=RAM_2P_BRAM
  /* Variable Declaration */

  // array
  // io_L2
  A_IO_L1_in_inter_trans_boundary<p0>(
    /* array */ local_A, 
    /* fifo */ fifo_A_in
  );
  A_IO_L1_in_intra_trans<p0>(
    /* array */ local_A, 
    /* fifo */ fifo_A_local_out
  );
}
/* Module Definition */

/* Module Definition */
template<int p0>
void A_IO_L1_in_boundary_wrapper(hls::stream<A_t4> &fifo_A_in, hls::stream<float> &fifo_A_local_out)
 {
  A_IO_L1_in_boundary<p0>(
    /* fifo */ fifo_A_in, 
    /* fifo */ fifo_A_local_out);
}
/* Module Definition */

/* Module Definition */
template<int p0>
void PE(hls::stream<float> &fifo_L_tmp_0_in, hls::stream<float> &fifo_L_tmp_0_out, hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_U_drain_out) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  prev_V_t1 local_prev_V[1][12];
  #pragma HLS RESOURCE variable=local_prev_V core=RAM_2P_BRAM
  V_t1 local_V[1][11];
  #pragma HLS RESOURCE variable=local_V core=RAM_2P_BRAM
  U_tmp_t1 local_U_tmp[1][12];
  #pragma HLS RESOURCE variable=local_U_tmp core=RAM_2P_BRAM
  L_tmp_t1 local_L_tmp[2][11];
  #pragma HLS RESOURCE variable=local_L_tmp core=RAM_2P_BRAM
  A_t1 local_A[1][1];
  #pragma HLS ARRAY_PARTITION variable=local_A dim=0 complete
  L_t1 local_L[1][1];
  #pragma HLS ARRAY_PARTITION variable=local_L dim=0 complete
  U_t1 local_U[1][1];
  #pragma HLS ARRAY_PARTITION variable=local_U dim=0 complete
  /* Variable Declaration */

  // array
  // pe
  for (int c1 = 0; c1 <= p0; c1 += 1)
    for (ap_uint<5> c2 = c1; c2 <= 11; c2 += 1) {
    #pragma HLS PIPELINE II=1
      {
        if (c1 == 0) {
          local_A[0][0] = fifo_A_in.read();
          local_prev_V[0][c2] = local_A[0][0];
        } else {
          local_prev_V[0][c2] = local_V[0][c2 - 1];
        }
        if (c2 == c1) {
          local_U_tmp[0][c1] = local_prev_V[0][c1];
          local_U[0][0] = local_U_tmp[0][c1];
          fifo_U_drain_out.write(local_U[0][0]);
        } else {
          local_U_tmp[0][c2] = local_U_tmp[0][c2 - 1];
          if (c1 == p0) {
            local_L_tmp[1][c2 - 1] = (local_prev_V[0][c2] / local_U_tmp[0][c2]);
            fifo_L_tmp_0_out.write(local_L_tmp[1][c2 - 1]);
            local_L[0][0] = local_L_tmp[1][c2 - 1];
            fifo_L_drain_out.write(local_L[0][0]);
          } else {
            local_L_tmp[0][c2 - 1] = fifo_L_tmp_0_in.read();
            local_L_tmp[1][c2 - 1] = local_L_tmp[0][c2 - 1];
            if (p0 <= 10)
              fifo_L_tmp_0_out.write(local_L_tmp[1][c2 - 1]);
          }
          local_V[0][c2 - 1] = (local_prev_V[0][c2] - (local_L_tmp[1][c2 - 1] * local_U_tmp[0][c2]));
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
template<int p0>
void PE_wrapper(hls::stream<float> &fifo_L_tmp_0_in, hls::stream<float> &fifo_L_tmp_0_out, hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_U_drain_out)
 {
  PE<p0>(
    /* fifo */ fifo_L_tmp_0_in, 
    /* fifo */ fifo_L_tmp_0_out, 
    /* fifo */ fifo_A_in, 
    /* fifo */ fifo_L_drain_out, 
    /* fifo */ fifo_U_drain_out);
}
/* Module Definition */

/* Module Definition */
template<int p0>
void L_drain_IO_L1_out_intra_trans(float local_L[1][11], hls::stream<float> &fifo_L_drain_local_in) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */


  // array
  // io_L2
  // io_L1
  // pe
  for (ap_uint<5> c2 = p0 + 1; c2 <= 11; c2 += 1) {
  #pragma HLS PIPELINE II=1
    // hls_pipeline
    {
      L_t1 in_data;
      L_t1 out_data;
      in_data = fifo_L_drain_local_in.read();
      out_data = in_data;
      local_L[0][c2 - p0 - 1] = out_data;
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0>
void L_drain_IO_L1_out_inter_trans(float local_L[1][11], hls::stream<float> &fifo_L_drain_in, hls::stream<float> &fifo_L_drain_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */

  for (ap_uint<5> c0 = p0; c0 <= 10; c0 += 1) {
    // io_L1
    if (c0 == p0) {
      // access_coalesce
      for (int c2 = 0; c2 <= -p0 + 10; c2 += 1) {
      #pragma HLS PIPELINE II=1
        {
          L_t1 in_data;
          L_t1 out_data;
          in_data = local_L[0][c2];
          out_data = in_data;
          fifo_L_drain_out.write(out_data);
        }
      }
    } else {
      // access_coalesce
      for (int c2 = 0; c2 <= -c0 + 10; c2 += 1) {
      #pragma HLS PIPELINE II=1
        {
          L_t1 in_data;
          L_t1 out_data;
          in_data = fifo_L_drain_in.read();
          out_data = in_data;
          fifo_L_drain_out.write(out_data);
        }
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0>
void L_drain_IO_L1_out_inter_trans_boundary(float local_L[1][11], hls::stream<float> &fifo_L_drain_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */

  for (ap_uint<5> c0 = p0; c0 <= 10; c0 += 1)
    if (c0 == p0) {
      // io_L1
      // access_coalesce
      for (int c2 = 0; c2 <= -p0 + 10; c2 += 1) {
      #pragma HLS PIPELINE II=1
        {
          L_t1 in_data;
          L_t1 out_data;
          in_data = local_L[0][c2];
          out_data = in_data;
          fifo_L_drain_out.write(out_data);
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
template<int p0>
void L_drain_IO_L1_out(hls::stream<float> &fifo_L_drain_in, hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_L_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  L_t1 local_L[1][11];
  #pragma HLS RESOURCE variable=local_L core=RAM_2P_BRAM
  /* Variable Declaration */

  // array
  // io_L2
  L_drain_IO_L1_out_intra_trans<p0>(
    /* array */ local_L, 
    /* fifo */ fifo_L_drain_local_in
  );
  L_drain_IO_L1_out_inter_trans<p0>(
    /* array */ local_L, 
    /* fifo */ fifo_L_drain_in, 
    /* fifo */ fifo_L_drain_out
  );
}
/* Module Definition */

/* Module Definition */
template<int p0>
void L_drain_IO_L1_out_wrapper(hls::stream<float> &fifo_L_drain_in, hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_L_drain_local_in)
 {
  L_drain_IO_L1_out<p0>(
    /* fifo */ fifo_L_drain_in, 
    /* fifo */ fifo_L_drain_out, 
    /* fifo */ fifo_L_drain_local_in);
}
/* Module Definition */

/* Module Definition */
template<int p0>
void L_drain_IO_L1_out_boundary(hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_L_drain_local_in) {
#pragma HLS INLINE
  /* Variable Declaration */
  L_t1 local_L[1][11];
  #pragma HLS RESOURCE variable=local_L core=RAM_2P_BRAM
  /* Variable Declaration */

  // array
  // io_L2
  L_drain_IO_L1_out_intra_trans<p0>(
    /* array */ local_L, 
    /* fifo */ fifo_L_drain_local_in
  );
  L_drain_IO_L1_out_inter_trans_boundary<p0>(
    /* array */ local_L, 
    /* fifo */ fifo_L_drain_out
  );
}
/* Module Definition */

/* Module Definition */
template<int p0>
void L_drain_IO_L1_out_boundary_wrapper(hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_L_drain_local_in)
 {
  L_drain_IO_L1_out_boundary<p0>(
    /* fifo */ fifo_L_drain_out, 
    /* fifo */ fifo_L_drain_local_in);
}
/* Module Definition */

/* Module Definition */
void L_drain_IO_L2_out(L_t1 *L, hls::stream<float> &fifo_L_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  /* Variable Declaration */

  // array
  // io_L2
  for (ap_uint<5> c0 = 0; c0 <= 10; c0 += 1) {
    // io_L1
    // access_coalesce
    for (int c2 = 0; c2 <= -c0 + 10; c2 += 1) {
    #pragma HLS PIPELINE II=1
      {
        L_t1 in_data;
        L_t1 out_data;
        in_data = fifo_L_drain_local_in.read();
        out_data = in_data;
        L[13*c0 + c2 + 1] = out_data;
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0>
void U_drain_IO_L1_out_intra_trans(float local_U[12][1], hls::stream<float> &fifo_U_drain_local_in) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */


  // array
  // io_L2
  // io_L1
  // pe
  for (int c1 = 0; c1 <= p0; c1 += 1) {
  #pragma HLS PIPELINE II=1
    // hls_pipeline
    {
      U_t1 in_data;
      U_t1 out_data;
      in_data = fifo_U_drain_local_in.read();
      out_data = in_data;
      local_U[c1][0] = out_data;
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0>
void U_drain_IO_L1_out_inter_trans(float local_U[12][1], hls::stream<float> &fifo_U_drain_in, hls::stream<float> &fifo_U_drain_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */

  for (ap_uint<5> c0 = p0; c0 <= 11; c0 += 1) {
    // io_L1
    if (c0 == p0) {
      for (ap_uint<5> c1 = 0; c1 <= 11; c1 += 1) {
      #pragma HLS PIPELINE II=1
        // access_coalesce
        {
          U_t1 in_data;
          U_t1 out_data;
          in_data = local_U[c1][0];
          out_data = in_data;
          fifo_U_drain_out.write(out_data);
        }
      }
    } else {
      for (ap_uint<5> c1 = 0; c1 <= 11; c1 += 1) {
      #pragma HLS PIPELINE II=1
        // access_coalesce
        {
          U_t1 in_data;
          U_t1 out_data;
          in_data = fifo_U_drain_in.read();
          out_data = in_data;
          fifo_U_drain_out.write(out_data);
        }
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0>
void U_drain_IO_L1_out_inter_trans_boundary(float local_U[12][1], hls::stream<float> &fifo_U_drain_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */

  for (ap_uint<5> c0 = p0; c0 <= 11; c0 += 1)
    if (c0 == p0) {
      // io_L1
      for (ap_uint<5> c1 = 0; c1 <= 11; c1 += 1) {
      #pragma HLS PIPELINE II=1
        // access_coalesce
        {
          U_t1 in_data;
          U_t1 out_data;
          in_data = local_U[c1][0];
          out_data = in_data;
          fifo_U_drain_out.write(out_data);
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
template<int p0>
void U_drain_IO_L1_out(hls::stream<float> &fifo_U_drain_in, hls::stream<float> &fifo_U_drain_out, hls::stream<float> &fifo_U_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  U_t1 local_U[12][1];
  #pragma HLS RESOURCE variable=local_U core=RAM_2P_BRAM
  /* Variable Declaration */

  // array
  // io_L2
  U_drain_IO_L1_out_intra_trans<p0>(
    /* array */ local_U, 
    /* fifo */ fifo_U_drain_local_in
  );
  U_drain_IO_L1_out_inter_trans<p0>(
    /* array */ local_U, 
    /* fifo */ fifo_U_drain_in, 
    /* fifo */ fifo_U_drain_out
  );
}
/* Module Definition */

/* Module Definition */
template<int p0>
void U_drain_IO_L1_out_wrapper(hls::stream<float> &fifo_U_drain_in, hls::stream<float> &fifo_U_drain_out, hls::stream<float> &fifo_U_drain_local_in)
 {
  U_drain_IO_L1_out<p0>(
    /* fifo */ fifo_U_drain_in, 
    /* fifo */ fifo_U_drain_out, 
    /* fifo */ fifo_U_drain_local_in);
}
/* Module Definition */

/* Module Definition */
template<int p0>
void U_drain_IO_L1_out_boundary(hls::stream<float> &fifo_U_drain_out, hls::stream<float> &fifo_U_drain_local_in) {
#pragma HLS INLINE
  /* Variable Declaration */
  U_t1 local_U[12][1];
  #pragma HLS RESOURCE variable=local_U core=RAM_2P_BRAM
  /* Variable Declaration */

  // array
  // io_L2
  U_drain_IO_L1_out_intra_trans<p0>(
    /* array */ local_U, 
    /* fifo */ fifo_U_drain_local_in
  );
  U_drain_IO_L1_out_inter_trans_boundary<p0>(
    /* array */ local_U, 
    /* fifo */ fifo_U_drain_out
  );
}
/* Module Definition */

/* Module Definition */
template<int p0>
void U_drain_IO_L1_out_boundary_wrapper(hls::stream<float> &fifo_U_drain_out, hls::stream<float> &fifo_U_drain_local_in)
 {
  U_drain_IO_L1_out_boundary<p0>(
    /* fifo */ fifo_U_drain_out, 
    /* fifo */ fifo_U_drain_local_in);
}
/* Module Definition */

/* Module Definition */
void U_drain_IO_L2_out(U_t1 *U, hls::stream<float> &fifo_U_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  /* Variable Declaration */

  // array
  // io_L2
  for (ap_uint<5> c0 = 0; c0 <= 11; c0 += 1) {
    // io_L1
    for (ap_uint<5> c1 = 0; c1 <= 11; c1 += 1) {
    #pragma HLS PIPELINE II=1
      // access_coalesce
      {
        U_t1 in_data;
        U_t1 out_data;
        in_data = fifo_U_drain_local_in.read();
        out_data = in_data;
        U[c0 + 12*c1] = out_data;
      }
    }
  }
}
/* Module Definition */

/* AutoSA dangling-stream fix: fifo_L_tmp_0_PE_12, fifo_L_drain_PE_11 stay local to this call. */
template<int p0>
void PE_wrapper_dfix0(hls::stream<float> &fifo_L_tmp_0_in, hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_U_drain_out) {
  hls::stream<float> fifo_L_tmp_0_out;
  hls::stream<float> fifo_L_drain_out;
  PE_wrapper<p0>(fifo_L_tmp_0_in, fifo_L_tmp_0_out, fifo_A_in, fifo_L_drain_out, fifo_U_drain_out);
#ifndef __SYNTHESIS__
  if (!fifo_L_tmp_0_out.empty()) { printf("AUTOSA_DFIX: dangling stream fifo_L_tmp_0_PE_12 received data\n"); exit(1); }
  if (!fifo_L_drain_out.empty()) { printf("AUTOSA_DFIX: dangling stream fifo_L_drain_PE_11 received data\n"); exit(1); }
#endif
}

void kernel0(A_t4 *A, L_t1 *L, U_t1 *U)
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem_A
#pragma HLS INTERFACE m_axi port=L offset=slave bundle=gmem_L
#pragma HLS INTERFACE m_axi port=U offset=slave bundle=gmem_U
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=L bundle=control
#pragma HLS INTERFACE s_axilite port=U bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW

  /* FIFO Declaration */
  /* A_IO_L1_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L1_in_0;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_0 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L1_in_1;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_1 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L1_in_2;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_2 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L1_in_3;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_3 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L1_in_4;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_4 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L1_in_5;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_5 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L1_in_6;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_6 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L1_in_7;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_7 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L1_in_8;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_8 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L1_in_9;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_9 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L1_in_10;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_10 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L1_in_11;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_11 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<A_t4> fifo_A_A_IO_L1_in_12;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_12 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_12 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_tmp_0_PE_0;
  #pragma HLS STREAM variable=fifo_L_tmp_0_PE_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_tmp_0_PE_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_tmp_0_PE_1;
  #pragma HLS STREAM variable=fifo_L_tmp_0_PE_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_tmp_0_PE_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_tmp_0_PE_2;
  #pragma HLS STREAM variable=fifo_L_tmp_0_PE_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_tmp_0_PE_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_tmp_0_PE_3;
  #pragma HLS STREAM variable=fifo_L_tmp_0_PE_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_tmp_0_PE_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_tmp_0_PE_4;
  #pragma HLS STREAM variable=fifo_L_tmp_0_PE_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_tmp_0_PE_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_tmp_0_PE_5;
  #pragma HLS STREAM variable=fifo_L_tmp_0_PE_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_tmp_0_PE_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_tmp_0_PE_6;
  #pragma HLS STREAM variable=fifo_L_tmp_0_PE_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_tmp_0_PE_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_tmp_0_PE_7;
  #pragma HLS STREAM variable=fifo_L_tmp_0_PE_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_tmp_0_PE_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_tmp_0_PE_8;
  #pragma HLS STREAM variable=fifo_L_tmp_0_PE_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_tmp_0_PE_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_tmp_0_PE_9;
  #pragma HLS STREAM variable=fifo_L_tmp_0_PE_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_tmp_0_PE_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_tmp_0_PE_10;
  #pragma HLS STREAM variable=fifo_L_tmp_0_PE_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_tmp_0_PE_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_tmp_0_PE_11;
  #pragma HLS STREAM variable=fifo_L_tmp_0_PE_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_tmp_0_PE_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_0;
  #pragma HLS STREAM variable=fifo_A_PE_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_1;
  #pragma HLS STREAM variable=fifo_A_PE_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_2;
  #pragma HLS STREAM variable=fifo_A_PE_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_3;
  #pragma HLS STREAM variable=fifo_A_PE_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_4;
  #pragma HLS STREAM variable=fifo_A_PE_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_5;
  #pragma HLS STREAM variable=fifo_A_PE_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_6;
  #pragma HLS STREAM variable=fifo_A_PE_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_7;
  #pragma HLS STREAM variable=fifo_A_PE_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_8;
  #pragma HLS STREAM variable=fifo_A_PE_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_9;
  #pragma HLS STREAM variable=fifo_A_PE_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_10;
  #pragma HLS STREAM variable=fifo_A_PE_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_11;
  #pragma HLS STREAM variable=fifo_A_PE_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_0;
  #pragma HLS STREAM variable=fifo_L_drain_PE_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_1;
  #pragma HLS STREAM variable=fifo_L_drain_PE_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_2;
  #pragma HLS STREAM variable=fifo_L_drain_PE_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_3;
  #pragma HLS STREAM variable=fifo_L_drain_PE_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_4;
  #pragma HLS STREAM variable=fifo_L_drain_PE_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_5;
  #pragma HLS STREAM variable=fifo_L_drain_PE_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_6;
  #pragma HLS STREAM variable=fifo_L_drain_PE_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_7;
  #pragma HLS STREAM variable=fifo_L_drain_PE_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_8;
  #pragma HLS STREAM variable=fifo_L_drain_PE_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_9;
  #pragma HLS STREAM variable=fifo_L_drain_PE_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_10;
  #pragma HLS STREAM variable=fifo_L_drain_PE_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_0;
  #pragma HLS STREAM variable=fifo_U_drain_PE_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_1;
  #pragma HLS STREAM variable=fifo_U_drain_PE_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_2;
  #pragma HLS STREAM variable=fifo_U_drain_PE_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_3;
  #pragma HLS STREAM variable=fifo_U_drain_PE_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_4;
  #pragma HLS STREAM variable=fifo_U_drain_PE_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_5;
  #pragma HLS STREAM variable=fifo_U_drain_PE_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_6;
  #pragma HLS STREAM variable=fifo_U_drain_PE_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_7;
  #pragma HLS STREAM variable=fifo_U_drain_PE_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_8;
  #pragma HLS STREAM variable=fifo_U_drain_PE_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_9;
  #pragma HLS STREAM variable=fifo_U_drain_PE_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_10;
  #pragma HLS STREAM variable=fifo_U_drain_PE_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_11;
  #pragma HLS STREAM variable=fifo_U_drain_PE_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_11 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_0;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_0 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_1;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_1 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_2;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_2 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_3;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_3 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_4;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_4 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_5;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_5 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_6;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_6 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_7;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_7 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_8;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_8 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_9;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_9 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_10;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_10 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_11;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_11 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_12;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_12 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_12 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<float> fifo_U_drain_U_drain_IO_L1_out_0;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_0 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<float> fifo_U_drain_U_drain_IO_L1_out_1;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_1 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<float> fifo_U_drain_U_drain_IO_L1_out_2;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_2 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<float> fifo_U_drain_U_drain_IO_L1_out_3;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_3 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<float> fifo_U_drain_U_drain_IO_L1_out_4;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_4 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<float> fifo_U_drain_U_drain_IO_L1_out_5;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_5 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<float> fifo_U_drain_U_drain_IO_L1_out_6;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_6 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<float> fifo_U_drain_U_drain_IO_L1_out_7;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_7 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<float> fifo_U_drain_U_drain_IO_L1_out_8;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_8 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<float> fifo_U_drain_U_drain_IO_L1_out_9;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_9 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<float> fifo_U_drain_U_drain_IO_L1_out_10;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_10 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<float> fifo_U_drain_U_drain_IO_L1_out_11;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_11 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<float> fifo_U_drain_U_drain_IO_L1_out_12;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_12 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_12 core=FIFO_SRL
  /* FIFO Declaration */

  /* Module Call */
  A_IO_L2_in(
    /* array */ A,
    /* fifo */ fifo_A_A_IO_L1_in_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_wrapper<0>(
    /* fifo */ fifo_A_A_IO_L1_in_0,
    /* fifo */ fifo_A_A_IO_L1_in_1,
    /* fifo */ fifo_A_PE_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_wrapper<1>(
    /* fifo */ fifo_A_A_IO_L1_in_1,
    /* fifo */ fifo_A_A_IO_L1_in_2,
    /* fifo */ fifo_A_PE_1
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_wrapper<2>(
    /* fifo */ fifo_A_A_IO_L1_in_2,
    /* fifo */ fifo_A_A_IO_L1_in_3,
    /* fifo */ fifo_A_PE_2
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_wrapper<3>(
    /* fifo */ fifo_A_A_IO_L1_in_3,
    /* fifo */ fifo_A_A_IO_L1_in_4,
    /* fifo */ fifo_A_PE_3
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_wrapper<4>(
    /* fifo */ fifo_A_A_IO_L1_in_4,
    /* fifo */ fifo_A_A_IO_L1_in_5,
    /* fifo */ fifo_A_PE_4
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_wrapper<5>(
    /* fifo */ fifo_A_A_IO_L1_in_5,
    /* fifo */ fifo_A_A_IO_L1_in_6,
    /* fifo */ fifo_A_PE_5
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_wrapper<6>(
    /* fifo */ fifo_A_A_IO_L1_in_6,
    /* fifo */ fifo_A_A_IO_L1_in_7,
    /* fifo */ fifo_A_PE_6
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_wrapper<7>(
    /* fifo */ fifo_A_A_IO_L1_in_7,
    /* fifo */ fifo_A_A_IO_L1_in_8,
    /* fifo */ fifo_A_PE_7
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_wrapper<8>(
    /* fifo */ fifo_A_A_IO_L1_in_8,
    /* fifo */ fifo_A_A_IO_L1_in_9,
    /* fifo */ fifo_A_PE_8
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_wrapper<9>(
    /* fifo */ fifo_A_A_IO_L1_in_9,
    /* fifo */ fifo_A_A_IO_L1_in_10,
    /* fifo */ fifo_A_PE_9
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_wrapper<10>(
    /* fifo */ fifo_A_A_IO_L1_in_10,
    /* fifo */ fifo_A_A_IO_L1_in_11,
    /* fifo */ fifo_A_PE_10
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_boundary_wrapper<11>(
    /* fifo */ fifo_A_A_IO_L1_in_11,
    /* fifo */ fifo_A_PE_11
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<0>(
    /* fifo */ fifo_L_tmp_0_PE_0,
    /* fifo */ fifo_L_tmp_0_PE_1,
    /* fifo */ fifo_A_PE_0,
    /* fifo */ fifo_L_drain_PE_0,
    /* fifo */ fifo_U_drain_PE_0
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<1>(
    /* fifo */ fifo_L_tmp_0_PE_1,
    /* fifo */ fifo_L_tmp_0_PE_2,
    /* fifo */ fifo_A_PE_1,
    /* fifo */ fifo_L_drain_PE_1,
    /* fifo */ fifo_U_drain_PE_1
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<2>(
    /* fifo */ fifo_L_tmp_0_PE_2,
    /* fifo */ fifo_L_tmp_0_PE_3,
    /* fifo */ fifo_A_PE_2,
    /* fifo */ fifo_L_drain_PE_2,
    /* fifo */ fifo_U_drain_PE_2
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<3>(
    /* fifo */ fifo_L_tmp_0_PE_3,
    /* fifo */ fifo_L_tmp_0_PE_4,
    /* fifo */ fifo_A_PE_3,
    /* fifo */ fifo_L_drain_PE_3,
    /* fifo */ fifo_U_drain_PE_3
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<4>(
    /* fifo */ fifo_L_tmp_0_PE_4,
    /* fifo */ fifo_L_tmp_0_PE_5,
    /* fifo */ fifo_A_PE_4,
    /* fifo */ fifo_L_drain_PE_4,
    /* fifo */ fifo_U_drain_PE_4
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<5>(
    /* fifo */ fifo_L_tmp_0_PE_5,
    /* fifo */ fifo_L_tmp_0_PE_6,
    /* fifo */ fifo_A_PE_5,
    /* fifo */ fifo_L_drain_PE_5,
    /* fifo */ fifo_U_drain_PE_5
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<6>(
    /* fifo */ fifo_L_tmp_0_PE_6,
    /* fifo */ fifo_L_tmp_0_PE_7,
    /* fifo */ fifo_A_PE_6,
    /* fifo */ fifo_L_drain_PE_6,
    /* fifo */ fifo_U_drain_PE_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<7>(
    /* fifo */ fifo_L_tmp_0_PE_7,
    /* fifo */ fifo_L_tmp_0_PE_8,
    /* fifo */ fifo_A_PE_7,
    /* fifo */ fifo_L_drain_PE_7,
    /* fifo */ fifo_U_drain_PE_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<8>(
    /* fifo */ fifo_L_tmp_0_PE_8,
    /* fifo */ fifo_L_tmp_0_PE_9,
    /* fifo */ fifo_A_PE_8,
    /* fifo */ fifo_L_drain_PE_8,
    /* fifo */ fifo_U_drain_PE_8
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<9>(
    /* fifo */ fifo_L_tmp_0_PE_9,
    /* fifo */ fifo_L_tmp_0_PE_10,
    /* fifo */ fifo_A_PE_9,
    /* fifo */ fifo_L_drain_PE_9,
    /* fifo */ fifo_U_drain_PE_9
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<10>(
    /* fifo */ fifo_L_tmp_0_PE_10,
    /* fifo */ fifo_L_tmp_0_PE_11,
    /* fifo */ fifo_A_PE_10,
    /* fifo */ fifo_L_drain_PE_10,
    /* fifo */ fifo_U_drain_PE_10
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper_dfix0<11>(
    /* fifo */ fifo_L_tmp_0_PE_11,
    /* fifo */ fifo_A_PE_11,
    /* fifo */ fifo_U_drain_PE_11
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_boundary_wrapper<10>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10,
    /* fifo */ fifo_L_drain_PE_10
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<9>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9,
    /* fifo */ fifo_L_drain_PE_9
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<8>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8,
    /* fifo */ fifo_L_drain_PE_8
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<7>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7,
    /* fifo */ fifo_L_drain_PE_7
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<6>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_6,
    /* fifo */ fifo_L_drain_PE_6
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<5>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_6,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_5,
    /* fifo */ fifo_L_drain_PE_5
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<4>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_5,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_4,
    /* fifo */ fifo_L_drain_PE_4
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<3>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_4,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_3,
    /* fifo */ fifo_L_drain_PE_3
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<2>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_3,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_2,
    /* fifo */ fifo_L_drain_PE_2
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<1>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_2,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_1,
    /* fifo */ fifo_L_drain_PE_1
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<0>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_1,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_0,
    /* fifo */ fifo_L_drain_PE_0
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L2_out(
    /* array */ L,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_boundary_wrapper<11>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11,
    /* fifo */ fifo_U_drain_PE_11
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<10>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10,
    /* fifo */ fifo_U_drain_PE_10
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<9>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9,
    /* fifo */ fifo_U_drain_PE_9
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<8>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8,
    /* fifo */ fifo_U_drain_PE_8
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<7>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7,
    /* fifo */ fifo_U_drain_PE_7
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<6>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6,
    /* fifo */ fifo_U_drain_PE_6
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<5>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_5,
    /* fifo */ fifo_U_drain_PE_5
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<4>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_5,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_4,
    /* fifo */ fifo_U_drain_PE_4
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<3>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_4,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_3,
    /* fifo */ fifo_U_drain_PE_3
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<2>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_3,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_2,
    /* fifo */ fifo_U_drain_PE_2
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<1>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_2,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_1,
    /* fifo */ fifo_U_drain_PE_1
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<0>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_1,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_0,
    /* fifo */ fifo_U_drain_PE_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L2_out(
    /* array */ U,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_0
  );
  /* Module Call */

}
