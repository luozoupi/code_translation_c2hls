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
typedef ap_uint<128> U_tmp_t4;
typedef ap_uint<128> U_t4;
/* Data Type */

extern "C" {
void kernel0(A_t1 *A, L_t1 *L, U_t4 *U);
}
template<int p0, int p1>
void A_IO_L1_in_intra_trans(float local_A[12][1], hls::stream<float> &fifo_A_local_out);
template<int p0, int p1>
void A_IO_L1_in_inter_trans(float local_A[12][1], hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_A_out);
template<int p0, int p1>
void A_IO_L1_in_inter_trans_boundary(float local_A[12][1], hls::stream<float> &fifo_A_in);
template<int p0, int p1>
void A_IO_L1_in_wrapper(hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_A_out, hls::stream<float> &fifo_A_local_out);
template<int p0, int p1>
void A_IO_L1_in_boundary_wrapper(hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_A_local_out);
template<int p0, int p1>
void PE_wrapper(hls::stream<float> &fifo_V_in, hls::stream<float> &fifo_V_out, hls::stream<float> &fifo_U_tmp_1_in, hls::stream<float> &fifo_U_tmp_1_out, hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_U_drain_out);
template<int p0, int p1>
void L_drain_IO_L1_out_intra_trans(float local_L[1][1], hls::stream<float> &fifo_L_drain_local_in);
template<int p0, int p1>
void L_drain_IO_L1_out_inter_trans(float local_L[1][1], hls::stream<float> &fifo_L_drain_in, hls::stream<float> &fifo_L_drain_out);
template<int p0, int p1>
void L_drain_IO_L1_out_inter_trans_boundary(float local_L[1][1], hls::stream<float> &fifo_L_drain_out);
template<int p0, int p1>
void L_drain_IO_L1_out_wrapper(hls::stream<float> &fifo_L_drain_in, hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_L_drain_local_in);
template<int p0, int p1>
void L_drain_IO_L1_out_boundary_wrapper(hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_L_drain_local_in);
template<int p0, int p1>
void U_drain_IO_L1_out_intra_trans(U_t4 local_U[1][3], hls::stream<float> &fifo_U_drain_local_in);
template<int p0, int p1>
void U_drain_IO_L1_out_inter_trans(U_t4 local_U[1][3], hls::stream<U_t4> &fifo_U_drain_in, hls::stream<U_t4> &fifo_U_drain_out);
template<int p0, int p1>
void U_drain_IO_L1_out_inter_trans_boundary(U_t4 local_U[1][3], hls::stream<U_t4> &fifo_U_drain_out);
template<int p0, int p1>
void U_drain_IO_L1_out_wrapper(hls::stream<U_t4> &fifo_U_drain_in, hls::stream<U_t4> &fifo_U_drain_out, hls::stream<float> &fifo_U_drain_local_in);
template<int p0, int p1>
void U_drain_IO_L1_out_boundary_wrapper(hls::stream<U_t4> &fifo_U_drain_out, hls::stream<float> &fifo_U_drain_local_in);

/* Module Definition */
void A_IO_L3_in(A_t1 *A, hls::stream<float> &fifo_A_local_out) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  /* Variable Declaration */

  // array
  // io_L3
  for (ap_uint<5> c0 = 0; c0 <= 11; c0 += 1) {
    // io_L2
    // io_L1
    for (ap_uint<5> c2 = 0; c2 <= 11; c2 += 1) {
    #pragma HLS PIPELINE II=1
      // access_coalesce
      {
        A_t1 in_data;
        A_t1 out_data;
        in_data = A[c0 + 12*c2];
        out_data = in_data;
        fifo_A_local_out.write(out_data);
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0>
void A_IO_L2_in(hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_A_out, hls::stream<float> &fifo_A_local_out) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  /* Variable Declaration */

  // array
  // io_L3
  for (ap_uint<5> c0 = p0; c0 <= 11; c0 += 1) {
    // io_L2
    if (c0 == p0) {
      // io_L1
      for (ap_uint<5> c2 = 0; c2 <= 11; c2 += 1) {
      #pragma HLS PIPELINE II=1
        // access_coalesce
        {
          A_t1 in_data;
          A_t1 out_data;
          in_data = fifo_A_in.read();
          out_data = in_data;
          fifo_A_local_out.write(out_data);
        }
      }
    } else {
      // io_L1
      for (ap_uint<5> c2 = 0; c2 <= 11; c2 += 1) {
      #pragma HLS PIPELINE II=1
        // access_coalesce
        {
          A_t1 in_data;
          A_t1 out_data;
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
void A_IO_L2_in_boundary(hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_A_local_out) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  /* Variable Declaration */

  // array
  // io_L3
  for (ap_uint<5> c0 = p0; c0 <= 11; c0 += 1)
    if (c0 == p0) {
      // io_L2
      // io_L1
      for (ap_uint<5> c2 = 0; c2 <= 11; c2 += 1) {
      #pragma HLS PIPELINE II=1
        // access_coalesce
        {
          A_t1 in_data;
          A_t1 out_data;
          in_data = fifo_A_in.read();
          out_data = in_data;
          fifo_A_local_out.write(out_data);
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void A_IO_L1_in_intra_trans(float local_A[12][1], hls::stream<float> &fifo_A_local_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */


  if (p1 == 0) {
    // array
    // io_L3
    // io_L2
    // io_L1
    // pe
    for (ap_uint<5> c2 = 0; c2 <= 11; c2 += 1) {
    #pragma HLS PIPELINE II=1
      // hls_pipeline
      {
        A_t1 in_data;
        A_t1 out_data;
        in_data = local_A[c2][0];
        out_data = in_data;
        fifo_A_local_out.write(out_data);
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void A_IO_L1_in_inter_trans(float local_A[12][1], hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_A_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */

  // io_L1
  for (ap_uint<5> c2 = 0; c2 <= 11; c2 += 1) {
  #pragma HLS PIPELINE II=1
    // access_coalesce
    {
      A_t1 in_data;
      A_t1 out_data;
      in_data = fifo_A_in.read();
      out_data = in_data;
      local_A[c2][0] = out_data;
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void A_IO_L1_in_inter_trans_boundary(float local_A[12][1], hls::stream<float> &fifo_A_in) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */

  // io_L1
  for (ap_uint<5> c2 = 0; c2 <= 11; c2 += 1) {
  #pragma HLS PIPELINE II=1
    // access_coalesce
    {
      A_t1 in_data;
      A_t1 out_data;
      in_data = fifo_A_in.read();
      out_data = in_data;
      local_A[c2][0] = out_data;
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void A_IO_L1_in(hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_A_out, hls::stream<float> &fifo_A_local_out) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  A_t1 local_A[12][1];
  #pragma HLS RESOURCE variable=local_A core=RAM_1P_BRAM
  /* Variable Declaration */

  // array
  // io_L3
  // io_L2
  A_IO_L1_in_inter_trans<p0, p1>(
    /* array */ local_A, 
    /* fifo */ fifo_A_in, 
    /* fifo */ fifo_A_out
  );
  A_IO_L1_in_intra_trans<p0, p1>(
    /* array */ local_A, 
    /* fifo */ fifo_A_local_out
  );
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void A_IO_L1_in_wrapper(hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_A_out, hls::stream<float> &fifo_A_local_out)
 {
  A_IO_L1_in<p0, p1>(
    /* fifo */ fifo_A_in, 
    /* fifo */ fifo_A_out, 
    /* fifo */ fifo_A_local_out);
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void A_IO_L1_in_boundary(hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_A_local_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  A_t1 local_A[12][1];
  #pragma HLS RESOURCE variable=local_A core=RAM_1P_BRAM
  /* Variable Declaration */

  // array
  // io_L3
  // io_L2
  A_IO_L1_in_inter_trans_boundary<p0, p1>(
    /* array */ local_A, 
    /* fifo */ fifo_A_in
  );
  A_IO_L1_in_intra_trans<p0, p1>(
    /* array */ local_A, 
    /* fifo */ fifo_A_local_out
  );
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void A_IO_L1_in_boundary_wrapper(hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_A_local_out)
 {
  A_IO_L1_in_boundary<p0, p1>(
    /* fifo */ fifo_A_in, 
    /* fifo */ fifo_A_local_out);
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void PE(hls::stream<float> &fifo_V_in, hls::stream<float> &fifo_V_out, hls::stream<float> &fifo_U_tmp_1_in, hls::stream<float> &fifo_U_tmp_1_out, hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_U_drain_out) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  prev_V_t1 local_prev_V[12][1];
  #pragma HLS RESOURCE variable=local_prev_V core=RAM_2P_BRAM
  V_t1 local_V[12][1];
  #pragma HLS RESOURCE variable=local_V core=RAM_2P_BRAM
  U_tmp_t1 local_U_tmp[12][2];
  #pragma HLS RESOURCE variable=local_U_tmp core=RAM_2P_BRAM
  L_tmp_t1 local_L_tmp[12][1];
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
  for (ap_uint<5> c2 = p0; c2 <= 11; c2 += 1) {
  #pragma HLS PIPELINE II=1
    {
      if (p0 == 0) {
        local_A[0][0] = fifo_A_in.read();
        local_prev_V[c2][0] = local_A[0][0];
      } else {
        local_V[c2 - p0][0] = fifo_V_in.read();
        local_prev_V[c2 - p0][0] = local_V[c2 - p0][0];
      }
      if (p1 == p0) {
        local_U_tmp[c2 - p0][1] = local_prev_V[c2 - p0][0];
        if (p0 <= 10)
          fifo_U_tmp_1_out.write(local_U_tmp[c2 - p0][1]);
        local_U[0][0] = local_U_tmp[c2 - p0][1];
        fifo_U_drain_out.write(local_U[0][0]);
      } else {
        local_U_tmp[c2 - p0][0] = fifo_U_tmp_1_in.read();
        local_U_tmp[c2 - p0][1] = local_U_tmp[c2 - p0][0];
        if (p1 <= 10)
          fifo_U_tmp_1_out.write(local_U_tmp[c2 - p0][1]);
        if (c2 == p0) {
          local_L_tmp[0][0] = (local_prev_V[0][0] / local_U_tmp[0][1]);
          local_L[0][0] = local_L_tmp[0][0];
          fifo_L_drain_out.write(local_L[0][0]);
        } else {
          local_L_tmp[c2 - p0][0] = local_L_tmp[c2 - p0 - 1][0];
        }
        local_V[c2 - p0][0] = (local_prev_V[c2 - p0][0] - (local_L_tmp[c2 - p0][0] * local_U_tmp[c2 - p0][1]));
        if (c2 >= p0 + 1)
          fifo_V_out.write(local_V[c2 - p0][0]);
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void PE_wrapper(hls::stream<float> &fifo_V_in, hls::stream<float> &fifo_V_out, hls::stream<float> &fifo_U_tmp_1_in, hls::stream<float> &fifo_U_tmp_1_out, hls::stream<float> &fifo_A_in, hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_U_drain_out)
 {
  PE<p0, p1>(
    /* fifo */ fifo_V_in, 
    /* fifo */ fifo_V_out, 
    /* fifo */ fifo_U_tmp_1_in, 
    /* fifo */ fifo_U_tmp_1_out, 
    /* fifo */ fifo_A_in, 
    /* fifo */ fifo_L_drain_out, 
    /* fifo */ fifo_U_drain_out);
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void L_drain_IO_L1_out_intra_trans(float local_L[1][1], hls::stream<float> &fifo_L_drain_local_in) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */


  if (p0 >= p1 + 1) {
    // array
    // io_L3
    // io_L2
    // io_L1
    // pe
    // hls_pipeline
    // hls_pipeline
    {
      L_t1 in_data;
      L_t1 out_data;
      in_data = fifo_L_drain_local_in.read();
      out_data = in_data;
      local_L[0][0] = out_data;
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void L_drain_IO_L1_out_inter_trans(float local_L[1][1], hls::stream<float> &fifo_L_drain_in, hls::stream<float> &fifo_L_drain_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */

  for (int c1 = p1; c1 < p0; c1 += 1) {
  #pragma HLS PIPELINE II=1
    // io_L1
    if (c1 == p1) {
      // access_coalesce
      {
        L_t1 in_data;
        L_t1 out_data;
        in_data = local_L[0][0];
        out_data = in_data;
        fifo_L_drain_out.write(out_data);
      }
    } else {
      // access_coalesce
      // hls_pipeline
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
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void L_drain_IO_L1_out_inter_trans_boundary(float local_L[1][1], hls::stream<float> &fifo_L_drain_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */

  for (int c1 = p1; c1 < p0; c1 += 1)
  #pragma HLS PIPELINE II=1
    if (c1 == p1) {
      // io_L1
      // access_coalesce
      {
        L_t1 in_data;
        L_t1 out_data;
        in_data = local_L[0][0];
        out_data = in_data;
        fifo_L_drain_out.write(out_data);
      }
    }
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void L_drain_IO_L1_out(hls::stream<float> &fifo_L_drain_in, hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_L_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  L_t1 local_L[1][1];
  #pragma HLS ARRAY_PARTITION variable=local_L dim=0 complete
  /* Variable Declaration */

  if (p0 >= 1) {
    // array
    // io_L3
    // io_L2
    L_drain_IO_L1_out_intra_trans<p0, p1>(
      /* array */ local_L, 
      /* fifo */ fifo_L_drain_local_in
    );
    L_drain_IO_L1_out_inter_trans<p0, p1>(
      /* array */ local_L, 
      /* fifo */ fifo_L_drain_in, 
      /* fifo */ fifo_L_drain_out
    );
  }
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void L_drain_IO_L1_out_wrapper(hls::stream<float> &fifo_L_drain_in, hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_L_drain_local_in)
 {
  L_drain_IO_L1_out<p0, p1>(
    /* fifo */ fifo_L_drain_in, 
    /* fifo */ fifo_L_drain_out, 
    /* fifo */ fifo_L_drain_local_in);
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void L_drain_IO_L1_out_boundary(hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_L_drain_local_in) {
#pragma HLS INLINE
  /* Variable Declaration */
  L_t1 local_L[1][1];
  #pragma HLS ARRAY_PARTITION variable=local_L dim=0 complete
  /* Variable Declaration */

  if (p0 >= 1) {
    // array
    // io_L3
    // io_L2
    L_drain_IO_L1_out_intra_trans<p0, p1>(
      /* array */ local_L, 
      /* fifo */ fifo_L_drain_local_in
    );
    L_drain_IO_L1_out_inter_trans_boundary<p0, p1>(
      /* array */ local_L, 
      /* fifo */ fifo_L_drain_out
    );
  }
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void L_drain_IO_L1_out_boundary_wrapper(hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_L_drain_local_in)
 {
  L_drain_IO_L1_out_boundary<p0, p1>(
    /* fifo */ fifo_L_drain_out, 
    /* fifo */ fifo_L_drain_local_in);
}
/* Module Definition */

/* Module Definition */
template<int p0>
void L_drain_IO_L2_out(hls::stream<float> &fifo_L_drain_in, hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_L_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  /* Variable Declaration */

  // array
  // io_L3
  for (ap_uint<5> c0 = max(1, p0); c0 <= 11; c0 += 1) {
    // io_L2
    if (c0 == p0) {
      for (int c1 = 0; c1 < p0; c1 += 1) {
      #pragma HLS PIPELINE II=1
        // io_L1
        // access_coalesce
        {
          L_t1 in_data;
          L_t1 out_data;
          in_data = fifo_L_drain_local_in.read();
          out_data = in_data;
          fifo_L_drain_out.write(out_data);
        }
      }
    } else {
      for (int c1 = 0; c1 < c0; c1 += 1) {
      #pragma HLS PIPELINE II=1
        // io_L1
        // access_coalesce
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
void L_drain_IO_L2_out_boundary(hls::stream<float> &fifo_L_drain_out, hls::stream<float> &fifo_L_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  /* Variable Declaration */

  // array
  // io_L3
  for (ap_uint<5> c0 = max(1, p0); c0 <= 11; c0 += 1)
    if (c0 == p0) {
      // io_L2
      for (int c1 = 0; c1 < p0; c1 += 1) {
      #pragma HLS PIPELINE II=1
        // io_L1
        // access_coalesce
        {
          L_t1 in_data;
          L_t1 out_data;
          in_data = fifo_L_drain_local_in.read();
          out_data = in_data;
          fifo_L_drain_out.write(out_data);
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void L_drain_IO_L3_out(L_t1 *L, hls::stream<float> &fifo_L_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  /* Variable Declaration */

  // array
  // io_L3
  for (ap_uint<5> c0 = 1; c0 <= 11; c0 += 1) {
    // io_L2
    for (int c1 = 0; c1 < c0; c1 += 1) {
    #pragma HLS PIPELINE II=1
      // io_L1
      // access_coalesce
      {
        L_t1 in_data;
        L_t1 out_data;
        in_data = fifo_L_drain_local_in.read();
        out_data = in_data;
        L[c0 + 12*c1] = out_data;
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void U_drain_IO_L1_out_intra_trans(U_t4 local_U[1][3], hls::stream<float> &fifo_U_drain_local_in) {
#pragma HLS INLINE
  /* Variable Declaration */
  ap_uint<32> data_split[4];
  #pragma HLS ARRAY_PARTITION variable=data_split complete
  /* Variable Declaration */


  if (p1 == p0) {
    // array
    // io_L3
    // io_L2
    // io_L1
    // pe
    for (ap_uint<5> c2 = p0; c2 <= 11; c2 += 1) {
    #pragma HLS PIPELINE II=1
      // hls_pipeline
      {
        U_t1 in_data;
        U_t4 out_data;
        in_data = fifo_U_drain_local_in.read();
        int split_idx = (c2 - p0) % 4;
        out_data = local_U[0][(-p0 + c2) / 4];
        for (ap_uint<3> n = 0; n < 4; n++) {
        #pragma HLS UNROLL
          data_split[n] = out_data(31, 0);
          out_data = out_data >> 32;
        }
        union {unsigned int ui; float ut;} u;
        u.ut = in_data;
        data_split[split_idx] = ap_uint<32>(u.ui);
        out_data = (data_split[3], data_split[2], data_split[1], data_split[0]);        local_U[0][(-p0 + c2) / 4] = out_data;
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void U_drain_IO_L1_out_inter_trans(U_t4 local_U[1][3], hls::stream<U_t4> &fifo_U_drain_in, hls::stream<U_t4> &fifo_U_drain_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */

  // io_L1
  if (p1 == p0) {
    // access_coalesce
    for (int c3 = 0; c3 <= -((p0 + 4) / 4) + 3; c3 += 1) {
    #pragma HLS PIPELINE II=1
      {
        U_t4 in_data;
        U_t4 out_data;
        in_data = local_U[0][c3];
        out_data = in_data;
        fifo_U_drain_out.write(out_data);
      }
    }
  } else {
    // access_coalesce
    for (int c3 = 0; c3 <= -((p0 + 4) / 4) + 3; c3 += 1) {
    #pragma HLS PIPELINE II=1
      {
        U_t4 in_data;
        U_t4 out_data;
        in_data = fifo_U_drain_in.read();
        out_data = in_data;
        fifo_U_drain_out.write(out_data);
      }
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void U_drain_IO_L1_out_inter_trans_boundary(U_t4 local_U[1][3], hls::stream<U_t4> &fifo_U_drain_out) {
#pragma HLS INLINE
  /* Variable Declaration */
  /* Variable Declaration */

  // io_L1
  // access_coalesce
  for (int c3 = 0; c3 <= -((p0 + 4) / 4) + 3; c3 += 1) {
  #pragma HLS PIPELINE II=1
    {
      U_t4 in_data;
      U_t4 out_data;
      in_data = local_U[0][c3];
      out_data = in_data;
      fifo_U_drain_out.write(out_data);
    }
  }
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void U_drain_IO_L1_out(hls::stream<U_t4> &fifo_U_drain_in, hls::stream<U_t4> &fifo_U_drain_out, hls::stream<float> &fifo_U_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  U_t4 local_U[1][3];
  #pragma HLS RESOURCE variable=local_U core=RAM_2P_BRAM
  /* Variable Declaration */

  // array
  // io_L3
  // io_L2
  U_drain_IO_L1_out_intra_trans<p0, p1>(
    /* array */ local_U, 
    /* fifo */ fifo_U_drain_local_in
  );
  U_drain_IO_L1_out_inter_trans<p0, p1>(
    /* array */ local_U, 
    /* fifo */ fifo_U_drain_in, 
    /* fifo */ fifo_U_drain_out
  );
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void U_drain_IO_L1_out_wrapper(hls::stream<U_t4> &fifo_U_drain_in, hls::stream<U_t4> &fifo_U_drain_out, hls::stream<float> &fifo_U_drain_local_in)
 {
  U_drain_IO_L1_out<p0, p1>(
    /* fifo */ fifo_U_drain_in, 
    /* fifo */ fifo_U_drain_out, 
    /* fifo */ fifo_U_drain_local_in);
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void U_drain_IO_L1_out_boundary(hls::stream<U_t4> &fifo_U_drain_out, hls::stream<float> &fifo_U_drain_local_in) {
#pragma HLS INLINE
  /* Variable Declaration */
  U_t4 local_U[1][3];
  #pragma HLS RESOURCE variable=local_U core=RAM_2P_BRAM
  /* Variable Declaration */

  // array
  // io_L3
  // io_L2
  U_drain_IO_L1_out_intra_trans<p0, p1>(
    /* array */ local_U, 
    /* fifo */ fifo_U_drain_local_in
  );
  U_drain_IO_L1_out_inter_trans_boundary<p0, p1>(
    /* array */ local_U, 
    /* fifo */ fifo_U_drain_out
  );
}
/* Module Definition */

/* Module Definition */
template<int p0, int p1>
void U_drain_IO_L1_out_boundary_wrapper(hls::stream<U_t4> &fifo_U_drain_out, hls::stream<float> &fifo_U_drain_local_in)
 {
  U_drain_IO_L1_out_boundary<p0, p1>(
    /* fifo */ fifo_U_drain_out, 
    /* fifo */ fifo_U_drain_local_in);
}
/* Module Definition */

/* Module Definition */
template<int p0>
void U_drain_IO_L2_out(hls::stream<U_t4> &fifo_U_drain_in, hls::stream<U_t4> &fifo_U_drain_out, hls::stream<U_t4> &fifo_U_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  /* Variable Declaration */

  // array
  // io_L3
  for (ap_uint<5> c0 = p0; c0 <= 11; c0 += 1) {
    // io_L2
    if (c0 == p0) {
      // io_L1
      // access_coalesce
      for (int c3 = 0; c3 <= -((p0 + 4) / 4) + 3; c3 += 1) {
      #pragma HLS PIPELINE II=1
        {
          U_t4 in_data;
          U_t4 out_data;
          in_data = fifo_U_drain_local_in.read();
          out_data = in_data;
          fifo_U_drain_out.write(out_data);
        }
      }
    } else {
      // io_L1
      // access_coalesce
      for (int c3 = 0; c3 <= -((c0 + 4) / 4) + 3; c3 += 1) {
      #pragma HLS PIPELINE II=1
        {
          U_t4 in_data;
          U_t4 out_data;
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
void U_drain_IO_L2_out_boundary(hls::stream<U_t4> &fifo_U_drain_out, hls::stream<U_t4> &fifo_U_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  /* Variable Declaration */

  // array
  // io_L3
  for (ap_uint<5> c0 = p0; c0 <= 11; c0 += 1)
    if (c0 == p0) {
      // io_L2
      // io_L1
      // access_coalesce
      for (int c3 = 0; c3 <= -((p0 + 4) / 4) + 3; c3 += 1) {
      #pragma HLS PIPELINE II=1
        {
          U_t4 in_data;
          U_t4 out_data;
          in_data = fifo_U_drain_local_in.read();
          out_data = in_data;
          fifo_U_drain_out.write(out_data);
        }
      }
    }
}
/* Module Definition */

/* Module Definition */
void U_drain_IO_L3_out(U_t4 *U, hls::stream<U_t4> &fifo_U_drain_local_in) {
#pragma HLS INLINE OFF
  /* Variable Declaration */
  /* Variable Declaration */

  // array
  // io_L3
  for (ap_uint<5> c0 = 0; c0 <= 11; c0 += 1) {
    // io_L2
    // io_L1
    // access_coalesce
    for (int c3 = 0; c3 <= -((c0 + 4) / 4) + 3; c3 += 1) {
    #pragma HLS PIPELINE II=1
      {
        U_t4 in_data;
        U_t4 out_data;
        in_data = fifo_U_drain_local_in.read();
        out_data = in_data;
        U[(c0 * 12 + (c0 + 4 * c3)) / 4] = out_data;
      }
    }
  }
}
/* Module Definition */

extern "C" {
void kernel0(A_t1 *A, L_t1 *L, U_t4 *U)
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
  /* A_IO_L2_in fifo */ hls::stream<float> fifo_A_A_IO_L2_in_0;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_0 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<float> fifo_A_A_IO_L2_in_1;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_1 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<float> fifo_A_A_IO_L2_in_2;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_2 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<float> fifo_A_A_IO_L2_in_3;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_3 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<float> fifo_A_A_IO_L2_in_4;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_4 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<float> fifo_A_A_IO_L2_in_5;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_5 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<float> fifo_A_A_IO_L2_in_6;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_6 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<float> fifo_A_A_IO_L2_in_7;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_7 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<float> fifo_A_A_IO_L2_in_8;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_8 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<float> fifo_A_A_IO_L2_in_9;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_9 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<float> fifo_A_A_IO_L2_in_10;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_10 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<float> fifo_A_A_IO_L2_in_11;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_11 core=FIFO_SRL
  /* A_IO_L2_in fifo */ hls::stream<float> fifo_A_A_IO_L2_in_12;
  #pragma HLS STREAM variable=fifo_A_A_IO_L2_in_12 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L2_in_12 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_0_0;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_0_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_0_0 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_0_1;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_0_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_0_1 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_1_0;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_1_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_1_0 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_1_1;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_1_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_1_1 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_1_2;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_1_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_1_2 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_2_0;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_2_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_2_0 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_2_1;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_2_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_2_1 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_2_2;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_2_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_2_2 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_2_3;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_2_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_2_3 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_3_0;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_3_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_3_0 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_3_1;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_3_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_3_1 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_3_2;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_3_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_3_2 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_3_3;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_3_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_3_3 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_3_4;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_3_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_3_4 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_4_0;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_4_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_4_0 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_4_1;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_4_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_4_1 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_4_2;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_4_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_4_2 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_4_3;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_4_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_4_3 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_4_4;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_4_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_4_4 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_4_5;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_4_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_4_5 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_5_0;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_5_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_5_0 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_5_1;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_5_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_5_1 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_5_2;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_5_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_5_2 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_5_3;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_5_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_5_3 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_5_4;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_5_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_5_4 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_5_5;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_5_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_5_5 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_5_6;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_5_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_5_6 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_6_0;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_6_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_6_0 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_6_1;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_6_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_6_1 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_6_2;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_6_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_6_2 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_6_3;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_6_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_6_3 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_6_4;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_6_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_6_4 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_6_5;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_6_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_6_5 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_6_6;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_6_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_6_6 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_6_7;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_6_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_6_7 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_7_0;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_7_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_7_0 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_7_1;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_7_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_7_1 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_7_2;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_7_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_7_2 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_7_3;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_7_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_7_3 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_7_4;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_7_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_7_4 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_7_5;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_7_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_7_5 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_7_6;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_7_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_7_6 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_7_7;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_7_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_7_7 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_7_8;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_7_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_7_8 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_8_0;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_8_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_8_0 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_8_1;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_8_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_8_1 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_8_2;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_8_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_8_2 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_8_3;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_8_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_8_3 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_8_4;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_8_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_8_4 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_8_5;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_8_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_8_5 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_8_6;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_8_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_8_6 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_8_7;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_8_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_8_7 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_8_8;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_8_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_8_8 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_8_9;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_8_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_8_9 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_9_0;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_9_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_9_0 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_9_1;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_9_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_9_1 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_9_2;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_9_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_9_2 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_9_3;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_9_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_9_3 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_9_4;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_9_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_9_4 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_9_5;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_9_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_9_5 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_9_6;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_9_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_9_6 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_9_7;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_9_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_9_7 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_9_8;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_9_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_9_8 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_9_9;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_9_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_9_9 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_9_10;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_9_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_9_10 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_10_0;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_10_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_10_0 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_10_1;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_10_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_10_1 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_10_2;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_10_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_10_2 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_10_3;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_10_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_10_3 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_10_4;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_10_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_10_4 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_10_5;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_10_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_10_5 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_10_6;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_10_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_10_6 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_10_7;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_10_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_10_7 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_10_8;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_10_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_10_8 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_10_9;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_10_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_10_9 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_10_10;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_10_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_10_10 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_10_11;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_10_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_10_11 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_11_0;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_11_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_11_0 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_11_1;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_11_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_11_1 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_11_2;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_11_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_11_2 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_11_3;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_11_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_11_3 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_11_4;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_11_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_11_4 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_11_5;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_11_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_11_5 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_11_6;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_11_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_11_6 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_11_7;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_11_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_11_7 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_11_8;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_11_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_11_8 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_11_9;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_11_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_11_9 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_11_10;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_11_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_11_10 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_11_11;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_11_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_11_11 core=FIFO_SRL
  /* A_IO_L1_in fifo */ hls::stream<float> fifo_A_A_IO_L1_in_11_12;
  #pragma HLS STREAM variable=fifo_A_A_IO_L1_in_11_12 depth=2
  #pragma HLS RESOURCE variable=fifo_A_A_IO_L1_in_11_12 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_0_0;
  #pragma HLS STREAM variable=fifo_V_PE_0_0 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_0_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_1_0;
  #pragma HLS STREAM variable=fifo_V_PE_1_0 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_1_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_0_1;
  #pragma HLS STREAM variable=fifo_V_PE_0_1 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_0_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_1_1;
  #pragma HLS STREAM variable=fifo_V_PE_1_1 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_1_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_2_1;
  #pragma HLS STREAM variable=fifo_V_PE_2_1 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_2_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_0_2;
  #pragma HLS STREAM variable=fifo_V_PE_0_2 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_0_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_1_2;
  #pragma HLS STREAM variable=fifo_V_PE_1_2 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_1_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_2_2;
  #pragma HLS STREAM variable=fifo_V_PE_2_2 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_2_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_3_2;
  #pragma HLS STREAM variable=fifo_V_PE_3_2 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_3_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_0_3;
  #pragma HLS STREAM variable=fifo_V_PE_0_3 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_0_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_1_3;
  #pragma HLS STREAM variable=fifo_V_PE_1_3 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_1_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_2_3;
  #pragma HLS STREAM variable=fifo_V_PE_2_3 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_2_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_3_3;
  #pragma HLS STREAM variable=fifo_V_PE_3_3 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_3_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_4_3;
  #pragma HLS STREAM variable=fifo_V_PE_4_3 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_4_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_0_4;
  #pragma HLS STREAM variable=fifo_V_PE_0_4 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_0_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_1_4;
  #pragma HLS STREAM variable=fifo_V_PE_1_4 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_1_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_2_4;
  #pragma HLS STREAM variable=fifo_V_PE_2_4 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_2_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_3_4;
  #pragma HLS STREAM variable=fifo_V_PE_3_4 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_3_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_4_4;
  #pragma HLS STREAM variable=fifo_V_PE_4_4 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_4_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_5_4;
  #pragma HLS STREAM variable=fifo_V_PE_5_4 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_5_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_0_5;
  #pragma HLS STREAM variable=fifo_V_PE_0_5 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_0_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_1_5;
  #pragma HLS STREAM variable=fifo_V_PE_1_5 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_1_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_2_5;
  #pragma HLS STREAM variable=fifo_V_PE_2_5 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_2_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_3_5;
  #pragma HLS STREAM variable=fifo_V_PE_3_5 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_3_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_4_5;
  #pragma HLS STREAM variable=fifo_V_PE_4_5 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_4_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_5_5;
  #pragma HLS STREAM variable=fifo_V_PE_5_5 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_5_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_6_5;
  #pragma HLS STREAM variable=fifo_V_PE_6_5 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_6_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_0_6;
  #pragma HLS STREAM variable=fifo_V_PE_0_6 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_0_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_1_6;
  #pragma HLS STREAM variable=fifo_V_PE_1_6 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_1_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_2_6;
  #pragma HLS STREAM variable=fifo_V_PE_2_6 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_2_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_3_6;
  #pragma HLS STREAM variable=fifo_V_PE_3_6 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_3_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_4_6;
  #pragma HLS STREAM variable=fifo_V_PE_4_6 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_4_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_5_6;
  #pragma HLS STREAM variable=fifo_V_PE_5_6 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_5_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_6_6;
  #pragma HLS STREAM variable=fifo_V_PE_6_6 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_6_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_7_6;
  #pragma HLS STREAM variable=fifo_V_PE_7_6 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_7_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_0_7;
  #pragma HLS STREAM variable=fifo_V_PE_0_7 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_0_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_1_7;
  #pragma HLS STREAM variable=fifo_V_PE_1_7 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_1_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_2_7;
  #pragma HLS STREAM variable=fifo_V_PE_2_7 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_2_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_3_7;
  #pragma HLS STREAM variable=fifo_V_PE_3_7 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_3_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_4_7;
  #pragma HLS STREAM variable=fifo_V_PE_4_7 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_4_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_5_7;
  #pragma HLS STREAM variable=fifo_V_PE_5_7 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_5_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_6_7;
  #pragma HLS STREAM variable=fifo_V_PE_6_7 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_6_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_7_7;
  #pragma HLS STREAM variable=fifo_V_PE_7_7 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_7_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_8_7;
  #pragma HLS STREAM variable=fifo_V_PE_8_7 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_8_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_0_8;
  #pragma HLS STREAM variable=fifo_V_PE_0_8 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_0_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_1_8;
  #pragma HLS STREAM variable=fifo_V_PE_1_8 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_1_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_2_8;
  #pragma HLS STREAM variable=fifo_V_PE_2_8 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_2_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_3_8;
  #pragma HLS STREAM variable=fifo_V_PE_3_8 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_3_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_4_8;
  #pragma HLS STREAM variable=fifo_V_PE_4_8 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_4_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_5_8;
  #pragma HLS STREAM variable=fifo_V_PE_5_8 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_5_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_6_8;
  #pragma HLS STREAM variable=fifo_V_PE_6_8 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_6_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_7_8;
  #pragma HLS STREAM variable=fifo_V_PE_7_8 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_7_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_8_8;
  #pragma HLS STREAM variable=fifo_V_PE_8_8 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_8_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_9_8;
  #pragma HLS STREAM variable=fifo_V_PE_9_8 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_9_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_0_9;
  #pragma HLS STREAM variable=fifo_V_PE_0_9 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_0_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_1_9;
  #pragma HLS STREAM variable=fifo_V_PE_1_9 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_1_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_2_9;
  #pragma HLS STREAM variable=fifo_V_PE_2_9 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_2_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_3_9;
  #pragma HLS STREAM variable=fifo_V_PE_3_9 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_3_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_4_9;
  #pragma HLS STREAM variable=fifo_V_PE_4_9 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_4_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_5_9;
  #pragma HLS STREAM variable=fifo_V_PE_5_9 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_5_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_6_9;
  #pragma HLS STREAM variable=fifo_V_PE_6_9 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_6_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_7_9;
  #pragma HLS STREAM variable=fifo_V_PE_7_9 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_7_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_8_9;
  #pragma HLS STREAM variable=fifo_V_PE_8_9 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_8_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_9_9;
  #pragma HLS STREAM variable=fifo_V_PE_9_9 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_9_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_10_9;
  #pragma HLS STREAM variable=fifo_V_PE_10_9 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_10_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_0_10;
  #pragma HLS STREAM variable=fifo_V_PE_0_10 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_0_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_1_10;
  #pragma HLS STREAM variable=fifo_V_PE_1_10 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_1_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_2_10;
  #pragma HLS STREAM variable=fifo_V_PE_2_10 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_2_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_3_10;
  #pragma HLS STREAM variable=fifo_V_PE_3_10 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_3_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_4_10;
  #pragma HLS STREAM variable=fifo_V_PE_4_10 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_4_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_5_10;
  #pragma HLS STREAM variable=fifo_V_PE_5_10 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_5_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_6_10;
  #pragma HLS STREAM variable=fifo_V_PE_6_10 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_6_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_7_10;
  #pragma HLS STREAM variable=fifo_V_PE_7_10 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_7_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_8_10;
  #pragma HLS STREAM variable=fifo_V_PE_8_10 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_8_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_9_10;
  #pragma HLS STREAM variable=fifo_V_PE_9_10 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_9_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_10_10;
  #pragma HLS STREAM variable=fifo_V_PE_10_10 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_10_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_11_10;
  #pragma HLS STREAM variable=fifo_V_PE_11_10 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_11_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_0_11;
  #pragma HLS STREAM variable=fifo_V_PE_0_11 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_0_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_1_11;
  #pragma HLS STREAM variable=fifo_V_PE_1_11 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_1_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_2_11;
  #pragma HLS STREAM variable=fifo_V_PE_2_11 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_2_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_3_11;
  #pragma HLS STREAM variable=fifo_V_PE_3_11 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_3_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_4_11;
  #pragma HLS STREAM variable=fifo_V_PE_4_11 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_4_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_5_11;
  #pragma HLS STREAM variable=fifo_V_PE_5_11 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_5_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_6_11;
  #pragma HLS STREAM variable=fifo_V_PE_6_11 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_6_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_7_11;
  #pragma HLS STREAM variable=fifo_V_PE_7_11 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_7_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_8_11;
  #pragma HLS STREAM variable=fifo_V_PE_8_11 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_8_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_9_11;
  #pragma HLS STREAM variable=fifo_V_PE_9_11 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_9_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_10_11;
  #pragma HLS STREAM variable=fifo_V_PE_10_11 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_10_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_11_11;
  #pragma HLS STREAM variable=fifo_V_PE_11_11 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_11_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_V_PE_12_11;
  #pragma HLS STREAM variable=fifo_V_PE_12_11 depth=2
  #pragma HLS RESOURCE variable=fifo_V_PE_12_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_0_0;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_0_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_0_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_0_1;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_0_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_0_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_0_2;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_0_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_0_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_0_3;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_0_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_0_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_0_4;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_0_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_0_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_0_5;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_0_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_0_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_0_6;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_0_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_0_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_0_7;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_0_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_0_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_0_8;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_0_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_0_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_0_9;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_0_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_0_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_0_10;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_0_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_0_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_0_11;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_0_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_0_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_0_12;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_0_12 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_0_12 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_1_1;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_1_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_1_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_1_2;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_1_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_1_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_1_3;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_1_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_1_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_1_4;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_1_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_1_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_1_5;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_1_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_1_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_1_6;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_1_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_1_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_1_7;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_1_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_1_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_1_8;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_1_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_1_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_1_9;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_1_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_1_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_1_10;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_1_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_1_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_1_11;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_1_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_1_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_1_12;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_1_12 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_1_12 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_2_2;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_2_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_2_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_2_3;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_2_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_2_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_2_4;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_2_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_2_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_2_5;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_2_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_2_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_2_6;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_2_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_2_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_2_7;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_2_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_2_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_2_8;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_2_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_2_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_2_9;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_2_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_2_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_2_10;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_2_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_2_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_2_11;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_2_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_2_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_2_12;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_2_12 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_2_12 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_3_3;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_3_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_3_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_3_4;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_3_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_3_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_3_5;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_3_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_3_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_3_6;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_3_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_3_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_3_7;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_3_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_3_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_3_8;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_3_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_3_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_3_9;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_3_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_3_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_3_10;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_3_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_3_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_3_11;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_3_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_3_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_3_12;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_3_12 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_3_12 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_4_4;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_4_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_4_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_4_5;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_4_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_4_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_4_6;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_4_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_4_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_4_7;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_4_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_4_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_4_8;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_4_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_4_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_4_9;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_4_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_4_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_4_10;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_4_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_4_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_4_11;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_4_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_4_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_4_12;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_4_12 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_4_12 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_5_5;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_5_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_5_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_5_6;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_5_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_5_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_5_7;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_5_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_5_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_5_8;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_5_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_5_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_5_9;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_5_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_5_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_5_10;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_5_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_5_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_5_11;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_5_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_5_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_5_12;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_5_12 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_5_12 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_6_6;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_6_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_6_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_6_7;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_6_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_6_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_6_8;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_6_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_6_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_6_9;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_6_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_6_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_6_10;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_6_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_6_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_6_11;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_6_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_6_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_6_12;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_6_12 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_6_12 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_7_7;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_7_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_7_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_7_8;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_7_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_7_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_7_9;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_7_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_7_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_7_10;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_7_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_7_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_7_11;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_7_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_7_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_7_12;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_7_12 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_7_12 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_8_8;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_8_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_8_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_8_9;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_8_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_8_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_8_10;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_8_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_8_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_8_11;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_8_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_8_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_8_12;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_8_12 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_8_12 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_9_9;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_9_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_9_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_9_10;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_9_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_9_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_9_11;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_9_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_9_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_9_12;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_9_12 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_9_12 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_10_10;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_10_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_10_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_10_11;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_10_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_10_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_10_12;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_10_12 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_10_12 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_11_11;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_11_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_11_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_tmp_1_PE_11_12;
  #pragma HLS STREAM variable=fifo_U_tmp_1_PE_11_12 depth=2
  #pragma HLS RESOURCE variable=fifo_U_tmp_1_PE_11_12 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_0_0;
  #pragma HLS STREAM variable=fifo_A_PE_0_0 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_0_1;
  #pragma HLS STREAM variable=fifo_A_PE_0_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_1_1;
  #pragma HLS STREAM variable=fifo_A_PE_1_1 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_0_2;
  #pragma HLS STREAM variable=fifo_A_PE_0_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_1_2;
  #pragma HLS STREAM variable=fifo_A_PE_1_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_2_2;
  #pragma HLS STREAM variable=fifo_A_PE_2_2 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_0_3;
  #pragma HLS STREAM variable=fifo_A_PE_0_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_1_3;
  #pragma HLS STREAM variable=fifo_A_PE_1_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_2_3;
  #pragma HLS STREAM variable=fifo_A_PE_2_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_3_3;
  #pragma HLS STREAM variable=fifo_A_PE_3_3 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_0_4;
  #pragma HLS STREAM variable=fifo_A_PE_0_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_1_4;
  #pragma HLS STREAM variable=fifo_A_PE_1_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_2_4;
  #pragma HLS STREAM variable=fifo_A_PE_2_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_3_4;
  #pragma HLS STREAM variable=fifo_A_PE_3_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_4_4;
  #pragma HLS STREAM variable=fifo_A_PE_4_4 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_0_5;
  #pragma HLS STREAM variable=fifo_A_PE_0_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_1_5;
  #pragma HLS STREAM variable=fifo_A_PE_1_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_2_5;
  #pragma HLS STREAM variable=fifo_A_PE_2_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_3_5;
  #pragma HLS STREAM variable=fifo_A_PE_3_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_4_5;
  #pragma HLS STREAM variable=fifo_A_PE_4_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_5_5;
  #pragma HLS STREAM variable=fifo_A_PE_5_5 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_0_6;
  #pragma HLS STREAM variable=fifo_A_PE_0_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_1_6;
  #pragma HLS STREAM variable=fifo_A_PE_1_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_2_6;
  #pragma HLS STREAM variable=fifo_A_PE_2_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_3_6;
  #pragma HLS STREAM variable=fifo_A_PE_3_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_4_6;
  #pragma HLS STREAM variable=fifo_A_PE_4_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_5_6;
  #pragma HLS STREAM variable=fifo_A_PE_5_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_6_6;
  #pragma HLS STREAM variable=fifo_A_PE_6_6 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_0_7;
  #pragma HLS STREAM variable=fifo_A_PE_0_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_1_7;
  #pragma HLS STREAM variable=fifo_A_PE_1_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_2_7;
  #pragma HLS STREAM variable=fifo_A_PE_2_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_3_7;
  #pragma HLS STREAM variable=fifo_A_PE_3_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_4_7;
  #pragma HLS STREAM variable=fifo_A_PE_4_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_5_7;
  #pragma HLS STREAM variable=fifo_A_PE_5_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_6_7;
  #pragma HLS STREAM variable=fifo_A_PE_6_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_7_7;
  #pragma HLS STREAM variable=fifo_A_PE_7_7 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_7_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_0_8;
  #pragma HLS STREAM variable=fifo_A_PE_0_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_1_8;
  #pragma HLS STREAM variable=fifo_A_PE_1_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_2_8;
  #pragma HLS STREAM variable=fifo_A_PE_2_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_3_8;
  #pragma HLS STREAM variable=fifo_A_PE_3_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_4_8;
  #pragma HLS STREAM variable=fifo_A_PE_4_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_5_8;
  #pragma HLS STREAM variable=fifo_A_PE_5_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_6_8;
  #pragma HLS STREAM variable=fifo_A_PE_6_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_7_8;
  #pragma HLS STREAM variable=fifo_A_PE_7_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_7_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_8_8;
  #pragma HLS STREAM variable=fifo_A_PE_8_8 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_8_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_0_9;
  #pragma HLS STREAM variable=fifo_A_PE_0_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_1_9;
  #pragma HLS STREAM variable=fifo_A_PE_1_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_2_9;
  #pragma HLS STREAM variable=fifo_A_PE_2_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_3_9;
  #pragma HLS STREAM variable=fifo_A_PE_3_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_4_9;
  #pragma HLS STREAM variable=fifo_A_PE_4_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_5_9;
  #pragma HLS STREAM variable=fifo_A_PE_5_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_6_9;
  #pragma HLS STREAM variable=fifo_A_PE_6_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_7_9;
  #pragma HLS STREAM variable=fifo_A_PE_7_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_7_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_8_9;
  #pragma HLS STREAM variable=fifo_A_PE_8_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_8_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_9_9;
  #pragma HLS STREAM variable=fifo_A_PE_9_9 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_9_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_0_10;
  #pragma HLS STREAM variable=fifo_A_PE_0_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_1_10;
  #pragma HLS STREAM variable=fifo_A_PE_1_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_2_10;
  #pragma HLS STREAM variable=fifo_A_PE_2_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_3_10;
  #pragma HLS STREAM variable=fifo_A_PE_3_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_4_10;
  #pragma HLS STREAM variable=fifo_A_PE_4_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_5_10;
  #pragma HLS STREAM variable=fifo_A_PE_5_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_6_10;
  #pragma HLS STREAM variable=fifo_A_PE_6_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_7_10;
  #pragma HLS STREAM variable=fifo_A_PE_7_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_7_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_8_10;
  #pragma HLS STREAM variable=fifo_A_PE_8_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_8_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_9_10;
  #pragma HLS STREAM variable=fifo_A_PE_9_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_9_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_10_10;
  #pragma HLS STREAM variable=fifo_A_PE_10_10 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_10_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_0_11;
  #pragma HLS STREAM variable=fifo_A_PE_0_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_0_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_1_11;
  #pragma HLS STREAM variable=fifo_A_PE_1_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_1_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_2_11;
  #pragma HLS STREAM variable=fifo_A_PE_2_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_2_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_3_11;
  #pragma HLS STREAM variable=fifo_A_PE_3_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_3_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_4_11;
  #pragma HLS STREAM variable=fifo_A_PE_4_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_4_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_5_11;
  #pragma HLS STREAM variable=fifo_A_PE_5_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_5_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_6_11;
  #pragma HLS STREAM variable=fifo_A_PE_6_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_6_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_7_11;
  #pragma HLS STREAM variable=fifo_A_PE_7_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_7_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_8_11;
  #pragma HLS STREAM variable=fifo_A_PE_8_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_8_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_9_11;
  #pragma HLS STREAM variable=fifo_A_PE_9_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_9_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_10_11;
  #pragma HLS STREAM variable=fifo_A_PE_10_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_10_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_A_PE_11_11;
  #pragma HLS STREAM variable=fifo_A_PE_11_11 depth=2
  #pragma HLS RESOURCE variable=fifo_A_PE_11_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_0_0;
  #pragma HLS STREAM variable=fifo_L_drain_PE_0_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_0_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_0_1;
  #pragma HLS STREAM variable=fifo_L_drain_PE_0_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_0_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_1_1;
  #pragma HLS STREAM variable=fifo_L_drain_PE_1_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_1_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_0_2;
  #pragma HLS STREAM variable=fifo_L_drain_PE_0_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_0_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_1_2;
  #pragma HLS STREAM variable=fifo_L_drain_PE_1_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_1_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_2_2;
  #pragma HLS STREAM variable=fifo_L_drain_PE_2_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_2_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_0_3;
  #pragma HLS STREAM variable=fifo_L_drain_PE_0_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_0_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_1_3;
  #pragma HLS STREAM variable=fifo_L_drain_PE_1_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_1_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_2_3;
  #pragma HLS STREAM variable=fifo_L_drain_PE_2_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_2_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_3_3;
  #pragma HLS STREAM variable=fifo_L_drain_PE_3_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_3_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_0_4;
  #pragma HLS STREAM variable=fifo_L_drain_PE_0_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_0_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_1_4;
  #pragma HLS STREAM variable=fifo_L_drain_PE_1_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_1_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_2_4;
  #pragma HLS STREAM variable=fifo_L_drain_PE_2_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_2_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_3_4;
  #pragma HLS STREAM variable=fifo_L_drain_PE_3_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_3_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_4_4;
  #pragma HLS STREAM variable=fifo_L_drain_PE_4_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_4_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_0_5;
  #pragma HLS STREAM variable=fifo_L_drain_PE_0_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_0_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_1_5;
  #pragma HLS STREAM variable=fifo_L_drain_PE_1_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_1_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_2_5;
  #pragma HLS STREAM variable=fifo_L_drain_PE_2_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_2_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_3_5;
  #pragma HLS STREAM variable=fifo_L_drain_PE_3_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_3_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_4_5;
  #pragma HLS STREAM variable=fifo_L_drain_PE_4_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_4_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_5_5;
  #pragma HLS STREAM variable=fifo_L_drain_PE_5_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_5_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_0_6;
  #pragma HLS STREAM variable=fifo_L_drain_PE_0_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_0_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_1_6;
  #pragma HLS STREAM variable=fifo_L_drain_PE_1_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_1_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_2_6;
  #pragma HLS STREAM variable=fifo_L_drain_PE_2_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_2_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_3_6;
  #pragma HLS STREAM variable=fifo_L_drain_PE_3_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_3_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_4_6;
  #pragma HLS STREAM variable=fifo_L_drain_PE_4_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_4_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_5_6;
  #pragma HLS STREAM variable=fifo_L_drain_PE_5_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_5_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_6_6;
  #pragma HLS STREAM variable=fifo_L_drain_PE_6_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_6_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_0_7;
  #pragma HLS STREAM variable=fifo_L_drain_PE_0_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_0_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_1_7;
  #pragma HLS STREAM variable=fifo_L_drain_PE_1_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_1_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_2_7;
  #pragma HLS STREAM variable=fifo_L_drain_PE_2_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_2_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_3_7;
  #pragma HLS STREAM variable=fifo_L_drain_PE_3_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_3_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_4_7;
  #pragma HLS STREAM variable=fifo_L_drain_PE_4_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_4_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_5_7;
  #pragma HLS STREAM variable=fifo_L_drain_PE_5_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_5_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_6_7;
  #pragma HLS STREAM variable=fifo_L_drain_PE_6_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_6_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_7_7;
  #pragma HLS STREAM variable=fifo_L_drain_PE_7_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_7_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_0_8;
  #pragma HLS STREAM variable=fifo_L_drain_PE_0_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_0_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_1_8;
  #pragma HLS STREAM variable=fifo_L_drain_PE_1_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_1_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_2_8;
  #pragma HLS STREAM variable=fifo_L_drain_PE_2_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_2_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_3_8;
  #pragma HLS STREAM variable=fifo_L_drain_PE_3_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_3_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_4_8;
  #pragma HLS STREAM variable=fifo_L_drain_PE_4_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_4_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_5_8;
  #pragma HLS STREAM variable=fifo_L_drain_PE_5_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_5_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_6_8;
  #pragma HLS STREAM variable=fifo_L_drain_PE_6_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_6_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_7_8;
  #pragma HLS STREAM variable=fifo_L_drain_PE_7_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_7_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_8_8;
  #pragma HLS STREAM variable=fifo_L_drain_PE_8_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_8_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_0_9;
  #pragma HLS STREAM variable=fifo_L_drain_PE_0_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_0_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_1_9;
  #pragma HLS STREAM variable=fifo_L_drain_PE_1_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_1_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_2_9;
  #pragma HLS STREAM variable=fifo_L_drain_PE_2_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_2_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_3_9;
  #pragma HLS STREAM variable=fifo_L_drain_PE_3_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_3_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_4_9;
  #pragma HLS STREAM variable=fifo_L_drain_PE_4_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_4_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_5_9;
  #pragma HLS STREAM variable=fifo_L_drain_PE_5_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_5_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_6_9;
  #pragma HLS STREAM variable=fifo_L_drain_PE_6_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_6_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_7_9;
  #pragma HLS STREAM variable=fifo_L_drain_PE_7_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_7_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_8_9;
  #pragma HLS STREAM variable=fifo_L_drain_PE_8_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_8_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_9_9;
  #pragma HLS STREAM variable=fifo_L_drain_PE_9_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_9_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_0_10;
  #pragma HLS STREAM variable=fifo_L_drain_PE_0_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_0_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_1_10;
  #pragma HLS STREAM variable=fifo_L_drain_PE_1_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_1_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_2_10;
  #pragma HLS STREAM variable=fifo_L_drain_PE_2_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_2_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_3_10;
  #pragma HLS STREAM variable=fifo_L_drain_PE_3_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_3_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_4_10;
  #pragma HLS STREAM variable=fifo_L_drain_PE_4_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_4_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_5_10;
  #pragma HLS STREAM variable=fifo_L_drain_PE_5_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_5_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_6_10;
  #pragma HLS STREAM variable=fifo_L_drain_PE_6_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_6_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_7_10;
  #pragma HLS STREAM variable=fifo_L_drain_PE_7_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_7_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_8_10;
  #pragma HLS STREAM variable=fifo_L_drain_PE_8_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_8_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_9_10;
  #pragma HLS STREAM variable=fifo_L_drain_PE_9_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_9_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_10_10;
  #pragma HLS STREAM variable=fifo_L_drain_PE_10_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_10_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_0_11;
  #pragma HLS STREAM variable=fifo_L_drain_PE_0_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_0_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_1_11;
  #pragma HLS STREAM variable=fifo_L_drain_PE_1_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_1_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_2_11;
  #pragma HLS STREAM variable=fifo_L_drain_PE_2_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_2_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_3_11;
  #pragma HLS STREAM variable=fifo_L_drain_PE_3_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_3_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_4_11;
  #pragma HLS STREAM variable=fifo_L_drain_PE_4_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_4_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_5_11;
  #pragma HLS STREAM variable=fifo_L_drain_PE_5_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_5_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_6_11;
  #pragma HLS STREAM variable=fifo_L_drain_PE_6_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_6_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_7_11;
  #pragma HLS STREAM variable=fifo_L_drain_PE_7_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_7_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_8_11;
  #pragma HLS STREAM variable=fifo_L_drain_PE_8_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_8_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_9_11;
  #pragma HLS STREAM variable=fifo_L_drain_PE_9_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_9_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_10_11;
  #pragma HLS STREAM variable=fifo_L_drain_PE_10_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_10_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_L_drain_PE_11_11;
  #pragma HLS STREAM variable=fifo_L_drain_PE_11_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_PE_11_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_0_0;
  #pragma HLS STREAM variable=fifo_U_drain_PE_0_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_0_0 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_0_1;
  #pragma HLS STREAM variable=fifo_U_drain_PE_0_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_0_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_1_1;
  #pragma HLS STREAM variable=fifo_U_drain_PE_1_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_1_1 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_0_2;
  #pragma HLS STREAM variable=fifo_U_drain_PE_0_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_0_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_1_2;
  #pragma HLS STREAM variable=fifo_U_drain_PE_1_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_1_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_2_2;
  #pragma HLS STREAM variable=fifo_U_drain_PE_2_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_2_2 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_0_3;
  #pragma HLS STREAM variable=fifo_U_drain_PE_0_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_0_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_1_3;
  #pragma HLS STREAM variable=fifo_U_drain_PE_1_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_1_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_2_3;
  #pragma HLS STREAM variable=fifo_U_drain_PE_2_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_2_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_3_3;
  #pragma HLS STREAM variable=fifo_U_drain_PE_3_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_3_3 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_0_4;
  #pragma HLS STREAM variable=fifo_U_drain_PE_0_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_0_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_1_4;
  #pragma HLS STREAM variable=fifo_U_drain_PE_1_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_1_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_2_4;
  #pragma HLS STREAM variable=fifo_U_drain_PE_2_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_2_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_3_4;
  #pragma HLS STREAM variable=fifo_U_drain_PE_3_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_3_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_4_4;
  #pragma HLS STREAM variable=fifo_U_drain_PE_4_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_4_4 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_0_5;
  #pragma HLS STREAM variable=fifo_U_drain_PE_0_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_0_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_1_5;
  #pragma HLS STREAM variable=fifo_U_drain_PE_1_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_1_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_2_5;
  #pragma HLS STREAM variable=fifo_U_drain_PE_2_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_2_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_3_5;
  #pragma HLS STREAM variable=fifo_U_drain_PE_3_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_3_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_4_5;
  #pragma HLS STREAM variable=fifo_U_drain_PE_4_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_4_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_5_5;
  #pragma HLS STREAM variable=fifo_U_drain_PE_5_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_5_5 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_0_6;
  #pragma HLS STREAM variable=fifo_U_drain_PE_0_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_0_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_1_6;
  #pragma HLS STREAM variable=fifo_U_drain_PE_1_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_1_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_2_6;
  #pragma HLS STREAM variable=fifo_U_drain_PE_2_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_2_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_3_6;
  #pragma HLS STREAM variable=fifo_U_drain_PE_3_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_3_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_4_6;
  #pragma HLS STREAM variable=fifo_U_drain_PE_4_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_4_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_5_6;
  #pragma HLS STREAM variable=fifo_U_drain_PE_5_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_5_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_6_6;
  #pragma HLS STREAM variable=fifo_U_drain_PE_6_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_6_6 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_0_7;
  #pragma HLS STREAM variable=fifo_U_drain_PE_0_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_0_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_1_7;
  #pragma HLS STREAM variable=fifo_U_drain_PE_1_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_1_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_2_7;
  #pragma HLS STREAM variable=fifo_U_drain_PE_2_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_2_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_3_7;
  #pragma HLS STREAM variable=fifo_U_drain_PE_3_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_3_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_4_7;
  #pragma HLS STREAM variable=fifo_U_drain_PE_4_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_4_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_5_7;
  #pragma HLS STREAM variable=fifo_U_drain_PE_5_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_5_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_6_7;
  #pragma HLS STREAM variable=fifo_U_drain_PE_6_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_6_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_7_7;
  #pragma HLS STREAM variable=fifo_U_drain_PE_7_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_7_7 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_0_8;
  #pragma HLS STREAM variable=fifo_U_drain_PE_0_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_0_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_1_8;
  #pragma HLS STREAM variable=fifo_U_drain_PE_1_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_1_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_2_8;
  #pragma HLS STREAM variable=fifo_U_drain_PE_2_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_2_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_3_8;
  #pragma HLS STREAM variable=fifo_U_drain_PE_3_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_3_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_4_8;
  #pragma HLS STREAM variable=fifo_U_drain_PE_4_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_4_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_5_8;
  #pragma HLS STREAM variable=fifo_U_drain_PE_5_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_5_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_6_8;
  #pragma HLS STREAM variable=fifo_U_drain_PE_6_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_6_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_7_8;
  #pragma HLS STREAM variable=fifo_U_drain_PE_7_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_7_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_8_8;
  #pragma HLS STREAM variable=fifo_U_drain_PE_8_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_8_8 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_0_9;
  #pragma HLS STREAM variable=fifo_U_drain_PE_0_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_0_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_1_9;
  #pragma HLS STREAM variable=fifo_U_drain_PE_1_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_1_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_2_9;
  #pragma HLS STREAM variable=fifo_U_drain_PE_2_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_2_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_3_9;
  #pragma HLS STREAM variable=fifo_U_drain_PE_3_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_3_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_4_9;
  #pragma HLS STREAM variable=fifo_U_drain_PE_4_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_4_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_5_9;
  #pragma HLS STREAM variable=fifo_U_drain_PE_5_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_5_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_6_9;
  #pragma HLS STREAM variable=fifo_U_drain_PE_6_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_6_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_7_9;
  #pragma HLS STREAM variable=fifo_U_drain_PE_7_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_7_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_8_9;
  #pragma HLS STREAM variable=fifo_U_drain_PE_8_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_8_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_9_9;
  #pragma HLS STREAM variable=fifo_U_drain_PE_9_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_9_9 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_0_10;
  #pragma HLS STREAM variable=fifo_U_drain_PE_0_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_0_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_1_10;
  #pragma HLS STREAM variable=fifo_U_drain_PE_1_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_1_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_2_10;
  #pragma HLS STREAM variable=fifo_U_drain_PE_2_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_2_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_3_10;
  #pragma HLS STREAM variable=fifo_U_drain_PE_3_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_3_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_4_10;
  #pragma HLS STREAM variable=fifo_U_drain_PE_4_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_4_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_5_10;
  #pragma HLS STREAM variable=fifo_U_drain_PE_5_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_5_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_6_10;
  #pragma HLS STREAM variable=fifo_U_drain_PE_6_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_6_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_7_10;
  #pragma HLS STREAM variable=fifo_U_drain_PE_7_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_7_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_8_10;
  #pragma HLS STREAM variable=fifo_U_drain_PE_8_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_8_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_9_10;
  #pragma HLS STREAM variable=fifo_U_drain_PE_9_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_9_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_10_10;
  #pragma HLS STREAM variable=fifo_U_drain_PE_10_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_10_10 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_0_11;
  #pragma HLS STREAM variable=fifo_U_drain_PE_0_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_0_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_1_11;
  #pragma HLS STREAM variable=fifo_U_drain_PE_1_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_1_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_2_11;
  #pragma HLS STREAM variable=fifo_U_drain_PE_2_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_2_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_3_11;
  #pragma HLS STREAM variable=fifo_U_drain_PE_3_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_3_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_4_11;
  #pragma HLS STREAM variable=fifo_U_drain_PE_4_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_4_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_5_11;
  #pragma HLS STREAM variable=fifo_U_drain_PE_5_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_5_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_6_11;
  #pragma HLS STREAM variable=fifo_U_drain_PE_6_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_6_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_7_11;
  #pragma HLS STREAM variable=fifo_U_drain_PE_7_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_7_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_8_11;
  #pragma HLS STREAM variable=fifo_U_drain_PE_8_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_8_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_9_11;
  #pragma HLS STREAM variable=fifo_U_drain_PE_9_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_9_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_10_11;
  #pragma HLS STREAM variable=fifo_U_drain_PE_10_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_10_11 core=FIFO_SRL
  /* PE fifo */ hls::stream<float> fifo_U_drain_PE_11_11;
  #pragma HLS STREAM variable=fifo_U_drain_PE_11_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_PE_11_11 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_0_0;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_0_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_0_0 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_0_1;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_0_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_0_1 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_1_0;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_1_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_1_0 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_1_1;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_1_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_1_1 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_1_2;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_1_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_1_2 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_2_0;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_2_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_2_0 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_2_1;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_2_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_2_1 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_2_2;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_2_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_2_2 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_2_3;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_2_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_2_3 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_3_0;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_3_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_3_0 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_3_1;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_3_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_3_1 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_3_2;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_3_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_3_2 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_3_3;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_3_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_3_3 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_3_4;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_3_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_3_4 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_4_0;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_4_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_4_0 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_4_1;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_4_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_4_1 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_4_2;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_4_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_4_2 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_4_3;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_4_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_4_3 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_4_4;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_4_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_4_4 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_4_5;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_4_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_4_5 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_5_0;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_5_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_5_0 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_5_1;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_5_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_5_1 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_5_2;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_5_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_5_2 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_5_3;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_5_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_5_3 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_5_4;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_5_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_5_4 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_5_5;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_5_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_5_5 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_5_6;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_5_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_5_6 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_6_0;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_6_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_6_0 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_6_1;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_6_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_6_1 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_6_2;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_6_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_6_2 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_6_3;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_6_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_6_3 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_6_4;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_6_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_6_4 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_6_5;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_6_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_6_5 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_6_6;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_6_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_6_6 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_6_7;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_6_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_6_7 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_7_0;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_7_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_7_0 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_7_1;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_7_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_7_1 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_7_2;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_7_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_7_2 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_7_3;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_7_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_7_3 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_7_4;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_7_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_7_4 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_7_5;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_7_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_7_5 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_7_6;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_7_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_7_6 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_7_7;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_7_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_7_7 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_7_8;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_7_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_7_8 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_8_0;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_8_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_8_0 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_8_1;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_8_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_8_1 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_8_2;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_8_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_8_2 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_8_3;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_8_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_8_3 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_8_4;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_8_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_8_4 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_8_5;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_8_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_8_5 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_8_6;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_8_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_8_6 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_8_7;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_8_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_8_7 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_8_8;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_8_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_8_8 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_8_9;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_8_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_8_9 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_9_0;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_9_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_9_0 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_9_1;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_9_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_9_1 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_9_2;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_9_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_9_2 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_9_3;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_9_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_9_3 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_9_4;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_9_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_9_4 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_9_5;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_9_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_9_5 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_9_6;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_9_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_9_6 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_9_7;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_9_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_9_7 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_9_8;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_9_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_9_8 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_9_9;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_9_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_9_9 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_9_10;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_9_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_9_10 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_10_0;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_10_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_10_0 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_10_1;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_10_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_10_1 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_10_2;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_10_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_10_2 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_10_3;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_10_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_10_3 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_10_4;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_10_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_10_4 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_10_5;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_10_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_10_5 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_10_6;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_10_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_10_6 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_10_7;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_10_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_10_7 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_10_8;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_10_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_10_8 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_10_9;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_10_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_10_9 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_10_10;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_10_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_10_10 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_10_11;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_10_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_10_11 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_11_0;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_11_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_11_0 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_11_1;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_11_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_11_1 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_11_2;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_11_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_11_2 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_11_3;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_11_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_11_3 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_11_4;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_11_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_11_4 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_11_5;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_11_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_11_5 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_11_6;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_11_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_11_6 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_11_7;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_11_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_11_7 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_11_8;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_11_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_11_8 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_11_9;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_11_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_11_9 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_11_10;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_11_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_11_10 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_11_11;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_11_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_11_11 core=FIFO_SRL
  /* L_drain_IO_L1_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L1_out_11_12;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L1_out_11_12 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L1_out_11_12 core=FIFO_SRL
  /* L_drain_IO_L2_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L2_out_0;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L2_out_0 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L2_out_0 core=FIFO_SRL
  /* L_drain_IO_L2_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L2_out_1;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L2_out_1 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L2_out_1 core=FIFO_SRL
  /* L_drain_IO_L2_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L2_out_2;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L2_out_2 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L2_out_2 core=FIFO_SRL
  /* L_drain_IO_L2_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L2_out_3;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L2_out_3 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L2_out_3 core=FIFO_SRL
  /* L_drain_IO_L2_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L2_out_4;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L2_out_4 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L2_out_4 core=FIFO_SRL
  /* L_drain_IO_L2_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L2_out_5;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L2_out_5 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L2_out_5 core=FIFO_SRL
  /* L_drain_IO_L2_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L2_out_6;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L2_out_6 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L2_out_6 core=FIFO_SRL
  /* L_drain_IO_L2_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L2_out_7;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L2_out_7 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L2_out_7 core=FIFO_SRL
  /* L_drain_IO_L2_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L2_out_8;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L2_out_8 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L2_out_8 core=FIFO_SRL
  /* L_drain_IO_L2_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L2_out_9;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L2_out_9 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L2_out_9 core=FIFO_SRL
  /* L_drain_IO_L2_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L2_out_10;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L2_out_10 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L2_out_10 core=FIFO_SRL
  /* L_drain_IO_L2_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L2_out_11;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L2_out_11 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L2_out_11 core=FIFO_SRL
  /* L_drain_IO_L2_out fifo */ hls::stream<float> fifo_L_drain_L_drain_IO_L2_out_12;
  #pragma HLS STREAM variable=fifo_L_drain_L_drain_IO_L2_out_12 depth=2
  #pragma HLS RESOURCE variable=fifo_L_drain_L_drain_IO_L2_out_12 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_0_0;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_0_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_0_0 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_0_1;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_0_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_0_1 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_1_0;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_1_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_1_0 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_1_1;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_1_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_1_1 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_1_2;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_1_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_1_2 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_2_0;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_2_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_2_0 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_2_1;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_2_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_2_1 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_2_2;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_2_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_2_2 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_2_3;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_2_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_2_3 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_3_0;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_3_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_3_0 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_3_1;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_3_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_3_1 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_3_2;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_3_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_3_2 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_3_3;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_3_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_3_3 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_3_4;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_3_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_3_4 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_4_0;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_4_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_4_0 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_4_1;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_4_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_4_1 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_4_2;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_4_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_4_2 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_4_3;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_4_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_4_3 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_4_4;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_4_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_4_4 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_4_5;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_4_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_4_5 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_5_0;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_5_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_5_0 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_5_1;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_5_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_5_1 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_5_2;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_5_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_5_2 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_5_3;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_5_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_5_3 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_5_4;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_5_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_5_4 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_5_5;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_5_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_5_5 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_5_6;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_5_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_5_6 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_6_0;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_6_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_6_0 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_6_1;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_6_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_6_1 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_6_2;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_6_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_6_2 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_6_3;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_6_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_6_3 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_6_4;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_6_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_6_4 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_6_5;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_6_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_6_5 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_6_6;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_6_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_6_6 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_6_7;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_6_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_6_7 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_7_0;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_7_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_7_0 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_7_1;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_7_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_7_1 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_7_2;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_7_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_7_2 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_7_3;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_7_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_7_3 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_7_4;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_7_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_7_4 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_7_5;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_7_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_7_5 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_7_6;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_7_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_7_6 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_7_7;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_7_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_7_7 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_7_8;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_7_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_7_8 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_8_0;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_8_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_8_0 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_8_1;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_8_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_8_1 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_8_2;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_8_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_8_2 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_8_3;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_8_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_8_3 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_8_4;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_8_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_8_4 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_8_5;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_8_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_8_5 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_8_6;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_8_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_8_6 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_8_7;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_8_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_8_7 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_8_8;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_8_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_8_8 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_8_9;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_8_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_8_9 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_9_0;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_9_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_9_0 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_9_1;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_9_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_9_1 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_9_2;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_9_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_9_2 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_9_3;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_9_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_9_3 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_9_4;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_9_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_9_4 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_9_5;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_9_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_9_5 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_9_6;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_9_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_9_6 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_9_7;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_9_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_9_7 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_9_8;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_9_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_9_8 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_9_9;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_9_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_9_9 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_9_10;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_9_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_9_10 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_10_0;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_10_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_10_0 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_10_1;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_10_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_10_1 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_10_2;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_10_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_10_2 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_10_3;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_10_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_10_3 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_10_4;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_10_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_10_4 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_10_5;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_10_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_10_5 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_10_6;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_10_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_10_6 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_10_7;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_10_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_10_7 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_10_8;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_10_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_10_8 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_10_9;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_10_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_10_9 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_10_10;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_10_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_10_10 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_10_11;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_10_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_10_11 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_11_0;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_11_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_11_0 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_11_1;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_11_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_11_1 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_11_2;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_11_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_11_2 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_11_3;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_11_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_11_3 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_11_4;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_11_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_11_4 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_11_5;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_11_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_11_5 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_11_6;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_11_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_11_6 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_11_7;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_11_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_11_7 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_11_8;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_11_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_11_8 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_11_9;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_11_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_11_9 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_11_10;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_11_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_11_10 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_11_11;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_11_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_11_11 core=FIFO_SRL
  /* U_drain_IO_L1_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L1_out_11_12;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L1_out_11_12 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L1_out_11_12 core=FIFO_SRL
  /* U_drain_IO_L2_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L2_out_0;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L2_out_0 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L2_out_0 core=FIFO_SRL
  /* U_drain_IO_L2_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L2_out_1;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L2_out_1 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L2_out_1 core=FIFO_SRL
  /* U_drain_IO_L2_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L2_out_2;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L2_out_2 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L2_out_2 core=FIFO_SRL
  /* U_drain_IO_L2_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L2_out_3;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L2_out_3 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L2_out_3 core=FIFO_SRL
  /* U_drain_IO_L2_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L2_out_4;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L2_out_4 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L2_out_4 core=FIFO_SRL
  /* U_drain_IO_L2_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L2_out_5;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L2_out_5 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L2_out_5 core=FIFO_SRL
  /* U_drain_IO_L2_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L2_out_6;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L2_out_6 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L2_out_6 core=FIFO_SRL
  /* U_drain_IO_L2_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L2_out_7;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L2_out_7 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L2_out_7 core=FIFO_SRL
  /* U_drain_IO_L2_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L2_out_8;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L2_out_8 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L2_out_8 core=FIFO_SRL
  /* U_drain_IO_L2_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L2_out_9;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L2_out_9 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L2_out_9 core=FIFO_SRL
  /* U_drain_IO_L2_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L2_out_10;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L2_out_10 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L2_out_10 core=FIFO_SRL
  /* U_drain_IO_L2_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L2_out_11;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L2_out_11 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L2_out_11 core=FIFO_SRL
  /* U_drain_IO_L2_out fifo */ hls::stream<U_t4> fifo_U_drain_U_drain_IO_L2_out_12;
  #pragma HLS STREAM variable=fifo_U_drain_U_drain_IO_L2_out_12 depth=2
  #pragma HLS RESOURCE variable=fifo_U_drain_U_drain_IO_L2_out_12 core=FIFO_SRL
  /* FIFO Declaration */

  /* Module Call */
  A_IO_L3_in(
    /* array */ A,
    /* fifo */ fifo_A_A_IO_L2_in_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in<0>(
    /* fifo */ fifo_A_A_IO_L2_in_0,
    /* fifo */ fifo_A_A_IO_L2_in_1,
    /* fifo */ fifo_A_A_IO_L1_in_0_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in<1>(
    /* fifo */ fifo_A_A_IO_L2_in_1,
    /* fifo */ fifo_A_A_IO_L2_in_2,
    /* fifo */ fifo_A_A_IO_L1_in_1_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in<2>(
    /* fifo */ fifo_A_A_IO_L2_in_2,
    /* fifo */ fifo_A_A_IO_L2_in_3,
    /* fifo */ fifo_A_A_IO_L1_in_2_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in<3>(
    /* fifo */ fifo_A_A_IO_L2_in_3,
    /* fifo */ fifo_A_A_IO_L2_in_4,
    /* fifo */ fifo_A_A_IO_L1_in_3_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in<4>(
    /* fifo */ fifo_A_A_IO_L2_in_4,
    /* fifo */ fifo_A_A_IO_L2_in_5,
    /* fifo */ fifo_A_A_IO_L1_in_4_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in<5>(
    /* fifo */ fifo_A_A_IO_L2_in_5,
    /* fifo */ fifo_A_A_IO_L2_in_6,
    /* fifo */ fifo_A_A_IO_L1_in_5_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in<6>(
    /* fifo */ fifo_A_A_IO_L2_in_6,
    /* fifo */ fifo_A_A_IO_L2_in_7,
    /* fifo */ fifo_A_A_IO_L1_in_6_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in<7>(
    /* fifo */ fifo_A_A_IO_L2_in_7,
    /* fifo */ fifo_A_A_IO_L2_in_8,
    /* fifo */ fifo_A_A_IO_L1_in_7_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in<8>(
    /* fifo */ fifo_A_A_IO_L2_in_8,
    /* fifo */ fifo_A_A_IO_L2_in_9,
    /* fifo */ fifo_A_A_IO_L1_in_8_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in<9>(
    /* fifo */ fifo_A_A_IO_L2_in_9,
    /* fifo */ fifo_A_A_IO_L2_in_10,
    /* fifo */ fifo_A_A_IO_L1_in_9_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in<10>(
    /* fifo */ fifo_A_A_IO_L2_in_10,
    /* fifo */ fifo_A_A_IO_L2_in_11,
    /* fifo */ fifo_A_A_IO_L1_in_10_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L2_in_boundary<11>(
    /* fifo */ fifo_A_A_IO_L2_in_11,
    /* fifo */ fifo_A_A_IO_L1_in_11_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_boundary_wrapper<0, 0>(
    /* fifo */ fifo_A_A_IO_L1_in_0_0,
    /* fifo */ fifo_A_PE_0_0
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_boundary_wrapper<1, 0>(
    /* fifo */ fifo_A_A_IO_L1_in_1_0,
    /* fifo */ fifo_A_PE_0_1
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_boundary_wrapper<2, 0>(
    /* fifo */ fifo_A_A_IO_L1_in_2_0,
    /* fifo */ fifo_A_PE_0_2
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_boundary_wrapper<3, 0>(
    /* fifo */ fifo_A_A_IO_L1_in_3_0,
    /* fifo */ fifo_A_PE_0_3
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_boundary_wrapper<4, 0>(
    /* fifo */ fifo_A_A_IO_L1_in_4_0,
    /* fifo */ fifo_A_PE_0_4
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_boundary_wrapper<5, 0>(
    /* fifo */ fifo_A_A_IO_L1_in_5_0,
    /* fifo */ fifo_A_PE_0_5
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_boundary_wrapper<6, 0>(
    /* fifo */ fifo_A_A_IO_L1_in_6_0,
    /* fifo */ fifo_A_PE_0_6
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_boundary_wrapper<7, 0>(
    /* fifo */ fifo_A_A_IO_L1_in_7_0,
    /* fifo */ fifo_A_PE_0_7
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_boundary_wrapper<8, 0>(
    /* fifo */ fifo_A_A_IO_L1_in_8_0,
    /* fifo */ fifo_A_PE_0_8
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_boundary_wrapper<9, 0>(
    /* fifo */ fifo_A_A_IO_L1_in_9_0,
    /* fifo */ fifo_A_PE_0_9
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_boundary_wrapper<10, 0>(
    /* fifo */ fifo_A_A_IO_L1_in_10_0,
    /* fifo */ fifo_A_PE_0_10
  );
  /* Module Call */

  /* Module Call */
  A_IO_L1_in_boundary_wrapper<11, 0>(
    /* fifo */ fifo_A_A_IO_L1_in_11_0,
    /* fifo */ fifo_A_PE_0_11
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<0, 0>(
    /* fifo */ fifo_V_PE_0_0,
    /* fifo */ fifo_V_PE_1_0,
    /* fifo */ fifo_U_tmp_1_PE_0_0,
    /* fifo */ fifo_U_tmp_1_PE_0_1,
    /* fifo */ fifo_A_PE_0_0,
    /* fifo */ fifo_L_drain_PE_0_0,
    /* fifo */ fifo_U_drain_PE_0_0
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<0, 1>(
    /* fifo */ fifo_V_PE_0_1,
    /* fifo */ fifo_V_PE_1_1,
    /* fifo */ fifo_U_tmp_1_PE_0_1,
    /* fifo */ fifo_U_tmp_1_PE_0_2,
    /* fifo */ fifo_A_PE_0_1,
    /* fifo */ fifo_L_drain_PE_0_1,
    /* fifo */ fifo_U_drain_PE_0_1
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<0, 2>(
    /* fifo */ fifo_V_PE_0_2,
    /* fifo */ fifo_V_PE_1_2,
    /* fifo */ fifo_U_tmp_1_PE_0_2,
    /* fifo */ fifo_U_tmp_1_PE_0_3,
    /* fifo */ fifo_A_PE_0_2,
    /* fifo */ fifo_L_drain_PE_0_2,
    /* fifo */ fifo_U_drain_PE_0_2
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<0, 3>(
    /* fifo */ fifo_V_PE_0_3,
    /* fifo */ fifo_V_PE_1_3,
    /* fifo */ fifo_U_tmp_1_PE_0_3,
    /* fifo */ fifo_U_tmp_1_PE_0_4,
    /* fifo */ fifo_A_PE_0_3,
    /* fifo */ fifo_L_drain_PE_0_3,
    /* fifo */ fifo_U_drain_PE_0_3
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<0, 4>(
    /* fifo */ fifo_V_PE_0_4,
    /* fifo */ fifo_V_PE_1_4,
    /* fifo */ fifo_U_tmp_1_PE_0_4,
    /* fifo */ fifo_U_tmp_1_PE_0_5,
    /* fifo */ fifo_A_PE_0_4,
    /* fifo */ fifo_L_drain_PE_0_4,
    /* fifo */ fifo_U_drain_PE_0_4
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<0, 5>(
    /* fifo */ fifo_V_PE_0_5,
    /* fifo */ fifo_V_PE_1_5,
    /* fifo */ fifo_U_tmp_1_PE_0_5,
    /* fifo */ fifo_U_tmp_1_PE_0_6,
    /* fifo */ fifo_A_PE_0_5,
    /* fifo */ fifo_L_drain_PE_0_5,
    /* fifo */ fifo_U_drain_PE_0_5
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<0, 6>(
    /* fifo */ fifo_V_PE_0_6,
    /* fifo */ fifo_V_PE_1_6,
    /* fifo */ fifo_U_tmp_1_PE_0_6,
    /* fifo */ fifo_U_tmp_1_PE_0_7,
    /* fifo */ fifo_A_PE_0_6,
    /* fifo */ fifo_L_drain_PE_0_6,
    /* fifo */ fifo_U_drain_PE_0_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<0, 7>(
    /* fifo */ fifo_V_PE_0_7,
    /* fifo */ fifo_V_PE_1_7,
    /* fifo */ fifo_U_tmp_1_PE_0_7,
    /* fifo */ fifo_U_tmp_1_PE_0_8,
    /* fifo */ fifo_A_PE_0_7,
    /* fifo */ fifo_L_drain_PE_0_7,
    /* fifo */ fifo_U_drain_PE_0_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<0, 8>(
    /* fifo */ fifo_V_PE_0_8,
    /* fifo */ fifo_V_PE_1_8,
    /* fifo */ fifo_U_tmp_1_PE_0_8,
    /* fifo */ fifo_U_tmp_1_PE_0_9,
    /* fifo */ fifo_A_PE_0_8,
    /* fifo */ fifo_L_drain_PE_0_8,
    /* fifo */ fifo_U_drain_PE_0_8
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<0, 9>(
    /* fifo */ fifo_V_PE_0_9,
    /* fifo */ fifo_V_PE_1_9,
    /* fifo */ fifo_U_tmp_1_PE_0_9,
    /* fifo */ fifo_U_tmp_1_PE_0_10,
    /* fifo */ fifo_A_PE_0_9,
    /* fifo */ fifo_L_drain_PE_0_9,
    /* fifo */ fifo_U_drain_PE_0_9
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<0, 10>(
    /* fifo */ fifo_V_PE_0_10,
    /* fifo */ fifo_V_PE_1_10,
    /* fifo */ fifo_U_tmp_1_PE_0_10,
    /* fifo */ fifo_U_tmp_1_PE_0_11,
    /* fifo */ fifo_A_PE_0_10,
    /* fifo */ fifo_L_drain_PE_0_10,
    /* fifo */ fifo_U_drain_PE_0_10
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<0, 11>(
    /* fifo */ fifo_V_PE_0_11,
    /* fifo */ fifo_V_PE_1_11,
    /* fifo */ fifo_U_tmp_1_PE_0_11,
    /* fifo */ fifo_U_tmp_1_PE_0_12,
    /* fifo */ fifo_A_PE_0_11,
    /* fifo */ fifo_L_drain_PE_0_11,
    /* fifo */ fifo_U_drain_PE_0_11
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<1, 1>(
    /* fifo */ fifo_V_PE_1_1,
    /* fifo */ fifo_V_PE_2_1,
    /* fifo */ fifo_U_tmp_1_PE_1_1,
    /* fifo */ fifo_U_tmp_1_PE_1_2,
    /* fifo */ fifo_A_PE_1_1,
    /* fifo */ fifo_L_drain_PE_1_1,
    /* fifo */ fifo_U_drain_PE_1_1
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<1, 2>(
    /* fifo */ fifo_V_PE_1_2,
    /* fifo */ fifo_V_PE_2_2,
    /* fifo */ fifo_U_tmp_1_PE_1_2,
    /* fifo */ fifo_U_tmp_1_PE_1_3,
    /* fifo */ fifo_A_PE_1_2,
    /* fifo */ fifo_L_drain_PE_1_2,
    /* fifo */ fifo_U_drain_PE_1_2
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<1, 3>(
    /* fifo */ fifo_V_PE_1_3,
    /* fifo */ fifo_V_PE_2_3,
    /* fifo */ fifo_U_tmp_1_PE_1_3,
    /* fifo */ fifo_U_tmp_1_PE_1_4,
    /* fifo */ fifo_A_PE_1_3,
    /* fifo */ fifo_L_drain_PE_1_3,
    /* fifo */ fifo_U_drain_PE_1_3
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<1, 4>(
    /* fifo */ fifo_V_PE_1_4,
    /* fifo */ fifo_V_PE_2_4,
    /* fifo */ fifo_U_tmp_1_PE_1_4,
    /* fifo */ fifo_U_tmp_1_PE_1_5,
    /* fifo */ fifo_A_PE_1_4,
    /* fifo */ fifo_L_drain_PE_1_4,
    /* fifo */ fifo_U_drain_PE_1_4
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<1, 5>(
    /* fifo */ fifo_V_PE_1_5,
    /* fifo */ fifo_V_PE_2_5,
    /* fifo */ fifo_U_tmp_1_PE_1_5,
    /* fifo */ fifo_U_tmp_1_PE_1_6,
    /* fifo */ fifo_A_PE_1_5,
    /* fifo */ fifo_L_drain_PE_1_5,
    /* fifo */ fifo_U_drain_PE_1_5
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<1, 6>(
    /* fifo */ fifo_V_PE_1_6,
    /* fifo */ fifo_V_PE_2_6,
    /* fifo */ fifo_U_tmp_1_PE_1_6,
    /* fifo */ fifo_U_tmp_1_PE_1_7,
    /* fifo */ fifo_A_PE_1_6,
    /* fifo */ fifo_L_drain_PE_1_6,
    /* fifo */ fifo_U_drain_PE_1_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<1, 7>(
    /* fifo */ fifo_V_PE_1_7,
    /* fifo */ fifo_V_PE_2_7,
    /* fifo */ fifo_U_tmp_1_PE_1_7,
    /* fifo */ fifo_U_tmp_1_PE_1_8,
    /* fifo */ fifo_A_PE_1_7,
    /* fifo */ fifo_L_drain_PE_1_7,
    /* fifo */ fifo_U_drain_PE_1_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<1, 8>(
    /* fifo */ fifo_V_PE_1_8,
    /* fifo */ fifo_V_PE_2_8,
    /* fifo */ fifo_U_tmp_1_PE_1_8,
    /* fifo */ fifo_U_tmp_1_PE_1_9,
    /* fifo */ fifo_A_PE_1_8,
    /* fifo */ fifo_L_drain_PE_1_8,
    /* fifo */ fifo_U_drain_PE_1_8
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<1, 9>(
    /* fifo */ fifo_V_PE_1_9,
    /* fifo */ fifo_V_PE_2_9,
    /* fifo */ fifo_U_tmp_1_PE_1_9,
    /* fifo */ fifo_U_tmp_1_PE_1_10,
    /* fifo */ fifo_A_PE_1_9,
    /* fifo */ fifo_L_drain_PE_1_9,
    /* fifo */ fifo_U_drain_PE_1_9
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<1, 10>(
    /* fifo */ fifo_V_PE_1_10,
    /* fifo */ fifo_V_PE_2_10,
    /* fifo */ fifo_U_tmp_1_PE_1_10,
    /* fifo */ fifo_U_tmp_1_PE_1_11,
    /* fifo */ fifo_A_PE_1_10,
    /* fifo */ fifo_L_drain_PE_1_10,
    /* fifo */ fifo_U_drain_PE_1_10
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<1, 11>(
    /* fifo */ fifo_V_PE_1_11,
    /* fifo */ fifo_V_PE_2_11,
    /* fifo */ fifo_U_tmp_1_PE_1_11,
    /* fifo */ fifo_U_tmp_1_PE_1_12,
    /* fifo */ fifo_A_PE_1_11,
    /* fifo */ fifo_L_drain_PE_1_11,
    /* fifo */ fifo_U_drain_PE_1_11
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<2, 2>(
    /* fifo */ fifo_V_PE_2_2,
    /* fifo */ fifo_V_PE_3_2,
    /* fifo */ fifo_U_tmp_1_PE_2_2,
    /* fifo */ fifo_U_tmp_1_PE_2_3,
    /* fifo */ fifo_A_PE_2_2,
    /* fifo */ fifo_L_drain_PE_2_2,
    /* fifo */ fifo_U_drain_PE_2_2
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<2, 3>(
    /* fifo */ fifo_V_PE_2_3,
    /* fifo */ fifo_V_PE_3_3,
    /* fifo */ fifo_U_tmp_1_PE_2_3,
    /* fifo */ fifo_U_tmp_1_PE_2_4,
    /* fifo */ fifo_A_PE_2_3,
    /* fifo */ fifo_L_drain_PE_2_3,
    /* fifo */ fifo_U_drain_PE_2_3
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<2, 4>(
    /* fifo */ fifo_V_PE_2_4,
    /* fifo */ fifo_V_PE_3_4,
    /* fifo */ fifo_U_tmp_1_PE_2_4,
    /* fifo */ fifo_U_tmp_1_PE_2_5,
    /* fifo */ fifo_A_PE_2_4,
    /* fifo */ fifo_L_drain_PE_2_4,
    /* fifo */ fifo_U_drain_PE_2_4
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<2, 5>(
    /* fifo */ fifo_V_PE_2_5,
    /* fifo */ fifo_V_PE_3_5,
    /* fifo */ fifo_U_tmp_1_PE_2_5,
    /* fifo */ fifo_U_tmp_1_PE_2_6,
    /* fifo */ fifo_A_PE_2_5,
    /* fifo */ fifo_L_drain_PE_2_5,
    /* fifo */ fifo_U_drain_PE_2_5
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<2, 6>(
    /* fifo */ fifo_V_PE_2_6,
    /* fifo */ fifo_V_PE_3_6,
    /* fifo */ fifo_U_tmp_1_PE_2_6,
    /* fifo */ fifo_U_tmp_1_PE_2_7,
    /* fifo */ fifo_A_PE_2_6,
    /* fifo */ fifo_L_drain_PE_2_6,
    /* fifo */ fifo_U_drain_PE_2_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<2, 7>(
    /* fifo */ fifo_V_PE_2_7,
    /* fifo */ fifo_V_PE_3_7,
    /* fifo */ fifo_U_tmp_1_PE_2_7,
    /* fifo */ fifo_U_tmp_1_PE_2_8,
    /* fifo */ fifo_A_PE_2_7,
    /* fifo */ fifo_L_drain_PE_2_7,
    /* fifo */ fifo_U_drain_PE_2_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<2, 8>(
    /* fifo */ fifo_V_PE_2_8,
    /* fifo */ fifo_V_PE_3_8,
    /* fifo */ fifo_U_tmp_1_PE_2_8,
    /* fifo */ fifo_U_tmp_1_PE_2_9,
    /* fifo */ fifo_A_PE_2_8,
    /* fifo */ fifo_L_drain_PE_2_8,
    /* fifo */ fifo_U_drain_PE_2_8
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<2, 9>(
    /* fifo */ fifo_V_PE_2_9,
    /* fifo */ fifo_V_PE_3_9,
    /* fifo */ fifo_U_tmp_1_PE_2_9,
    /* fifo */ fifo_U_tmp_1_PE_2_10,
    /* fifo */ fifo_A_PE_2_9,
    /* fifo */ fifo_L_drain_PE_2_9,
    /* fifo */ fifo_U_drain_PE_2_9
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<2, 10>(
    /* fifo */ fifo_V_PE_2_10,
    /* fifo */ fifo_V_PE_3_10,
    /* fifo */ fifo_U_tmp_1_PE_2_10,
    /* fifo */ fifo_U_tmp_1_PE_2_11,
    /* fifo */ fifo_A_PE_2_10,
    /* fifo */ fifo_L_drain_PE_2_10,
    /* fifo */ fifo_U_drain_PE_2_10
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<2, 11>(
    /* fifo */ fifo_V_PE_2_11,
    /* fifo */ fifo_V_PE_3_11,
    /* fifo */ fifo_U_tmp_1_PE_2_11,
    /* fifo */ fifo_U_tmp_1_PE_2_12,
    /* fifo */ fifo_A_PE_2_11,
    /* fifo */ fifo_L_drain_PE_2_11,
    /* fifo */ fifo_U_drain_PE_2_11
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<3, 3>(
    /* fifo */ fifo_V_PE_3_3,
    /* fifo */ fifo_V_PE_4_3,
    /* fifo */ fifo_U_tmp_1_PE_3_3,
    /* fifo */ fifo_U_tmp_1_PE_3_4,
    /* fifo */ fifo_A_PE_3_3,
    /* fifo */ fifo_L_drain_PE_3_3,
    /* fifo */ fifo_U_drain_PE_3_3
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<3, 4>(
    /* fifo */ fifo_V_PE_3_4,
    /* fifo */ fifo_V_PE_4_4,
    /* fifo */ fifo_U_tmp_1_PE_3_4,
    /* fifo */ fifo_U_tmp_1_PE_3_5,
    /* fifo */ fifo_A_PE_3_4,
    /* fifo */ fifo_L_drain_PE_3_4,
    /* fifo */ fifo_U_drain_PE_3_4
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<3, 5>(
    /* fifo */ fifo_V_PE_3_5,
    /* fifo */ fifo_V_PE_4_5,
    /* fifo */ fifo_U_tmp_1_PE_3_5,
    /* fifo */ fifo_U_tmp_1_PE_3_6,
    /* fifo */ fifo_A_PE_3_5,
    /* fifo */ fifo_L_drain_PE_3_5,
    /* fifo */ fifo_U_drain_PE_3_5
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<3, 6>(
    /* fifo */ fifo_V_PE_3_6,
    /* fifo */ fifo_V_PE_4_6,
    /* fifo */ fifo_U_tmp_1_PE_3_6,
    /* fifo */ fifo_U_tmp_1_PE_3_7,
    /* fifo */ fifo_A_PE_3_6,
    /* fifo */ fifo_L_drain_PE_3_6,
    /* fifo */ fifo_U_drain_PE_3_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<3, 7>(
    /* fifo */ fifo_V_PE_3_7,
    /* fifo */ fifo_V_PE_4_7,
    /* fifo */ fifo_U_tmp_1_PE_3_7,
    /* fifo */ fifo_U_tmp_1_PE_3_8,
    /* fifo */ fifo_A_PE_3_7,
    /* fifo */ fifo_L_drain_PE_3_7,
    /* fifo */ fifo_U_drain_PE_3_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<3, 8>(
    /* fifo */ fifo_V_PE_3_8,
    /* fifo */ fifo_V_PE_4_8,
    /* fifo */ fifo_U_tmp_1_PE_3_8,
    /* fifo */ fifo_U_tmp_1_PE_3_9,
    /* fifo */ fifo_A_PE_3_8,
    /* fifo */ fifo_L_drain_PE_3_8,
    /* fifo */ fifo_U_drain_PE_3_8
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<3, 9>(
    /* fifo */ fifo_V_PE_3_9,
    /* fifo */ fifo_V_PE_4_9,
    /* fifo */ fifo_U_tmp_1_PE_3_9,
    /* fifo */ fifo_U_tmp_1_PE_3_10,
    /* fifo */ fifo_A_PE_3_9,
    /* fifo */ fifo_L_drain_PE_3_9,
    /* fifo */ fifo_U_drain_PE_3_9
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<3, 10>(
    /* fifo */ fifo_V_PE_3_10,
    /* fifo */ fifo_V_PE_4_10,
    /* fifo */ fifo_U_tmp_1_PE_3_10,
    /* fifo */ fifo_U_tmp_1_PE_3_11,
    /* fifo */ fifo_A_PE_3_10,
    /* fifo */ fifo_L_drain_PE_3_10,
    /* fifo */ fifo_U_drain_PE_3_10
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<3, 11>(
    /* fifo */ fifo_V_PE_3_11,
    /* fifo */ fifo_V_PE_4_11,
    /* fifo */ fifo_U_tmp_1_PE_3_11,
    /* fifo */ fifo_U_tmp_1_PE_3_12,
    /* fifo */ fifo_A_PE_3_11,
    /* fifo */ fifo_L_drain_PE_3_11,
    /* fifo */ fifo_U_drain_PE_3_11
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<4, 4>(
    /* fifo */ fifo_V_PE_4_4,
    /* fifo */ fifo_V_PE_5_4,
    /* fifo */ fifo_U_tmp_1_PE_4_4,
    /* fifo */ fifo_U_tmp_1_PE_4_5,
    /* fifo */ fifo_A_PE_4_4,
    /* fifo */ fifo_L_drain_PE_4_4,
    /* fifo */ fifo_U_drain_PE_4_4
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<4, 5>(
    /* fifo */ fifo_V_PE_4_5,
    /* fifo */ fifo_V_PE_5_5,
    /* fifo */ fifo_U_tmp_1_PE_4_5,
    /* fifo */ fifo_U_tmp_1_PE_4_6,
    /* fifo */ fifo_A_PE_4_5,
    /* fifo */ fifo_L_drain_PE_4_5,
    /* fifo */ fifo_U_drain_PE_4_5
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<4, 6>(
    /* fifo */ fifo_V_PE_4_6,
    /* fifo */ fifo_V_PE_5_6,
    /* fifo */ fifo_U_tmp_1_PE_4_6,
    /* fifo */ fifo_U_tmp_1_PE_4_7,
    /* fifo */ fifo_A_PE_4_6,
    /* fifo */ fifo_L_drain_PE_4_6,
    /* fifo */ fifo_U_drain_PE_4_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<4, 7>(
    /* fifo */ fifo_V_PE_4_7,
    /* fifo */ fifo_V_PE_5_7,
    /* fifo */ fifo_U_tmp_1_PE_4_7,
    /* fifo */ fifo_U_tmp_1_PE_4_8,
    /* fifo */ fifo_A_PE_4_7,
    /* fifo */ fifo_L_drain_PE_4_7,
    /* fifo */ fifo_U_drain_PE_4_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<4, 8>(
    /* fifo */ fifo_V_PE_4_8,
    /* fifo */ fifo_V_PE_5_8,
    /* fifo */ fifo_U_tmp_1_PE_4_8,
    /* fifo */ fifo_U_tmp_1_PE_4_9,
    /* fifo */ fifo_A_PE_4_8,
    /* fifo */ fifo_L_drain_PE_4_8,
    /* fifo */ fifo_U_drain_PE_4_8
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<4, 9>(
    /* fifo */ fifo_V_PE_4_9,
    /* fifo */ fifo_V_PE_5_9,
    /* fifo */ fifo_U_tmp_1_PE_4_9,
    /* fifo */ fifo_U_tmp_1_PE_4_10,
    /* fifo */ fifo_A_PE_4_9,
    /* fifo */ fifo_L_drain_PE_4_9,
    /* fifo */ fifo_U_drain_PE_4_9
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<4, 10>(
    /* fifo */ fifo_V_PE_4_10,
    /* fifo */ fifo_V_PE_5_10,
    /* fifo */ fifo_U_tmp_1_PE_4_10,
    /* fifo */ fifo_U_tmp_1_PE_4_11,
    /* fifo */ fifo_A_PE_4_10,
    /* fifo */ fifo_L_drain_PE_4_10,
    /* fifo */ fifo_U_drain_PE_4_10
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<4, 11>(
    /* fifo */ fifo_V_PE_4_11,
    /* fifo */ fifo_V_PE_5_11,
    /* fifo */ fifo_U_tmp_1_PE_4_11,
    /* fifo */ fifo_U_tmp_1_PE_4_12,
    /* fifo */ fifo_A_PE_4_11,
    /* fifo */ fifo_L_drain_PE_4_11,
    /* fifo */ fifo_U_drain_PE_4_11
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<5, 5>(
    /* fifo */ fifo_V_PE_5_5,
    /* fifo */ fifo_V_PE_6_5,
    /* fifo */ fifo_U_tmp_1_PE_5_5,
    /* fifo */ fifo_U_tmp_1_PE_5_6,
    /* fifo */ fifo_A_PE_5_5,
    /* fifo */ fifo_L_drain_PE_5_5,
    /* fifo */ fifo_U_drain_PE_5_5
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<5, 6>(
    /* fifo */ fifo_V_PE_5_6,
    /* fifo */ fifo_V_PE_6_6,
    /* fifo */ fifo_U_tmp_1_PE_5_6,
    /* fifo */ fifo_U_tmp_1_PE_5_7,
    /* fifo */ fifo_A_PE_5_6,
    /* fifo */ fifo_L_drain_PE_5_6,
    /* fifo */ fifo_U_drain_PE_5_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<5, 7>(
    /* fifo */ fifo_V_PE_5_7,
    /* fifo */ fifo_V_PE_6_7,
    /* fifo */ fifo_U_tmp_1_PE_5_7,
    /* fifo */ fifo_U_tmp_1_PE_5_8,
    /* fifo */ fifo_A_PE_5_7,
    /* fifo */ fifo_L_drain_PE_5_7,
    /* fifo */ fifo_U_drain_PE_5_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<5, 8>(
    /* fifo */ fifo_V_PE_5_8,
    /* fifo */ fifo_V_PE_6_8,
    /* fifo */ fifo_U_tmp_1_PE_5_8,
    /* fifo */ fifo_U_tmp_1_PE_5_9,
    /* fifo */ fifo_A_PE_5_8,
    /* fifo */ fifo_L_drain_PE_5_8,
    /* fifo */ fifo_U_drain_PE_5_8
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<5, 9>(
    /* fifo */ fifo_V_PE_5_9,
    /* fifo */ fifo_V_PE_6_9,
    /* fifo */ fifo_U_tmp_1_PE_5_9,
    /* fifo */ fifo_U_tmp_1_PE_5_10,
    /* fifo */ fifo_A_PE_5_9,
    /* fifo */ fifo_L_drain_PE_5_9,
    /* fifo */ fifo_U_drain_PE_5_9
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<5, 10>(
    /* fifo */ fifo_V_PE_5_10,
    /* fifo */ fifo_V_PE_6_10,
    /* fifo */ fifo_U_tmp_1_PE_5_10,
    /* fifo */ fifo_U_tmp_1_PE_5_11,
    /* fifo */ fifo_A_PE_5_10,
    /* fifo */ fifo_L_drain_PE_5_10,
    /* fifo */ fifo_U_drain_PE_5_10
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<5, 11>(
    /* fifo */ fifo_V_PE_5_11,
    /* fifo */ fifo_V_PE_6_11,
    /* fifo */ fifo_U_tmp_1_PE_5_11,
    /* fifo */ fifo_U_tmp_1_PE_5_12,
    /* fifo */ fifo_A_PE_5_11,
    /* fifo */ fifo_L_drain_PE_5_11,
    /* fifo */ fifo_U_drain_PE_5_11
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<6, 6>(
    /* fifo */ fifo_V_PE_6_6,
    /* fifo */ fifo_V_PE_7_6,
    /* fifo */ fifo_U_tmp_1_PE_6_6,
    /* fifo */ fifo_U_tmp_1_PE_6_7,
    /* fifo */ fifo_A_PE_6_6,
    /* fifo */ fifo_L_drain_PE_6_6,
    /* fifo */ fifo_U_drain_PE_6_6
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<6, 7>(
    /* fifo */ fifo_V_PE_6_7,
    /* fifo */ fifo_V_PE_7_7,
    /* fifo */ fifo_U_tmp_1_PE_6_7,
    /* fifo */ fifo_U_tmp_1_PE_6_8,
    /* fifo */ fifo_A_PE_6_7,
    /* fifo */ fifo_L_drain_PE_6_7,
    /* fifo */ fifo_U_drain_PE_6_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<6, 8>(
    /* fifo */ fifo_V_PE_6_8,
    /* fifo */ fifo_V_PE_7_8,
    /* fifo */ fifo_U_tmp_1_PE_6_8,
    /* fifo */ fifo_U_tmp_1_PE_6_9,
    /* fifo */ fifo_A_PE_6_8,
    /* fifo */ fifo_L_drain_PE_6_8,
    /* fifo */ fifo_U_drain_PE_6_8
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<6, 9>(
    /* fifo */ fifo_V_PE_6_9,
    /* fifo */ fifo_V_PE_7_9,
    /* fifo */ fifo_U_tmp_1_PE_6_9,
    /* fifo */ fifo_U_tmp_1_PE_6_10,
    /* fifo */ fifo_A_PE_6_9,
    /* fifo */ fifo_L_drain_PE_6_9,
    /* fifo */ fifo_U_drain_PE_6_9
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<6, 10>(
    /* fifo */ fifo_V_PE_6_10,
    /* fifo */ fifo_V_PE_7_10,
    /* fifo */ fifo_U_tmp_1_PE_6_10,
    /* fifo */ fifo_U_tmp_1_PE_6_11,
    /* fifo */ fifo_A_PE_6_10,
    /* fifo */ fifo_L_drain_PE_6_10,
    /* fifo */ fifo_U_drain_PE_6_10
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<6, 11>(
    /* fifo */ fifo_V_PE_6_11,
    /* fifo */ fifo_V_PE_7_11,
    /* fifo */ fifo_U_tmp_1_PE_6_11,
    /* fifo */ fifo_U_tmp_1_PE_6_12,
    /* fifo */ fifo_A_PE_6_11,
    /* fifo */ fifo_L_drain_PE_6_11,
    /* fifo */ fifo_U_drain_PE_6_11
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<7, 7>(
    /* fifo */ fifo_V_PE_7_7,
    /* fifo */ fifo_V_PE_8_7,
    /* fifo */ fifo_U_tmp_1_PE_7_7,
    /* fifo */ fifo_U_tmp_1_PE_7_8,
    /* fifo */ fifo_A_PE_7_7,
    /* fifo */ fifo_L_drain_PE_7_7,
    /* fifo */ fifo_U_drain_PE_7_7
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<7, 8>(
    /* fifo */ fifo_V_PE_7_8,
    /* fifo */ fifo_V_PE_8_8,
    /* fifo */ fifo_U_tmp_1_PE_7_8,
    /* fifo */ fifo_U_tmp_1_PE_7_9,
    /* fifo */ fifo_A_PE_7_8,
    /* fifo */ fifo_L_drain_PE_7_8,
    /* fifo */ fifo_U_drain_PE_7_8
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<7, 9>(
    /* fifo */ fifo_V_PE_7_9,
    /* fifo */ fifo_V_PE_8_9,
    /* fifo */ fifo_U_tmp_1_PE_7_9,
    /* fifo */ fifo_U_tmp_1_PE_7_10,
    /* fifo */ fifo_A_PE_7_9,
    /* fifo */ fifo_L_drain_PE_7_9,
    /* fifo */ fifo_U_drain_PE_7_9
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<7, 10>(
    /* fifo */ fifo_V_PE_7_10,
    /* fifo */ fifo_V_PE_8_10,
    /* fifo */ fifo_U_tmp_1_PE_7_10,
    /* fifo */ fifo_U_tmp_1_PE_7_11,
    /* fifo */ fifo_A_PE_7_10,
    /* fifo */ fifo_L_drain_PE_7_10,
    /* fifo */ fifo_U_drain_PE_7_10
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<7, 11>(
    /* fifo */ fifo_V_PE_7_11,
    /* fifo */ fifo_V_PE_8_11,
    /* fifo */ fifo_U_tmp_1_PE_7_11,
    /* fifo */ fifo_U_tmp_1_PE_7_12,
    /* fifo */ fifo_A_PE_7_11,
    /* fifo */ fifo_L_drain_PE_7_11,
    /* fifo */ fifo_U_drain_PE_7_11
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<8, 8>(
    /* fifo */ fifo_V_PE_8_8,
    /* fifo */ fifo_V_PE_9_8,
    /* fifo */ fifo_U_tmp_1_PE_8_8,
    /* fifo */ fifo_U_tmp_1_PE_8_9,
    /* fifo */ fifo_A_PE_8_8,
    /* fifo */ fifo_L_drain_PE_8_8,
    /* fifo */ fifo_U_drain_PE_8_8
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<8, 9>(
    /* fifo */ fifo_V_PE_8_9,
    /* fifo */ fifo_V_PE_9_9,
    /* fifo */ fifo_U_tmp_1_PE_8_9,
    /* fifo */ fifo_U_tmp_1_PE_8_10,
    /* fifo */ fifo_A_PE_8_9,
    /* fifo */ fifo_L_drain_PE_8_9,
    /* fifo */ fifo_U_drain_PE_8_9
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<8, 10>(
    /* fifo */ fifo_V_PE_8_10,
    /* fifo */ fifo_V_PE_9_10,
    /* fifo */ fifo_U_tmp_1_PE_8_10,
    /* fifo */ fifo_U_tmp_1_PE_8_11,
    /* fifo */ fifo_A_PE_8_10,
    /* fifo */ fifo_L_drain_PE_8_10,
    /* fifo */ fifo_U_drain_PE_8_10
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<8, 11>(
    /* fifo */ fifo_V_PE_8_11,
    /* fifo */ fifo_V_PE_9_11,
    /* fifo */ fifo_U_tmp_1_PE_8_11,
    /* fifo */ fifo_U_tmp_1_PE_8_12,
    /* fifo */ fifo_A_PE_8_11,
    /* fifo */ fifo_L_drain_PE_8_11,
    /* fifo */ fifo_U_drain_PE_8_11
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<9, 9>(
    /* fifo */ fifo_V_PE_9_9,
    /* fifo */ fifo_V_PE_10_9,
    /* fifo */ fifo_U_tmp_1_PE_9_9,
    /* fifo */ fifo_U_tmp_1_PE_9_10,
    /* fifo */ fifo_A_PE_9_9,
    /* fifo */ fifo_L_drain_PE_9_9,
    /* fifo */ fifo_U_drain_PE_9_9
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<9, 10>(
    /* fifo */ fifo_V_PE_9_10,
    /* fifo */ fifo_V_PE_10_10,
    /* fifo */ fifo_U_tmp_1_PE_9_10,
    /* fifo */ fifo_U_tmp_1_PE_9_11,
    /* fifo */ fifo_A_PE_9_10,
    /* fifo */ fifo_L_drain_PE_9_10,
    /* fifo */ fifo_U_drain_PE_9_10
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<9, 11>(
    /* fifo */ fifo_V_PE_9_11,
    /* fifo */ fifo_V_PE_10_11,
    /* fifo */ fifo_U_tmp_1_PE_9_11,
    /* fifo */ fifo_U_tmp_1_PE_9_12,
    /* fifo */ fifo_A_PE_9_11,
    /* fifo */ fifo_L_drain_PE_9_11,
    /* fifo */ fifo_U_drain_PE_9_11
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<10, 10>(
    /* fifo */ fifo_V_PE_10_10,
    /* fifo */ fifo_V_PE_11_10,
    /* fifo */ fifo_U_tmp_1_PE_10_10,
    /* fifo */ fifo_U_tmp_1_PE_10_11,
    /* fifo */ fifo_A_PE_10_10,
    /* fifo */ fifo_L_drain_PE_10_10,
    /* fifo */ fifo_U_drain_PE_10_10
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<10, 11>(
    /* fifo */ fifo_V_PE_10_11,
    /* fifo */ fifo_V_PE_11_11,
    /* fifo */ fifo_U_tmp_1_PE_10_11,
    /* fifo */ fifo_U_tmp_1_PE_10_12,
    /* fifo */ fifo_A_PE_10_11,
    /* fifo */ fifo_L_drain_PE_10_11,
    /* fifo */ fifo_U_drain_PE_10_11
  );
  /* Module Call */

  /* Module Call */
  PE_wrapper<11, 11>(
    /* fifo */ fifo_V_PE_11_11,
    /* fifo */ fifo_V_PE_12_11,
    /* fifo */ fifo_U_tmp_1_PE_11_11,
    /* fifo */ fifo_U_tmp_1_PE_11_12,
    /* fifo */ fifo_A_PE_11_11,
    /* fifo */ fifo_L_drain_PE_11_11,
    /* fifo */ fifo_U_drain_PE_11_11
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_boundary_wrapper<1, 0>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_1_0,
    /* fifo */ fifo_L_drain_PE_0_1
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_boundary_wrapper<2, 1>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_2_1,
    /* fifo */ fifo_L_drain_PE_1_2
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<2, 0>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_2_1,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_2_0,
    /* fifo */ fifo_L_drain_PE_0_2
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_boundary_wrapper<3, 2>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_3_2,
    /* fifo */ fifo_L_drain_PE_2_3
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<3, 1>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_3_2,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_3_1,
    /* fifo */ fifo_L_drain_PE_1_3
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<3, 0>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_3_1,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_3_0,
    /* fifo */ fifo_L_drain_PE_0_3
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_boundary_wrapper<4, 3>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_4_3,
    /* fifo */ fifo_L_drain_PE_3_4
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<4, 2>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_4_3,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_4_2,
    /* fifo */ fifo_L_drain_PE_2_4
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<4, 1>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_4_2,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_4_1,
    /* fifo */ fifo_L_drain_PE_1_4
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<4, 0>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_4_1,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_4_0,
    /* fifo */ fifo_L_drain_PE_0_4
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_boundary_wrapper<5, 4>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_5_4,
    /* fifo */ fifo_L_drain_PE_4_5
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<5, 3>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_5_4,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_5_3,
    /* fifo */ fifo_L_drain_PE_3_5
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<5, 2>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_5_3,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_5_2,
    /* fifo */ fifo_L_drain_PE_2_5
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<5, 1>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_5_2,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_5_1,
    /* fifo */ fifo_L_drain_PE_1_5
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<5, 0>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_5_1,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_5_0,
    /* fifo */ fifo_L_drain_PE_0_5
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_boundary_wrapper<6, 5>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_6_5,
    /* fifo */ fifo_L_drain_PE_5_6
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<6, 4>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_6_5,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_6_4,
    /* fifo */ fifo_L_drain_PE_4_6
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<6, 3>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_6_4,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_6_3,
    /* fifo */ fifo_L_drain_PE_3_6
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<6, 2>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_6_3,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_6_2,
    /* fifo */ fifo_L_drain_PE_2_6
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<6, 1>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_6_2,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_6_1,
    /* fifo */ fifo_L_drain_PE_1_6
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<6, 0>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_6_1,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_6_0,
    /* fifo */ fifo_L_drain_PE_0_6
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_boundary_wrapper<7, 6>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7_6,
    /* fifo */ fifo_L_drain_PE_6_7
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<7, 5>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7_6,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7_5,
    /* fifo */ fifo_L_drain_PE_5_7
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<7, 4>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7_5,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7_4,
    /* fifo */ fifo_L_drain_PE_4_7
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<7, 3>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7_4,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7_3,
    /* fifo */ fifo_L_drain_PE_3_7
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<7, 2>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7_3,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7_2,
    /* fifo */ fifo_L_drain_PE_2_7
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<7, 1>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7_2,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7_1,
    /* fifo */ fifo_L_drain_PE_1_7
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<7, 0>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7_1,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7_0,
    /* fifo */ fifo_L_drain_PE_0_7
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_boundary_wrapper<8, 7>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_7,
    /* fifo */ fifo_L_drain_PE_7_8
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<8, 6>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_7,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_6,
    /* fifo */ fifo_L_drain_PE_6_8
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<8, 5>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_6,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_5,
    /* fifo */ fifo_L_drain_PE_5_8
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<8, 4>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_5,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_4,
    /* fifo */ fifo_L_drain_PE_4_8
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<8, 3>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_4,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_3,
    /* fifo */ fifo_L_drain_PE_3_8
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<8, 2>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_3,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_2,
    /* fifo */ fifo_L_drain_PE_2_8
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<8, 1>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_2,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_1,
    /* fifo */ fifo_L_drain_PE_1_8
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<8, 0>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_1,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_0,
    /* fifo */ fifo_L_drain_PE_0_8
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_boundary_wrapper<9, 8>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_8,
    /* fifo */ fifo_L_drain_PE_8_9
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<9, 7>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_8,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_7,
    /* fifo */ fifo_L_drain_PE_7_9
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<9, 6>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_7,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_6,
    /* fifo */ fifo_L_drain_PE_6_9
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<9, 5>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_6,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_5,
    /* fifo */ fifo_L_drain_PE_5_9
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<9, 4>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_5,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_4,
    /* fifo */ fifo_L_drain_PE_4_9
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<9, 3>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_4,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_3,
    /* fifo */ fifo_L_drain_PE_3_9
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<9, 2>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_3,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_2,
    /* fifo */ fifo_L_drain_PE_2_9
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<9, 1>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_2,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_1,
    /* fifo */ fifo_L_drain_PE_1_9
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<9, 0>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_1,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_0,
    /* fifo */ fifo_L_drain_PE_0_9
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_boundary_wrapper<10, 9>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_9,
    /* fifo */ fifo_L_drain_PE_9_10
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<10, 8>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_9,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_8,
    /* fifo */ fifo_L_drain_PE_8_10
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<10, 7>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_8,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_7,
    /* fifo */ fifo_L_drain_PE_7_10
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<10, 6>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_7,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_6,
    /* fifo */ fifo_L_drain_PE_6_10
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<10, 5>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_6,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_5,
    /* fifo */ fifo_L_drain_PE_5_10
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<10, 4>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_5,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_4,
    /* fifo */ fifo_L_drain_PE_4_10
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<10, 3>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_4,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_3,
    /* fifo */ fifo_L_drain_PE_3_10
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<10, 2>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_3,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_2,
    /* fifo */ fifo_L_drain_PE_2_10
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<10, 1>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_2,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_1,
    /* fifo */ fifo_L_drain_PE_1_10
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<10, 0>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_1,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_0,
    /* fifo */ fifo_L_drain_PE_0_10
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_boundary_wrapper<11, 10>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_10,
    /* fifo */ fifo_L_drain_PE_10_11
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<11, 9>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_10,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_9,
    /* fifo */ fifo_L_drain_PE_9_11
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<11, 8>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_9,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_8,
    /* fifo */ fifo_L_drain_PE_8_11
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<11, 7>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_8,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_7,
    /* fifo */ fifo_L_drain_PE_7_11
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<11, 6>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_7,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_6,
    /* fifo */ fifo_L_drain_PE_6_11
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<11, 5>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_6,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_5,
    /* fifo */ fifo_L_drain_PE_5_11
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<11, 4>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_5,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_4,
    /* fifo */ fifo_L_drain_PE_4_11
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<11, 3>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_4,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_3,
    /* fifo */ fifo_L_drain_PE_3_11
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<11, 2>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_3,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_2,
    /* fifo */ fifo_L_drain_PE_2_11
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<11, 1>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_2,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_1,
    /* fifo */ fifo_L_drain_PE_1_11
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L1_out_wrapper<11, 0>(
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_1,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_0,
    /* fifo */ fifo_L_drain_PE_0_11
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L2_out_boundary<11>(
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_11,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_11_0
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L2_out<10>(
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_11,
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_10,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_10_0
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L2_out<9>(
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_10,
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_9,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_9_0
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L2_out<8>(
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_9,
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_8,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_8_0
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L2_out<7>(
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_8,
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_7,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_7_0
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L2_out<6>(
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_7,
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_6,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_6_0
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L2_out<5>(
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_6,
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_5,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_5_0
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L2_out<4>(
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_5,
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_4,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_4_0
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L2_out<3>(
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_4,
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_3,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_3_0
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L2_out<2>(
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_3,
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_2,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_2_0
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L2_out<1>(
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_2,
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_1,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_1_0
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L2_out<0>(
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_1,
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_0,
    /* fifo */ fifo_L_drain_L_drain_IO_L1_out_0_0
  );
  /* Module Call */

  /* Module Call */
  L_drain_IO_L3_out(
    /* array */ L,
    /* fifo */ fifo_L_drain_L_drain_IO_L2_out_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_boundary_wrapper<0, 0>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_0_0,
    /* fifo */ fifo_U_drain_PE_0_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_boundary_wrapper<1, 1>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_1_1,
    /* fifo */ fifo_U_drain_PE_1_1
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<1, 0>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_1_1,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_1_0,
    /* fifo */ fifo_U_drain_PE_0_1
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_boundary_wrapper<2, 2>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_2_2,
    /* fifo */ fifo_U_drain_PE_2_2
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<2, 1>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_2_2,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_2_1,
    /* fifo */ fifo_U_drain_PE_1_2
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<2, 0>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_2_1,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_2_0,
    /* fifo */ fifo_U_drain_PE_0_2
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_boundary_wrapper<3, 3>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_3_3,
    /* fifo */ fifo_U_drain_PE_3_3
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<3, 2>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_3_3,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_3_2,
    /* fifo */ fifo_U_drain_PE_2_3
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<3, 1>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_3_2,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_3_1,
    /* fifo */ fifo_U_drain_PE_1_3
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<3, 0>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_3_1,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_3_0,
    /* fifo */ fifo_U_drain_PE_0_3
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_boundary_wrapper<4, 4>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_4_4,
    /* fifo */ fifo_U_drain_PE_4_4
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<4, 3>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_4_4,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_4_3,
    /* fifo */ fifo_U_drain_PE_3_4
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<4, 2>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_4_3,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_4_2,
    /* fifo */ fifo_U_drain_PE_2_4
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<4, 1>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_4_2,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_4_1,
    /* fifo */ fifo_U_drain_PE_1_4
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<4, 0>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_4_1,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_4_0,
    /* fifo */ fifo_U_drain_PE_0_4
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_boundary_wrapper<5, 5>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_5_5,
    /* fifo */ fifo_U_drain_PE_5_5
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<5, 4>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_5_5,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_5_4,
    /* fifo */ fifo_U_drain_PE_4_5
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<5, 3>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_5_4,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_5_3,
    /* fifo */ fifo_U_drain_PE_3_5
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<5, 2>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_5_3,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_5_2,
    /* fifo */ fifo_U_drain_PE_2_5
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<5, 1>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_5_2,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_5_1,
    /* fifo */ fifo_U_drain_PE_1_5
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<5, 0>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_5_1,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_5_0,
    /* fifo */ fifo_U_drain_PE_0_5
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_boundary_wrapper<6, 6>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6_6,
    /* fifo */ fifo_U_drain_PE_6_6
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<6, 5>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6_6,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6_5,
    /* fifo */ fifo_U_drain_PE_5_6
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<6, 4>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6_5,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6_4,
    /* fifo */ fifo_U_drain_PE_4_6
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<6, 3>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6_4,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6_3,
    /* fifo */ fifo_U_drain_PE_3_6
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<6, 2>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6_3,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6_2,
    /* fifo */ fifo_U_drain_PE_2_6
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<6, 1>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6_2,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6_1,
    /* fifo */ fifo_U_drain_PE_1_6
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<6, 0>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6_1,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6_0,
    /* fifo */ fifo_U_drain_PE_0_6
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_boundary_wrapper<7, 7>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_7,
    /* fifo */ fifo_U_drain_PE_7_7
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<7, 6>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_7,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_6,
    /* fifo */ fifo_U_drain_PE_6_7
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<7, 5>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_6,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_5,
    /* fifo */ fifo_U_drain_PE_5_7
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<7, 4>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_5,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_4,
    /* fifo */ fifo_U_drain_PE_4_7
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<7, 3>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_4,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_3,
    /* fifo */ fifo_U_drain_PE_3_7
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<7, 2>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_3,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_2,
    /* fifo */ fifo_U_drain_PE_2_7
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<7, 1>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_2,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_1,
    /* fifo */ fifo_U_drain_PE_1_7
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<7, 0>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_1,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_0,
    /* fifo */ fifo_U_drain_PE_0_7
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_boundary_wrapper<8, 8>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_8,
    /* fifo */ fifo_U_drain_PE_8_8
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<8, 7>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_8,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_7,
    /* fifo */ fifo_U_drain_PE_7_8
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<8, 6>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_7,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_6,
    /* fifo */ fifo_U_drain_PE_6_8
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<8, 5>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_6,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_5,
    /* fifo */ fifo_U_drain_PE_5_8
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<8, 4>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_5,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_4,
    /* fifo */ fifo_U_drain_PE_4_8
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<8, 3>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_4,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_3,
    /* fifo */ fifo_U_drain_PE_3_8
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<8, 2>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_3,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_2,
    /* fifo */ fifo_U_drain_PE_2_8
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<8, 1>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_2,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_1,
    /* fifo */ fifo_U_drain_PE_1_8
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<8, 0>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_1,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_0,
    /* fifo */ fifo_U_drain_PE_0_8
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_boundary_wrapper<9, 9>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_9,
    /* fifo */ fifo_U_drain_PE_9_9
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<9, 8>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_9,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_8,
    /* fifo */ fifo_U_drain_PE_8_9
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<9, 7>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_8,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_7,
    /* fifo */ fifo_U_drain_PE_7_9
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<9, 6>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_7,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_6,
    /* fifo */ fifo_U_drain_PE_6_9
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<9, 5>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_6,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_5,
    /* fifo */ fifo_U_drain_PE_5_9
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<9, 4>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_5,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_4,
    /* fifo */ fifo_U_drain_PE_4_9
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<9, 3>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_4,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_3,
    /* fifo */ fifo_U_drain_PE_3_9
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<9, 2>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_3,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_2,
    /* fifo */ fifo_U_drain_PE_2_9
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<9, 1>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_2,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_1,
    /* fifo */ fifo_U_drain_PE_1_9
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<9, 0>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_1,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_0,
    /* fifo */ fifo_U_drain_PE_0_9
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_boundary_wrapper<10, 10>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_10,
    /* fifo */ fifo_U_drain_PE_10_10
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<10, 9>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_10,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_9,
    /* fifo */ fifo_U_drain_PE_9_10
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<10, 8>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_9,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_8,
    /* fifo */ fifo_U_drain_PE_8_10
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<10, 7>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_8,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_7,
    /* fifo */ fifo_U_drain_PE_7_10
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<10, 6>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_7,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_6,
    /* fifo */ fifo_U_drain_PE_6_10
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<10, 5>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_6,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_5,
    /* fifo */ fifo_U_drain_PE_5_10
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<10, 4>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_5,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_4,
    /* fifo */ fifo_U_drain_PE_4_10
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<10, 3>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_4,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_3,
    /* fifo */ fifo_U_drain_PE_3_10
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<10, 2>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_3,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_2,
    /* fifo */ fifo_U_drain_PE_2_10
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<10, 1>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_2,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_1,
    /* fifo */ fifo_U_drain_PE_1_10
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<10, 0>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_1,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_0,
    /* fifo */ fifo_U_drain_PE_0_10
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_boundary_wrapper<11, 11>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_11,
    /* fifo */ fifo_U_drain_PE_11_11
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<11, 10>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_11,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_10,
    /* fifo */ fifo_U_drain_PE_10_11
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<11, 9>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_10,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_9,
    /* fifo */ fifo_U_drain_PE_9_11
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<11, 8>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_9,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_8,
    /* fifo */ fifo_U_drain_PE_8_11
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<11, 7>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_8,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_7,
    /* fifo */ fifo_U_drain_PE_7_11
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<11, 6>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_7,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_6,
    /* fifo */ fifo_U_drain_PE_6_11
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<11, 5>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_6,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_5,
    /* fifo */ fifo_U_drain_PE_5_11
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<11, 4>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_5,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_4,
    /* fifo */ fifo_U_drain_PE_4_11
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<11, 3>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_4,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_3,
    /* fifo */ fifo_U_drain_PE_3_11
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<11, 2>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_3,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_2,
    /* fifo */ fifo_U_drain_PE_2_11
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<11, 1>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_2,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_1,
    /* fifo */ fifo_U_drain_PE_1_11
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L1_out_wrapper<11, 0>(
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_1,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_0,
    /* fifo */ fifo_U_drain_PE_0_11
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L2_out_boundary<11>(
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_11,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_11_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L2_out<10>(
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_11,
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_10,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_10_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L2_out<9>(
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_10,
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_9,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_9_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L2_out<8>(
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_9,
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_8,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_8_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L2_out<7>(
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_8,
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_7,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_7_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L2_out<6>(
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_7,
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_6,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_6_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L2_out<5>(
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_6,
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_5,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_5_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L2_out<4>(
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_5,
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_4,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_4_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L2_out<3>(
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_4,
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_3,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_3_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L2_out<2>(
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_3,
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_2,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_2_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L2_out<1>(
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_2,
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_1,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_1_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L2_out<0>(
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_1,
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_0,
    /* fifo */ fifo_U_drain_U_drain_IO_L1_out_0_0
  );
  /* Module Call */

  /* Module Call */
  U_drain_IO_L3_out(
    /* array */ U,
    /* fifo */ fifo_U_drain_U_drain_IO_L2_out_0
  );
  /* Module Call */

}
}
