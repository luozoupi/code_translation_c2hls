// this file can be generated from the following SODA DSL
/*
kernel: heat3d
burst width: 512
iterate: 5
unroll factor: 4
input dram 0 float: in(32, 32, *)
output dram 1 float: out(0, 0, 0) = (.125f * in(1, 0, 0)) + (.125f * in(-1, 0, 0)) + (.125f * in(0, 1, 0)) + (.125f * in(0, -1, 0)) + (.125f * in(0, 0, 1)) + (.125f * in(0, 0, -1)) + (.25f * in(0, 0, 0))
border: ignore
cluster: none
*/

// stencil window size: (11, 11, 11)
// stencil distace: 164485
// data layout is documented at
// https://github.com/Blaok/soda/blob/master/docs/data-layout.md

#include <cfloat>
#include <cmath>
#include <cstdbool>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <tuple>

#include <ap_int.h>
#include <hls_stream.h>

#ifdef __SYNTHESIS__
#warning this file should be used for simulation only
#warning synthesis result may be sub-optimal
#endif  // __SYNTHESIS__

template<typename To, typename From>
To Reinterpret(From val) {
#pragma HLS inline
  return reinterpret_cast<To&>(val);
}

template<typename T>
struct Data {
  T data;
  bool ctrl;
};

template<typename T>
bool ReadData(T& data, hls::stream<Data<T>>& from) {
#pragma HLS inline
  const auto tmp = from.read();
  data = tmp.data;
  return tmp.ctrl;
}

template<typename T>
void WriteData(hls::stream<Data<T>>& to, const T& data, bool ctrl) {
#pragma HLS inline
  Data<T> tmp;
  tmp.data = data;
  tmp.ctrl = ctrl;
  to.write(tmp);
}

template <typename T>
void BurstRead(hls::stream<Data<T>>& to, T* from, uint64_t data_num) {
load:
  for (uint64_t i = 0; i < data_num;) {
#pragma HLS pipeline II = 1 style = stp  // inv3d: was auto
    const uint64_t next_i = i + 1;
    WriteData(to, from[i], next_i < data_num);
    i = next_i;
  }
}

template <typename T>
void BurstWrite(T* to, hls::stream<Data<T>>& from, uint64_t data_num) {
store:
  for (uint64_t i = 0; i < data_num; ++i) {
#pragma HLS pipeline II = 1 style = stp  // inv3d: was auto
    T buf;
    ReadData(buf, from);
    to[i] = buf;
  }
}

void Module0Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /*output*/ hls::stream<Data<float>>& fifo_st_3, 
  /* input*/ hls::stream<Data<ap_uint<512>>>& dram_in_bank_0_fifo)
{
#pragma HLS aggregate bit variable = dram_in_bank_0_fifo
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_0:
  bool enable = true;  // patch: fixed trip count 141353 (measured at x 128 128 128)
  for (uint64_t t_patch = 0; t_patch < 141353ULL; ++t_patch)
  {
#pragma HLS pipeline II = 4 style = stp  // inv3d: was auto
    if (true)  // patch: blocking reads (was: !dram_in_bank_0_fifo.empty())
    {
      ap_uint<512> dram_in_bank_0_buf;
      const bool dram_in_bank_0_buf_enable = ReadData(dram_in_bank_0_buf, dram_in_bank_0_fifo);
      const bool enabled = dram_in_bank_0_buf_enable;
      enable = enabled;
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(127, 96))), true);
      WriteData(fifo_st_1, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(95, 64))), true);
      WriteData(fifo_st_2, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(63, 32))), true);
      WriteData(fifo_st_3, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(31, 0))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(255, 224))), true);
      WriteData(fifo_st_1, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(223, 192))), true);
      WriteData(fifo_st_2, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(191, 160))), true);
      WriteData(fifo_st_3, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(159, 128))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(383, 352))), true);
      WriteData(fifo_st_1, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(351, 320))), true);
      WriteData(fifo_st_2, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(319, 288))), true);
      WriteData(fifo_st_3, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(287, 256))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(511, 480))), enabled);
      WriteData(fifo_st_1, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(479, 448))), enabled);
      WriteData(fifo_st_2, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(447, 416))), enabled);
      WriteData(fifo_st_3, Reinterpret<float>(static_cast<ap_uint<32>>(dram_in_bank_0_buf(415, 384))), enabled);
    } // if not empty
  } // for module_0
} // Module0Func

void Module1Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_1:
  bool enable = true;  // patch: fixed trip count 565412 (measured at x 128 128 128)
  for (uint64_t t_patch = 0; t_patch < 565412ULL; ++t_patch)
  {
#pragma HLS pipeline II = 1 style = stp  // inv3d: was auto
    if (true)  // patch: blocking reads (was: !fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, float(fifo_ref_0), enabled);
      WriteData(fifo_st_1, float(fifo_ref_0), enabled);
    } // if not empty
  } // for module_1
} // Module1Func

void Module2Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<12> ptr_delay_4064 = 0;
  float fifo_ref_0_delayed_4064_buf[4064];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_4064_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_2:
  bool enable = true;  // patch: fixed trip count 565412 (measured at x 128 128 128)
  for (uint64_t t_patch = 0; t_patch < 565412ULL; ++t_patch)
  {
#pragma HLS pipeline II = 1 style = stp  // inv3d: was auto
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_4064_buf distance = 4064
    if (true)  // patch: blocking reads (was: !fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_4064 = fifo_ref_0_delayed_4064_buf[ptr_delay_4064];;
      const float let_0 = fifo_ref_0_delayed_4064;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      fifo_ref_0_delayed_4064_buf[ptr_delay_4064] = fifo_ref_0;
      ptr_delay_4064 < 4063 ? (++ptr_delay_4064) : (ptr_delay_4064 = 0);
    } // if not empty
  } // for module_2
} // Module2Func

void Module3Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<5> ptr_delay_32 = 0;
  float fifo_ref_0_delayed_32_buf[32];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_32_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_3:
  bool enable = true;  // patch: fixed trip count 565412 (measured at x 128 128 128)
  for (uint64_t t_patch = 0; t_patch < 565412ULL; ++t_patch)
  {
#pragma HLS pipeline II = 1 style = stp  // inv3d: was auto
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_32_buf distance = 32
    if (true)  // patch: blocking reads (was: !fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_32 = fifo_ref_0_delayed_32_buf[ptr_delay_32];;
      const float let_0 = fifo_ref_0_delayed_32;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      fifo_ref_0_delayed_32_buf[ptr_delay_32] = fifo_ref_0;
      ptr_delay_32 < 31 ? (++ptr_delay_32) : (ptr_delay_32 = 0);
    } // if not empty
  } // for module_3
} // Module3Func

void Module4Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /*output*/ hls::stream<Data<float>>& fifo_st_3, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<5> ptr_delay_32 = 0;
  float fifo_ref_0_delayed_32_buf[32];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_32_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_4:
  bool enable = true;  // patch: fixed trip count 565412 (measured at x 128 128 128)
  for (uint64_t t_patch = 0; t_patch < 565412ULL; ++t_patch)
  {
#pragma HLS pipeline II = 1 style = stp  // inv3d: was auto
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_32_buf distance = 32
    if (true)  // patch: blocking reads (was: !fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_32 = fifo_ref_0_delayed_32_buf[ptr_delay_32];;
      const float let_0 = fifo_ref_0_delayed_32;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      WriteData(fifo_st_3, float(let_0), enabled);
      fifo_ref_0_delayed_32_buf[ptr_delay_32] = fifo_ref_0;
      ptr_delay_32 < 31 ? (++ptr_delay_32) : (ptr_delay_32 = 0);
    } // if not empty
  } // for module_4
} // Module4Func

void Module5Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<5> ptr_delay_31 = 0;
  float fifo_ref_0_delayed_31_buf[31];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_31_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_5:
  bool enable = true;  // patch: fixed trip count 565412 (measured at x 128 128 128)
  for (uint64_t t_patch = 0; t_patch < 565412ULL; ++t_patch)
  {
#pragma HLS pipeline II = 1 style = stp  // inv3d: was auto
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_31_buf distance = 31
    if (true)  // patch: blocking reads (was: !fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_31 = fifo_ref_0_delayed_31_buf[ptr_delay_31];;
      const float let_0 = fifo_ref_0_delayed_31;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      fifo_ref_0_delayed_31_buf[ptr_delay_31] = fifo_ref_0;
      ptr_delay_31 < 30 ? (++ptr_delay_31) : (ptr_delay_31 = 0);
    } // if not empty
  } // for module_5
} // Module5Func

void Module6Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<1> ptr_delay_1 = 0;
  float fifo_ref_0_delayed_1_buf[1];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_1_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_6:
  bool enable = true;  // patch: fixed trip count 565412 (measured at x 128 128 128)
  for (uint64_t t_patch = 0; t_patch < 565412ULL; ++t_patch)
  {
#pragma HLS pipeline II = 1 style = stp  // inv3d: was auto
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_1_buf distance = 1
    if (true)  // patch: blocking reads (was: !fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const float let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_6
} // Module6Func

void Module7Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<5> ptr_delay_32 = 0;
  float fifo_ref_0_delayed_32_buf[32];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_32_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_7:
  bool enable = true;  // patch: fixed trip count 565412 (measured at x 128 128 128)
  for (uint64_t t_patch = 0; t_patch < 565412ULL; ++t_patch)
  {
#pragma HLS pipeline II = 1 style = stp  // inv3d: was auto
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_32_buf distance = 32
    if (true)  // patch: blocking reads (was: !fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_32 = fifo_ref_0_delayed_32_buf[ptr_delay_32];;
      const float let_0 = fifo_ref_0_delayed_32;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      fifo_ref_0_delayed_32_buf[ptr_delay_32] = fifo_ref_0;
      ptr_delay_32 < 31 ? (++ptr_delay_32) : (ptr_delay_32 = 0);
    } // if not empty
  } // for module_7
} // Module7Func

void Module8Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<1> ptr_delay_1 = 0;
  float fifo_ref_0_delayed_1_buf[1];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_1_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_8:
  bool enable = true;  // patch: fixed trip count 565412 (measured at x 128 128 128)
  for (uint64_t t_patch = 0; t_patch < 565412ULL; ++t_patch)
  {
#pragma HLS pipeline II = 1 style = stp  // inv3d: was auto
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_1_buf distance = 1
    if (true)  // patch: blocking reads (was: !fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const float let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_8
} // Module8Func

void Module9Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<12> ptr_delay_4064 = 0;
  float fifo_ref_0_delayed_4064_buf[4064];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_4064_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
module_9:
  bool enable = true;  // patch: fixed trip count 565412 (measured at x 128 128 128)
  for (uint64_t t_patch = 0; t_patch < 565412ULL; ++t_patch)
  {
#pragma HLS pipeline II = 1 style = stp  // inv3d: was auto
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_4064_buf distance = 4064
    if (true)  // patch: blocking reads (was: !fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_4064 = fifo_ref_0_delayed_4064_buf[ptr_delay_4064];;
      const float let_0 = fifo_ref_0_delayed_4064;
      WriteData(fifo_st_0, float(let_0), enabled);
      fifo_ref_0_delayed_4064_buf[ptr_delay_4064] = fifo_ref_0;
      ptr_delay_4064 < 4063 ? (++ptr_delay_4064) : (ptr_delay_4064 = 0);
    } // if not empty
  } // for module_9
} // Module9Func

void Module10Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_3, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_4, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_5, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_6)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#pragma HLS aggregate bit variable = fifo_ld_2
#pragma HLS aggregate bit variable = fifo_ld_3
#pragma HLS aggregate bit variable = fifo_ld_4
#pragma HLS aggregate bit variable = fifo_ld_5
#pragma HLS aggregate bit variable = fifo_ld_6
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_10:
  bool enable = true;  // patch: fixed trip count 565412 (measured at x 128 128 128)
  for (uint64_t t_patch = 0; t_patch < 565412ULL; ++t_patch)
  {
#pragma HLS pipeline II = 1 style = stp  // inv3d: was auto
    if (true)  // patch: blocking reads (was: !fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty() && !fifo_ld_4.empty() && !fifo_ld_5.empty() && !fifo_ld_6.empty())
    {
      float fifo_ref_0;
      float fifo_ref_1;
      float fifo_ref_2;
      float fifo_ref_3;
      float fifo_ref_4;
      float fifo_ref_5;
      float fifo_ref_6;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool fifo_ref_3_enable = ReadData(fifo_ref_3, fifo_ld_3);
      const bool fifo_ref_4_enable = ReadData(fifo_ref_4, fifo_ld_4);
      const bool fifo_ref_5_enable = ReadData(fifo_ref_5, fifo_ld_5);
      const bool fifo_ref_6_enable = ReadData(fifo_ref_6, fifo_ld_6);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable && fifo_ref_4_enable && fifo_ref_5_enable && fifo_ref_6_enable;
      enable = enabled;
      WriteData(fifo_st_0, float((float)((.125f * fifo_ref_0) + (.125f * fifo_ref_1) + (.125f * fifo_ref_2) + (.125f * fifo_ref_3) + (.125f * fifo_ref_4) + (.125f * fifo_ref_5) + (.25f * fifo_ref_6))), enabled);
    } // if not empty
  } // for module_10
} // Module10Func

void Module11Func(
  /*output*/ hls::stream<Data<ap_uint<512>>>& dram_out_bank_1_fifo, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_3)
{
#pragma HLS aggregate bit variable = dram_out_bank_1_fifo
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#pragma HLS aggregate bit variable = fifo_ld_2
#pragma HLS aggregate bit variable = fifo_ld_3
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
module_11:
  bool enable = true;  // patch: fixed trip count 141353 (measured at x 128 128 128)
  for (uint64_t t_patch = 0; t_patch < 141353ULL; ++t_patch)
  {
#pragma HLS pipeline II = 4 style = stp  // inv3d: was auto
    if (true)  // patch: blocking reads (was: !fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty())
    {
      float fifo_ref_0;
      float fifo_ref_1;
      float fifo_ref_2;
      float fifo_ref_3;
      ap_uint<512> dram_out_bank_1_buf;
      ReadData(fifo_ref_0, fifo_ld_0);
      ReadData(fifo_ref_1, fifo_ld_1);
      ReadData(fifo_ref_2, fifo_ld_2);
      ReadData(fifo_ref_3, fifo_ld_3);
      dram_out_bank_1_buf(95, 64) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      dram_out_bank_1_buf(63, 32) = Reinterpret<ap_uint<32>>(fifo_ref_1);
      dram_out_bank_1_buf(127, 96) = Reinterpret<ap_uint<32>>(fifo_ref_2);
      dram_out_bank_1_buf(31, 0) = Reinterpret<ap_uint<32>>(fifo_ref_3);
      ReadData(fifo_ref_0, fifo_ld_0);
      ReadData(fifo_ref_1, fifo_ld_1);
      ReadData(fifo_ref_2, fifo_ld_2);
      ReadData(fifo_ref_3, fifo_ld_3);
      dram_out_bank_1_buf(223, 192) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      dram_out_bank_1_buf(191, 160) = Reinterpret<ap_uint<32>>(fifo_ref_1);
      dram_out_bank_1_buf(255, 224) = Reinterpret<ap_uint<32>>(fifo_ref_2);
      dram_out_bank_1_buf(159, 128) = Reinterpret<ap_uint<32>>(fifo_ref_3);
      ReadData(fifo_ref_0, fifo_ld_0);
      ReadData(fifo_ref_1, fifo_ld_1);
      ReadData(fifo_ref_2, fifo_ld_2);
      ReadData(fifo_ref_3, fifo_ld_3);
      dram_out_bank_1_buf(351, 320) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      dram_out_bank_1_buf(319, 288) = Reinterpret<ap_uint<32>>(fifo_ref_1);
      dram_out_bank_1_buf(383, 352) = Reinterpret<ap_uint<32>>(fifo_ref_2);
      dram_out_bank_1_buf(287, 256) = Reinterpret<ap_uint<32>>(fifo_ref_3);
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool fifo_ref_3_enable = ReadData(fifo_ref_3, fifo_ld_3);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable;
      enable = enabled;
      dram_out_bank_1_buf(479, 448) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      dram_out_bank_1_buf(447, 416) = Reinterpret<ap_uint<32>>(fifo_ref_1);
      dram_out_bank_1_buf(511, 480) = Reinterpret<ap_uint<32>>(fifo_ref_2);
      dram_out_bank_1_buf(415, 384) = Reinterpret<ap_uint<32>>(fifo_ref_3);
      WriteData(dram_out_bank_1_fifo, dram_out_bank_1_buf, enabled);
    } // if not empty
  } // for module_11
} // Module11Func

extern "C" {

void heat3d_kernel(
  ap_uint<512>* bank_1_out, 
  ap_uint<512>* bank_0_in, 
  uint64_t coalesced_data_num)
{
#pragma HLS interface m_axi port = bank_1_out offset = slave bundle = out_bank_1 depth = 141354
#pragma HLS interface m_axi port = bank_0_in offset = slave bundle = in_bank_0 depth = 141354
#pragma HLS interface s_axilite port = bank_1_out bundle = control
#pragma HLS interface s_axilite port = bank_0_in bundle = control
#pragma HLS interface s_axilite port = coalesced_data_num bundle = control
#pragma HLS interface s_axilite port = return bundle = control

  hls::stream<Data<ap_uint<512>>> bank_0_in_buf("bank_0_in_buf");
#pragma HLS stream variable = bank_0_in_buf depth = 32
#pragma HLS aggregate bit variable = bank_0_in_buf
  hls::stream<Data<ap_uint<512>>> bank_1_out_buf("bank_1_out_buf");
#pragma HLS stream variable = bank_1_out_buf depth = 32
#pragma HLS aggregate bit variable = bank_1_out_buf

  hls::stream<Data<float>> from_in_bank_0_to_in_offset_0("from_in_bank_0_to_in_offset_0");
#pragma HLS stream variable = from_in_bank_0_to_in_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_in_bank_0_to_in_offset_0
  hls::stream<Data<float>> from_in_bank_0_to_in_offset_1("from_in_bank_0_to_in_offset_1");
#pragma HLS stream variable = from_in_bank_0_to_in_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_in_bank_0_to_in_offset_1
  hls::stream<Data<float>> from_in_bank_0_to_in_offset_2("from_in_bank_0_to_in_offset_2");
#pragma HLS stream variable = from_in_bank_0_to_in_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_in_bank_0_to_in_offset_2
  hls::stream<Data<float>> from_in_bank_0_to_in_offset_3("from_in_bank_0_to_in_offset_3");
#pragma HLS stream variable = from_in_bank_0_to_in_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_in_bank_0_to_in_offset_3
  hls::stream<Data<float>> from_in_offset_0_to_in_offset_16256("from_in_offset_0_to_in_offset_16256");
#pragma HLS stream variable = from_in_offset_0_to_in_offset_16256 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_0_to_in_offset_16256
  hls::stream<Data<float>> from_in_offset_0_to_in_iter1_pe_3("from_in_offset_0_to_in_iter1_pe_3");
#pragma HLS stream variable = from_in_offset_0_to_in_iter1_pe_3 depth = 13
#pragma HLS aggregate bit variable = from_in_offset_0_to_in_iter1_pe_3
  hls::stream<Data<float>> from_in_offset_1_to_in_offset_16257("from_in_offset_1_to_in_offset_16257");
#pragma HLS stream variable = from_in_offset_1_to_in_offset_16257 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_1_to_in_offset_16257
  hls::stream<Data<float>> from_in_offset_1_to_in_iter1_pe_2("from_in_offset_1_to_in_iter1_pe_2");
#pragma HLS stream variable = from_in_offset_1_to_in_iter1_pe_2 depth = 11
#pragma HLS aggregate bit variable = from_in_offset_1_to_in_iter1_pe_2
  hls::stream<Data<float>> from_in_offset_2_to_in_offset_16258("from_in_offset_2_to_in_offset_16258");
#pragma HLS stream variable = from_in_offset_2_to_in_offset_16258 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_2_to_in_offset_16258
  hls::stream<Data<float>> from_in_offset_2_to_in_iter1_pe_1("from_in_offset_2_to_in_iter1_pe_1");
#pragma HLS stream variable = from_in_offset_2_to_in_iter1_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_in_offset_2_to_in_iter1_pe_1
  hls::stream<Data<float>> from_in_offset_3_to_in_offset_16259("from_in_offset_3_to_in_offset_16259");
#pragma HLS stream variable = from_in_offset_3_to_in_offset_16259 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_3_to_in_offset_16259
  hls::stream<Data<float>> from_in_offset_3_to_in_iter1_pe_0("from_in_offset_3_to_in_iter1_pe_0");
#pragma HLS stream variable = from_in_offset_3_to_in_iter1_pe_0 depth = 13
#pragma HLS aggregate bit variable = from_in_offset_3_to_in_iter1_pe_0
  hls::stream<Data<float>> from_in_offset_16256_to_in_offset_16384("from_in_offset_16256_to_in_offset_16384");
#pragma HLS stream variable = from_in_offset_16256_to_in_offset_16384 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_16256_to_in_offset_16384
  hls::stream<Data<float>> from_in_offset_16256_to_in_iter1_pe_3("from_in_offset_16256_to_in_iter1_pe_3");
#pragma HLS stream variable = from_in_offset_16256_to_in_iter1_pe_3 depth = 11
#pragma HLS aggregate bit variable = from_in_offset_16256_to_in_iter1_pe_3
  hls::stream<Data<float>> from_in_offset_16257_to_in_offset_16385("from_in_offset_16257_to_in_offset_16385");
#pragma HLS stream variable = from_in_offset_16257_to_in_offset_16385 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_16257_to_in_offset_16385
  hls::stream<Data<float>> from_in_offset_16257_to_in_iter1_pe_2("from_in_offset_16257_to_in_iter1_pe_2");
#pragma HLS stream variable = from_in_offset_16257_to_in_iter1_pe_2 depth = 9
#pragma HLS aggregate bit variable = from_in_offset_16257_to_in_iter1_pe_2
  hls::stream<Data<float>> from_in_offset_16258_to_in_offset_16386("from_in_offset_16258_to_in_offset_16386");
#pragma HLS stream variable = from_in_offset_16258_to_in_offset_16386 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_16258_to_in_offset_16386
  hls::stream<Data<float>> from_in_offset_16258_to_in_iter1_pe_1("from_in_offset_16258_to_in_iter1_pe_1");
#pragma HLS stream variable = from_in_offset_16258_to_in_iter1_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_in_offset_16258_to_in_iter1_pe_1
  hls::stream<Data<float>> from_in_offset_16259_to_in_offset_16383("from_in_offset_16259_to_in_offset_16383");
#pragma HLS stream variable = from_in_offset_16259_to_in_offset_16383 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_16259_to_in_offset_16383
  hls::stream<Data<float>> from_in_offset_16259_to_in_iter1_pe_0("from_in_offset_16259_to_in_iter1_pe_0");
#pragma HLS stream variable = from_in_offset_16259_to_in_iter1_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_in_offset_16259_to_in_iter1_pe_0
  hls::stream<Data<float>> from_in_offset_16384_to_in_offset_16388("from_in_offset_16384_to_in_offset_16388");
#pragma HLS stream variable = from_in_offset_16384_to_in_offset_16388 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_16384_to_in_offset_16388
  hls::stream<Data<float>> from_in_offset_16384_to_in_iter1_pe_3("from_in_offset_16384_to_in_iter1_pe_3");
#pragma HLS stream variable = from_in_offset_16384_to_in_iter1_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_in_offset_16384_to_in_iter1_pe_3
  hls::stream<Data<float>> from_in_offset_16384_to_in_iter1_pe_2("from_in_offset_16384_to_in_iter1_pe_2");
#pragma HLS stream variable = from_in_offset_16384_to_in_iter1_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_in_offset_16384_to_in_iter1_pe_2
  hls::stream<Data<float>> from_in_offset_16385_to_in_offset_16513("from_in_offset_16385_to_in_offset_16513");
#pragma HLS stream variable = from_in_offset_16385_to_in_offset_16513 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_16385_to_in_offset_16513
  hls::stream<Data<float>> from_in_offset_16385_to_in_iter1_pe_3("from_in_offset_16385_to_in_iter1_pe_3");
#pragma HLS stream variable = from_in_offset_16385_to_in_iter1_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_in_offset_16385_to_in_iter1_pe_3
  hls::stream<Data<float>> from_in_offset_16385_to_in_iter1_pe_2("from_in_offset_16385_to_in_iter1_pe_2");
#pragma HLS stream variable = from_in_offset_16385_to_in_iter1_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_in_offset_16385_to_in_iter1_pe_2
  hls::stream<Data<float>> from_in_offset_16385_to_in_iter1_pe_1("from_in_offset_16385_to_in_iter1_pe_1");
#pragma HLS stream variable = from_in_offset_16385_to_in_iter1_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_in_offset_16385_to_in_iter1_pe_1
  hls::stream<Data<float>> from_in_offset_16386_to_in_offset_16514("from_in_offset_16386_to_in_offset_16514");
#pragma HLS stream variable = from_in_offset_16386_to_in_offset_16514 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_16386_to_in_offset_16514
  hls::stream<Data<float>> from_in_offset_16386_to_in_iter1_pe_2("from_in_offset_16386_to_in_iter1_pe_2");
#pragma HLS stream variable = from_in_offset_16386_to_in_iter1_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_in_offset_16386_to_in_iter1_pe_2
  hls::stream<Data<float>> from_in_offset_16386_to_in_iter1_pe_1("from_in_offset_16386_to_in_iter1_pe_1");
#pragma HLS stream variable = from_in_offset_16386_to_in_iter1_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_in_offset_16386_to_in_iter1_pe_1
  hls::stream<Data<float>> from_in_offset_16386_to_in_iter1_pe_0("from_in_offset_16386_to_in_iter1_pe_0");
#pragma HLS stream variable = from_in_offset_16386_to_in_iter1_pe_0 depth = 9
#pragma HLS aggregate bit variable = from_in_offset_16386_to_in_iter1_pe_0
  hls::stream<Data<float>> from_in_offset_16383_to_in_offset_16387("from_in_offset_16383_to_in_offset_16387");
#pragma HLS stream variable = from_in_offset_16383_to_in_offset_16387 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_16383_to_in_offset_16387
  hls::stream<Data<float>> from_in_offset_16383_to_in_iter1_pe_3("from_in_offset_16383_to_in_iter1_pe_3");
#pragma HLS stream variable = from_in_offset_16383_to_in_iter1_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_in_offset_16383_to_in_iter1_pe_3
  hls::stream<Data<float>> from_in_offset_16388_to_in_offset_16512("from_in_offset_16388_to_in_offset_16512");
#pragma HLS stream variable = from_in_offset_16388_to_in_offset_16512 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_16388_to_in_offset_16512
  hls::stream<Data<float>> from_in_offset_16388_to_in_iter1_pe_0("from_in_offset_16388_to_in_iter1_pe_0");
#pragma HLS stream variable = from_in_offset_16388_to_in_iter1_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_in_offset_16388_to_in_iter1_pe_0
  hls::stream<Data<float>> from_in_offset_16513_to_in_offset_32769("from_in_offset_16513_to_in_offset_32769");
#pragma HLS stream variable = from_in_offset_16513_to_in_offset_32769 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_16513_to_in_offset_32769
  hls::stream<Data<float>> from_in_offset_16513_to_in_iter1_pe_2("from_in_offset_16513_to_in_iter1_pe_2");
#pragma HLS stream variable = from_in_offset_16513_to_in_iter1_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_in_offset_16513_to_in_iter1_pe_2
  hls::stream<Data<float>> from_in_offset_16514_to_in_offset_32770("from_in_offset_16514_to_in_offset_32770");
#pragma HLS stream variable = from_in_offset_16514_to_in_offset_32770 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_16514_to_in_offset_32770
  hls::stream<Data<float>> from_in_offset_16514_to_in_iter1_pe_1("from_in_offset_16514_to_in_iter1_pe_1");
#pragma HLS stream variable = from_in_offset_16514_to_in_iter1_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_in_offset_16514_to_in_iter1_pe_1
  hls::stream<Data<float>> from_in_offset_16387_to_in_offset_16515("from_in_offset_16387_to_in_offset_16515");
#pragma HLS stream variable = from_in_offset_16387_to_in_offset_16515 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_16387_to_in_offset_16515
  hls::stream<Data<float>> from_in_offset_16387_to_in_iter1_pe_1("from_in_offset_16387_to_in_iter1_pe_1");
#pragma HLS stream variable = from_in_offset_16387_to_in_iter1_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_in_offset_16387_to_in_iter1_pe_1
  hls::stream<Data<float>> from_in_offset_16387_to_in_iter1_pe_0("from_in_offset_16387_to_in_iter1_pe_0");
#pragma HLS stream variable = from_in_offset_16387_to_in_iter1_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_in_offset_16387_to_in_iter1_pe_0
  hls::stream<Data<float>> from_in_offset_16512_to_in_offset_32768("from_in_offset_16512_to_in_offset_32768");
#pragma HLS stream variable = from_in_offset_16512_to_in_offset_32768 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_16512_to_in_offset_32768
  hls::stream<Data<float>> from_in_offset_16512_to_in_iter1_pe_3("from_in_offset_16512_to_in_iter1_pe_3");
#pragma HLS stream variable = from_in_offset_16512_to_in_iter1_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_in_offset_16512_to_in_iter1_pe_3
  hls::stream<Data<float>> from_in_offset_32769_to_in_iter1_pe_2("from_in_offset_32769_to_in_iter1_pe_2");
#pragma HLS stream variable = from_in_offset_32769_to_in_iter1_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_32769_to_in_iter1_pe_2
  hls::stream<Data<float>> from_in_iter1_pe_2_to_in_iter1_offset_1("from_in_iter1_pe_2_to_in_iter1_offset_1");
#pragma HLS stream variable = from_in_iter1_pe_2_to_in_iter1_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_pe_2_to_in_iter1_offset_1
  hls::stream<Data<float>> from_in_iter1_offset_1_to_in_iter1_offset_16257("from_in_iter1_offset_1_to_in_iter1_offset_16257");
#pragma HLS stream variable = from_in_iter1_offset_1_to_in_iter1_offset_16257 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_1_to_in_iter1_offset_16257
  hls::stream<Data<float>> from_in_iter1_offset_1_to_in_iter2_pe_2("from_in_iter1_offset_1_to_in_iter2_pe_2");
#pragma HLS stream variable = from_in_iter1_offset_1_to_in_iter2_pe_2 depth = 11
#pragma HLS aggregate bit variable = from_in_iter1_offset_1_to_in_iter2_pe_2
  hls::stream<Data<float>> from_in_iter1_offset_16257_to_in_iter1_offset_16385("from_in_iter1_offset_16257_to_in_iter1_offset_16385");
#pragma HLS stream variable = from_in_iter1_offset_16257_to_in_iter1_offset_16385 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_16257_to_in_iter1_offset_16385
  hls::stream<Data<float>> from_in_iter1_offset_16257_to_in_iter2_pe_2("from_in_iter1_offset_16257_to_in_iter2_pe_2");
#pragma HLS stream variable = from_in_iter1_offset_16257_to_in_iter2_pe_2 depth = 9
#pragma HLS aggregate bit variable = from_in_iter1_offset_16257_to_in_iter2_pe_2
  hls::stream<Data<float>> from_in_iter1_offset_16385_to_in_iter1_offset_16513("from_in_iter1_offset_16385_to_in_iter1_offset_16513");
#pragma HLS stream variable = from_in_iter1_offset_16385_to_in_iter1_offset_16513 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_16385_to_in_iter1_offset_16513
  hls::stream<Data<float>> from_in_iter1_offset_16385_to_in_iter2_pe_3("from_in_iter1_offset_16385_to_in_iter2_pe_3");
#pragma HLS stream variable = from_in_iter1_offset_16385_to_in_iter2_pe_3 depth = 11
#pragma HLS aggregate bit variable = from_in_iter1_offset_16385_to_in_iter2_pe_3
  hls::stream<Data<float>> from_in_iter1_offset_16385_to_in_iter2_pe_2("from_in_iter1_offset_16385_to_in_iter2_pe_2");
#pragma HLS stream variable = from_in_iter1_offset_16385_to_in_iter2_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_in_iter1_offset_16385_to_in_iter2_pe_2
  hls::stream<Data<float>> from_in_iter1_offset_16385_to_in_iter2_pe_1("from_in_iter1_offset_16385_to_in_iter2_pe_1");
#pragma HLS stream variable = from_in_iter1_offset_16385_to_in_iter2_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_in_iter1_offset_16385_to_in_iter2_pe_1
  hls::stream<Data<float>> from_in_offset_32770_to_in_iter1_pe_1("from_in_offset_32770_to_in_iter1_pe_1");
#pragma HLS stream variable = from_in_offset_32770_to_in_iter1_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_32770_to_in_iter1_pe_1
  hls::stream<Data<float>> from_in_iter1_pe_1_to_in_iter1_offset_2("from_in_iter1_pe_1_to_in_iter1_offset_2");
#pragma HLS stream variable = from_in_iter1_pe_1_to_in_iter1_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_pe_1_to_in_iter1_offset_2
  hls::stream<Data<float>> from_in_iter1_offset_2_to_in_iter1_offset_16258("from_in_iter1_offset_2_to_in_iter1_offset_16258");
#pragma HLS stream variable = from_in_iter1_offset_2_to_in_iter1_offset_16258 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_2_to_in_iter1_offset_16258
  hls::stream<Data<float>> from_in_iter1_offset_2_to_in_iter2_pe_1("from_in_iter1_offset_2_to_in_iter2_pe_1");
#pragma HLS stream variable = from_in_iter1_offset_2_to_in_iter2_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_in_iter1_offset_2_to_in_iter2_pe_1
  hls::stream<Data<float>> from_in_iter1_offset_16258_to_in_iter1_offset_16386("from_in_iter1_offset_16258_to_in_iter1_offset_16386");
#pragma HLS stream variable = from_in_iter1_offset_16258_to_in_iter1_offset_16386 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_16258_to_in_iter1_offset_16386
  hls::stream<Data<float>> from_in_iter1_offset_16258_to_in_iter2_pe_1("from_in_iter1_offset_16258_to_in_iter2_pe_1");
#pragma HLS stream variable = from_in_iter1_offset_16258_to_in_iter2_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_in_iter1_offset_16258_to_in_iter2_pe_1
  hls::stream<Data<float>> from_in_iter1_offset_16386_to_in_iter1_offset_16514("from_in_iter1_offset_16386_to_in_iter1_offset_16514");
#pragma HLS stream variable = from_in_iter1_offset_16386_to_in_iter1_offset_16514 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_16386_to_in_iter1_offset_16514
  hls::stream<Data<float>> from_in_iter1_offset_16386_to_in_iter2_pe_2("from_in_iter1_offset_16386_to_in_iter2_pe_2");
#pragma HLS stream variable = from_in_iter1_offset_16386_to_in_iter2_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_in_iter1_offset_16386_to_in_iter2_pe_2
  hls::stream<Data<float>> from_in_iter1_offset_16386_to_in_iter2_pe_1("from_in_iter1_offset_16386_to_in_iter2_pe_1");
#pragma HLS stream variable = from_in_iter1_offset_16386_to_in_iter2_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_in_iter1_offset_16386_to_in_iter2_pe_1
  hls::stream<Data<float>> from_in_iter1_offset_16386_to_in_iter2_pe_0("from_in_iter1_offset_16386_to_in_iter2_pe_0");
#pragma HLS stream variable = from_in_iter1_offset_16386_to_in_iter2_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_in_iter1_offset_16386_to_in_iter2_pe_0
  hls::stream<Data<float>> from_in_offset_16515_to_in_offset_32771("from_in_offset_16515_to_in_offset_32771");
#pragma HLS stream variable = from_in_offset_16515_to_in_offset_32771 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_16515_to_in_offset_32771
  hls::stream<Data<float>> from_in_offset_16515_to_in_iter1_pe_0("from_in_offset_16515_to_in_iter1_pe_0");
#pragma HLS stream variable = from_in_offset_16515_to_in_iter1_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_in_offset_16515_to_in_iter1_pe_0
  hls::stream<Data<float>> from_in_offset_32768_to_in_iter1_pe_3("from_in_offset_32768_to_in_iter1_pe_3");
#pragma HLS stream variable = from_in_offset_32768_to_in_iter1_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_32768_to_in_iter1_pe_3
  hls::stream<Data<float>> from_in_iter1_pe_3_to_in_iter1_offset_0("from_in_iter1_pe_3_to_in_iter1_offset_0");
#pragma HLS stream variable = from_in_iter1_pe_3_to_in_iter1_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_pe_3_to_in_iter1_offset_0
  hls::stream<Data<float>> from_in_iter1_offset_0_to_in_iter1_offset_16256("from_in_iter1_offset_0_to_in_iter1_offset_16256");
#pragma HLS stream variable = from_in_iter1_offset_0_to_in_iter1_offset_16256 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_0_to_in_iter1_offset_16256
  hls::stream<Data<float>> from_in_iter1_offset_0_to_in_iter2_pe_3("from_in_iter1_offset_0_to_in_iter2_pe_3");
#pragma HLS stream variable = from_in_iter1_offset_0_to_in_iter2_pe_3 depth = 13
#pragma HLS aggregate bit variable = from_in_iter1_offset_0_to_in_iter2_pe_3
  hls::stream<Data<float>> from_in_iter1_offset_16256_to_in_iter1_offset_16384("from_in_iter1_offset_16256_to_in_iter1_offset_16384");
#pragma HLS stream variable = from_in_iter1_offset_16256_to_in_iter1_offset_16384 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_16256_to_in_iter1_offset_16384
  hls::stream<Data<float>> from_in_iter1_offset_16256_to_in_iter2_pe_3("from_in_iter1_offset_16256_to_in_iter2_pe_3");
#pragma HLS stream variable = from_in_iter1_offset_16256_to_in_iter2_pe_3 depth = 11
#pragma HLS aggregate bit variable = from_in_iter1_offset_16256_to_in_iter2_pe_3
  hls::stream<Data<float>> from_in_iter1_offset_16384_to_in_iter1_offset_16388("from_in_iter1_offset_16384_to_in_iter1_offset_16388");
#pragma HLS stream variable = from_in_iter1_offset_16384_to_in_iter1_offset_16388 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_16384_to_in_iter1_offset_16388
  hls::stream<Data<float>> from_in_iter1_offset_16384_to_in_iter2_pe_3("from_in_iter1_offset_16384_to_in_iter2_pe_3");
#pragma HLS stream variable = from_in_iter1_offset_16384_to_in_iter2_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_in_iter1_offset_16384_to_in_iter2_pe_3
  hls::stream<Data<float>> from_in_iter1_offset_16384_to_in_iter2_pe_2("from_in_iter1_offset_16384_to_in_iter2_pe_2");
#pragma HLS stream variable = from_in_iter1_offset_16384_to_in_iter2_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_in_iter1_offset_16384_to_in_iter2_pe_2
  hls::stream<Data<float>> from_in_iter1_offset_16388_to_in_iter1_offset_16512("from_in_iter1_offset_16388_to_in_iter1_offset_16512");
#pragma HLS stream variable = from_in_iter1_offset_16388_to_in_iter1_offset_16512 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_16388_to_in_iter1_offset_16512
  hls::stream<Data<float>> from_in_iter1_offset_16388_to_in_iter2_pe_0("from_in_iter1_offset_16388_to_in_iter2_pe_0");
#pragma HLS stream variable = from_in_iter1_offset_16388_to_in_iter2_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_in_iter1_offset_16388_to_in_iter2_pe_0
  hls::stream<Data<float>> from_in_iter1_offset_16513_to_in_iter1_offset_32769("from_in_iter1_offset_16513_to_in_iter1_offset_32769");
#pragma HLS stream variable = from_in_iter1_offset_16513_to_in_iter1_offset_32769 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_16513_to_in_iter1_offset_32769
  hls::stream<Data<float>> from_in_iter1_offset_16513_to_in_iter2_pe_2("from_in_iter1_offset_16513_to_in_iter2_pe_2");
#pragma HLS stream variable = from_in_iter1_offset_16513_to_in_iter2_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_in_iter1_offset_16513_to_in_iter2_pe_2
  hls::stream<Data<float>> from_in_iter1_offset_16514_to_in_iter1_offset_32770("from_in_iter1_offset_16514_to_in_iter1_offset_32770");
#pragma HLS stream variable = from_in_iter1_offset_16514_to_in_iter1_offset_32770 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_16514_to_in_iter1_offset_32770
  hls::stream<Data<float>> from_in_iter1_offset_16514_to_in_iter2_pe_1("from_in_iter1_offset_16514_to_in_iter2_pe_1");
#pragma HLS stream variable = from_in_iter1_offset_16514_to_in_iter2_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_in_iter1_offset_16514_to_in_iter2_pe_1
  hls::stream<Data<float>> from_in_offset_32771_to_in_iter1_pe_0("from_in_offset_32771_to_in_iter1_pe_0");
#pragma HLS stream variable = from_in_offset_32771_to_in_iter1_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_in_offset_32771_to_in_iter1_pe_0
  hls::stream<Data<float>> from_in_iter1_pe_0_to_in_iter1_offset_3("from_in_iter1_pe_0_to_in_iter1_offset_3");
#pragma HLS stream variable = from_in_iter1_pe_0_to_in_iter1_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_pe_0_to_in_iter1_offset_3
  hls::stream<Data<float>> from_in_iter1_offset_3_to_in_iter1_offset_16259("from_in_iter1_offset_3_to_in_iter1_offset_16259");
#pragma HLS stream variable = from_in_iter1_offset_3_to_in_iter1_offset_16259 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_3_to_in_iter1_offset_16259
  hls::stream<Data<float>> from_in_iter1_offset_3_to_in_iter2_pe_0("from_in_iter1_offset_3_to_in_iter2_pe_0");
#pragma HLS stream variable = from_in_iter1_offset_3_to_in_iter2_pe_0 depth = 13
#pragma HLS aggregate bit variable = from_in_iter1_offset_3_to_in_iter2_pe_0
  hls::stream<Data<float>> from_in_iter1_offset_16259_to_in_iter1_offset_16383("from_in_iter1_offset_16259_to_in_iter1_offset_16383");
#pragma HLS stream variable = from_in_iter1_offset_16259_to_in_iter1_offset_16383 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_16259_to_in_iter1_offset_16383
  hls::stream<Data<float>> from_in_iter1_offset_16259_to_in_iter2_pe_0("from_in_iter1_offset_16259_to_in_iter2_pe_0");
#pragma HLS stream variable = from_in_iter1_offset_16259_to_in_iter2_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_in_iter1_offset_16259_to_in_iter2_pe_0
  hls::stream<Data<float>> from_in_iter1_offset_16383_to_in_iter1_offset_16387("from_in_iter1_offset_16383_to_in_iter1_offset_16387");
#pragma HLS stream variable = from_in_iter1_offset_16383_to_in_iter1_offset_16387 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_16383_to_in_iter1_offset_16387
  hls::stream<Data<float>> from_in_iter1_offset_16383_to_in_iter2_pe_3("from_in_iter1_offset_16383_to_in_iter2_pe_3");
#pragma HLS stream variable = from_in_iter1_offset_16383_to_in_iter2_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_in_iter1_offset_16383_to_in_iter2_pe_3
  hls::stream<Data<float>> from_in_iter1_offset_16387_to_in_iter1_offset_16515("from_in_iter1_offset_16387_to_in_iter1_offset_16515");
#pragma HLS stream variable = from_in_iter1_offset_16387_to_in_iter1_offset_16515 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_16387_to_in_iter1_offset_16515
  hls::stream<Data<float>> from_in_iter1_offset_16387_to_in_iter2_pe_1("from_in_iter1_offset_16387_to_in_iter2_pe_1");
#pragma HLS stream variable = from_in_iter1_offset_16387_to_in_iter2_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_16387_to_in_iter2_pe_1
  hls::stream<Data<float>> from_in_iter1_offset_16387_to_in_iter2_pe_0("from_in_iter1_offset_16387_to_in_iter2_pe_0");
#pragma HLS stream variable = from_in_iter1_offset_16387_to_in_iter2_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_in_iter1_offset_16387_to_in_iter2_pe_0
  hls::stream<Data<float>> from_in_iter1_offset_16512_to_in_iter1_offset_32768("from_in_iter1_offset_16512_to_in_iter1_offset_32768");
#pragma HLS stream variable = from_in_iter1_offset_16512_to_in_iter1_offset_32768 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_16512_to_in_iter1_offset_32768
  hls::stream<Data<float>> from_in_iter1_offset_16512_to_in_iter2_pe_3("from_in_iter1_offset_16512_to_in_iter2_pe_3");
#pragma HLS stream variable = from_in_iter1_offset_16512_to_in_iter2_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_in_iter1_offset_16512_to_in_iter2_pe_3
  hls::stream<Data<float>> from_in_iter1_offset_32769_to_in_iter2_pe_2("from_in_iter1_offset_32769_to_in_iter2_pe_2");
#pragma HLS stream variable = from_in_iter1_offset_32769_to_in_iter2_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_32769_to_in_iter2_pe_2
  hls::stream<Data<float>> from_in_iter2_pe_2_to_in_iter2_offset_1("from_in_iter2_pe_2_to_in_iter2_offset_1");
#pragma HLS stream variable = from_in_iter2_pe_2_to_in_iter2_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_pe_2_to_in_iter2_offset_1
  hls::stream<Data<float>> from_in_iter2_offset_1_to_in_iter2_offset_16257("from_in_iter2_offset_1_to_in_iter2_offset_16257");
#pragma HLS stream variable = from_in_iter2_offset_1_to_in_iter2_offset_16257 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_1_to_in_iter2_offset_16257
  hls::stream<Data<float>> from_in_iter2_offset_1_to_in_iter3_pe_2("from_in_iter2_offset_1_to_in_iter3_pe_2");
#pragma HLS stream variable = from_in_iter2_offset_1_to_in_iter3_pe_2 depth = 11
#pragma HLS aggregate bit variable = from_in_iter2_offset_1_to_in_iter3_pe_2
  hls::stream<Data<float>> from_in_iter2_offset_16257_to_in_iter2_offset_16385("from_in_iter2_offset_16257_to_in_iter2_offset_16385");
#pragma HLS stream variable = from_in_iter2_offset_16257_to_in_iter2_offset_16385 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16257_to_in_iter2_offset_16385
  hls::stream<Data<float>> from_in_iter2_offset_16257_to_in_iter3_pe_2("from_in_iter2_offset_16257_to_in_iter3_pe_2");
#pragma HLS stream variable = from_in_iter2_offset_16257_to_in_iter3_pe_2 depth = 9
#pragma HLS aggregate bit variable = from_in_iter2_offset_16257_to_in_iter3_pe_2
  hls::stream<Data<float>> from_in_iter2_offset_16385_to_in_iter2_offset_16513("from_in_iter2_offset_16385_to_in_iter2_offset_16513");
#pragma HLS stream variable = from_in_iter2_offset_16385_to_in_iter2_offset_16513 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16385_to_in_iter2_offset_16513
  hls::stream<Data<float>> from_in_iter2_offset_16385_to_in_iter3_pe_3("from_in_iter2_offset_16385_to_in_iter3_pe_3");
#pragma HLS stream variable = from_in_iter2_offset_16385_to_in_iter3_pe_3 depth = 13
#pragma HLS aggregate bit variable = from_in_iter2_offset_16385_to_in_iter3_pe_3
  hls::stream<Data<float>> from_in_iter2_offset_16385_to_in_iter3_pe_2("from_in_iter2_offset_16385_to_in_iter3_pe_2");
#pragma HLS stream variable = from_in_iter2_offset_16385_to_in_iter3_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_in_iter2_offset_16385_to_in_iter3_pe_2
  hls::stream<Data<float>> from_in_iter2_offset_16385_to_in_iter3_pe_1("from_in_iter2_offset_16385_to_in_iter3_pe_1");
#pragma HLS stream variable = from_in_iter2_offset_16385_to_in_iter3_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_in_iter2_offset_16385_to_in_iter3_pe_1
  hls::stream<Data<float>> from_in_iter1_offset_32770_to_in_iter2_pe_1("from_in_iter1_offset_32770_to_in_iter2_pe_1");
#pragma HLS stream variable = from_in_iter1_offset_32770_to_in_iter2_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_32770_to_in_iter2_pe_1
  hls::stream<Data<float>> from_in_iter2_pe_1_to_in_iter2_offset_2("from_in_iter2_pe_1_to_in_iter2_offset_2");
#pragma HLS stream variable = from_in_iter2_pe_1_to_in_iter2_offset_2 depth = 5
#pragma HLS aggregate bit variable = from_in_iter2_pe_1_to_in_iter2_offset_2
  hls::stream<Data<float>> from_in_iter2_offset_2_to_in_iter2_offset_16258("from_in_iter2_offset_2_to_in_iter2_offset_16258");
#pragma HLS stream variable = from_in_iter2_offset_2_to_in_iter2_offset_16258 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_2_to_in_iter2_offset_16258
  hls::stream<Data<float>> from_in_iter2_offset_2_to_in_iter3_pe_1("from_in_iter2_offset_2_to_in_iter3_pe_1");
#pragma HLS stream variable = from_in_iter2_offset_2_to_in_iter3_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_in_iter2_offset_2_to_in_iter3_pe_1
  hls::stream<Data<float>> from_in_iter2_offset_16258_to_in_iter2_offset_16386("from_in_iter2_offset_16258_to_in_iter2_offset_16386");
#pragma HLS stream variable = from_in_iter2_offset_16258_to_in_iter2_offset_16386 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16258_to_in_iter2_offset_16386
  hls::stream<Data<float>> from_in_iter2_offset_16258_to_in_iter3_pe_1("from_in_iter2_offset_16258_to_in_iter3_pe_1");
#pragma HLS stream variable = from_in_iter2_offset_16258_to_in_iter3_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_in_iter2_offset_16258_to_in_iter3_pe_1
  hls::stream<Data<float>> from_in_iter2_offset_16386_to_in_iter2_offset_16514("from_in_iter2_offset_16386_to_in_iter2_offset_16514");
#pragma HLS stream variable = from_in_iter2_offset_16386_to_in_iter2_offset_16514 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16386_to_in_iter2_offset_16514
  hls::stream<Data<float>> from_in_iter2_offset_16386_to_in_iter3_pe_2("from_in_iter2_offset_16386_to_in_iter3_pe_2");
#pragma HLS stream variable = from_in_iter2_offset_16386_to_in_iter3_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_in_iter2_offset_16386_to_in_iter3_pe_2
  hls::stream<Data<float>> from_in_iter2_offset_16386_to_in_iter3_pe_1("from_in_iter2_offset_16386_to_in_iter3_pe_1");
#pragma HLS stream variable = from_in_iter2_offset_16386_to_in_iter3_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_in_iter2_offset_16386_to_in_iter3_pe_1
  hls::stream<Data<float>> from_in_iter2_offset_16386_to_in_iter3_pe_0("from_in_iter2_offset_16386_to_in_iter3_pe_0");
#pragma HLS stream variable = from_in_iter2_offset_16386_to_in_iter3_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_in_iter2_offset_16386_to_in_iter3_pe_0
  hls::stream<Data<float>> from_in_iter1_offset_16515_to_in_iter1_offset_32771("from_in_iter1_offset_16515_to_in_iter1_offset_32771");
#pragma HLS stream variable = from_in_iter1_offset_16515_to_in_iter1_offset_32771 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_16515_to_in_iter1_offset_32771
  hls::stream<Data<float>> from_in_iter1_offset_16515_to_in_iter2_pe_0("from_in_iter1_offset_16515_to_in_iter2_pe_0");
#pragma HLS stream variable = from_in_iter1_offset_16515_to_in_iter2_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_in_iter1_offset_16515_to_in_iter2_pe_0
  hls::stream<Data<float>> from_in_iter1_offset_32768_to_in_iter2_pe_3("from_in_iter1_offset_32768_to_in_iter2_pe_3");
#pragma HLS stream variable = from_in_iter1_offset_32768_to_in_iter2_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_32768_to_in_iter2_pe_3
  hls::stream<Data<float>> from_in_iter2_pe_3_to_in_iter2_offset_0("from_in_iter2_pe_3_to_in_iter2_offset_0");
#pragma HLS stream variable = from_in_iter2_pe_3_to_in_iter2_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_pe_3_to_in_iter2_offset_0
  hls::stream<Data<float>> from_in_iter2_offset_0_to_in_iter2_offset_16256("from_in_iter2_offset_0_to_in_iter2_offset_16256");
#pragma HLS stream variable = from_in_iter2_offset_0_to_in_iter2_offset_16256 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_0_to_in_iter2_offset_16256
  hls::stream<Data<float>> from_in_iter2_offset_0_to_in_iter3_pe_3("from_in_iter2_offset_0_to_in_iter3_pe_3");
#pragma HLS stream variable = from_in_iter2_offset_0_to_in_iter3_pe_3 depth = 13
#pragma HLS aggregate bit variable = from_in_iter2_offset_0_to_in_iter3_pe_3
  hls::stream<Data<float>> from_in_iter2_offset_16256_to_in_iter2_offset_16384("from_in_iter2_offset_16256_to_in_iter2_offset_16384");
#pragma HLS stream variable = from_in_iter2_offset_16256_to_in_iter2_offset_16384 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16256_to_in_iter2_offset_16384
  hls::stream<Data<float>> from_in_iter2_offset_16256_to_in_iter3_pe_3("from_in_iter2_offset_16256_to_in_iter3_pe_3");
#pragma HLS stream variable = from_in_iter2_offset_16256_to_in_iter3_pe_3 depth = 11
#pragma HLS aggregate bit variable = from_in_iter2_offset_16256_to_in_iter3_pe_3
  hls::stream<Data<float>> from_in_iter2_offset_16384_to_in_iter2_offset_16388("from_in_iter2_offset_16384_to_in_iter2_offset_16388");
#pragma HLS stream variable = from_in_iter2_offset_16384_to_in_iter2_offset_16388 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16384_to_in_iter2_offset_16388
  hls::stream<Data<float>> from_in_iter2_offset_16384_to_in_iter3_pe_3("from_in_iter2_offset_16384_to_in_iter3_pe_3");
#pragma HLS stream variable = from_in_iter2_offset_16384_to_in_iter3_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_in_iter2_offset_16384_to_in_iter3_pe_3
  hls::stream<Data<float>> from_in_iter2_offset_16384_to_in_iter3_pe_2("from_in_iter2_offset_16384_to_in_iter3_pe_2");
#pragma HLS stream variable = from_in_iter2_offset_16384_to_in_iter3_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16384_to_in_iter3_pe_2
  hls::stream<Data<float>> from_in_iter2_offset_16388_to_in_iter2_offset_16512("from_in_iter2_offset_16388_to_in_iter2_offset_16512");
#pragma HLS stream variable = from_in_iter2_offset_16388_to_in_iter2_offset_16512 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16388_to_in_iter2_offset_16512
  hls::stream<Data<float>> from_in_iter2_offset_16388_to_in_iter3_pe_0("from_in_iter2_offset_16388_to_in_iter3_pe_0");
#pragma HLS stream variable = from_in_iter2_offset_16388_to_in_iter3_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_in_iter2_offset_16388_to_in_iter3_pe_0
  hls::stream<Data<float>> from_in_iter2_offset_16513_to_in_iter2_offset_32769("from_in_iter2_offset_16513_to_in_iter2_offset_32769");
#pragma HLS stream variable = from_in_iter2_offset_16513_to_in_iter2_offset_32769 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16513_to_in_iter2_offset_32769
  hls::stream<Data<float>> from_in_iter2_offset_16513_to_in_iter3_pe_2("from_in_iter2_offset_16513_to_in_iter3_pe_2");
#pragma HLS stream variable = from_in_iter2_offset_16513_to_in_iter3_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_in_iter2_offset_16513_to_in_iter3_pe_2
  hls::stream<Data<float>> from_in_iter2_offset_16514_to_in_iter2_offset_32770("from_in_iter2_offset_16514_to_in_iter2_offset_32770");
#pragma HLS stream variable = from_in_iter2_offset_16514_to_in_iter2_offset_32770 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16514_to_in_iter2_offset_32770
  hls::stream<Data<float>> from_in_iter2_offset_16514_to_in_iter3_pe_1("from_in_iter2_offset_16514_to_in_iter3_pe_1");
#pragma HLS stream variable = from_in_iter2_offset_16514_to_in_iter3_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_in_iter2_offset_16514_to_in_iter3_pe_1
  hls::stream<Data<float>> from_in_iter1_offset_32771_to_in_iter2_pe_0("from_in_iter1_offset_32771_to_in_iter2_pe_0");
#pragma HLS stream variable = from_in_iter1_offset_32771_to_in_iter2_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_in_iter1_offset_32771_to_in_iter2_pe_0
  hls::stream<Data<float>> from_in_iter2_pe_0_to_in_iter2_offset_3("from_in_iter2_pe_0_to_in_iter2_offset_3");
#pragma HLS stream variable = from_in_iter2_pe_0_to_in_iter2_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_pe_0_to_in_iter2_offset_3
  hls::stream<Data<float>> from_in_iter2_offset_3_to_in_iter2_offset_16259("from_in_iter2_offset_3_to_in_iter2_offset_16259");
#pragma HLS stream variable = from_in_iter2_offset_3_to_in_iter2_offset_16259 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_3_to_in_iter2_offset_16259
  hls::stream<Data<float>> from_in_iter2_offset_3_to_in_iter3_pe_0("from_in_iter2_offset_3_to_in_iter3_pe_0");
#pragma HLS stream variable = from_in_iter2_offset_3_to_in_iter3_pe_0 depth = 13
#pragma HLS aggregate bit variable = from_in_iter2_offset_3_to_in_iter3_pe_0
  hls::stream<Data<float>> from_in_iter2_offset_16259_to_in_iter2_offset_16383("from_in_iter2_offset_16259_to_in_iter2_offset_16383");
#pragma HLS stream variable = from_in_iter2_offset_16259_to_in_iter2_offset_16383 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16259_to_in_iter2_offset_16383
  hls::stream<Data<float>> from_in_iter2_offset_16259_to_in_iter3_pe_0("from_in_iter2_offset_16259_to_in_iter3_pe_0");
#pragma HLS stream variable = from_in_iter2_offset_16259_to_in_iter3_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_in_iter2_offset_16259_to_in_iter3_pe_0
  hls::stream<Data<float>> from_in_iter2_offset_16383_to_in_iter2_offset_16387("from_in_iter2_offset_16383_to_in_iter2_offset_16387");
#pragma HLS stream variable = from_in_iter2_offset_16383_to_in_iter2_offset_16387 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16383_to_in_iter2_offset_16387
  hls::stream<Data<float>> from_in_iter2_offset_16383_to_in_iter3_pe_3("from_in_iter2_offset_16383_to_in_iter3_pe_3");
#pragma HLS stream variable = from_in_iter2_offset_16383_to_in_iter3_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_in_iter2_offset_16383_to_in_iter3_pe_3
  hls::stream<Data<float>> from_in_iter2_offset_16387_to_in_iter2_offset_16515("from_in_iter2_offset_16387_to_in_iter2_offset_16515");
#pragma HLS stream variable = from_in_iter2_offset_16387_to_in_iter2_offset_16515 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16387_to_in_iter2_offset_16515
  hls::stream<Data<float>> from_in_iter2_offset_16387_to_in_iter3_pe_1("from_in_iter2_offset_16387_to_in_iter3_pe_1");
#pragma HLS stream variable = from_in_iter2_offset_16387_to_in_iter3_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16387_to_in_iter3_pe_1
  hls::stream<Data<float>> from_in_iter2_offset_16387_to_in_iter3_pe_0("from_in_iter2_offset_16387_to_in_iter3_pe_0");
#pragma HLS stream variable = from_in_iter2_offset_16387_to_in_iter3_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_in_iter2_offset_16387_to_in_iter3_pe_0
  hls::stream<Data<float>> from_in_iter2_offset_16512_to_in_iter2_offset_32768("from_in_iter2_offset_16512_to_in_iter2_offset_32768");
#pragma HLS stream variable = from_in_iter2_offset_16512_to_in_iter2_offset_32768 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16512_to_in_iter2_offset_32768
  hls::stream<Data<float>> from_in_iter2_offset_16512_to_in_iter3_pe_3("from_in_iter2_offset_16512_to_in_iter3_pe_3");
#pragma HLS stream variable = from_in_iter2_offset_16512_to_in_iter3_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_in_iter2_offset_16512_to_in_iter3_pe_3
  hls::stream<Data<float>> from_in_iter2_offset_32769_to_in_iter3_pe_2("from_in_iter2_offset_32769_to_in_iter3_pe_2");
#pragma HLS stream variable = from_in_iter2_offset_32769_to_in_iter3_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_32769_to_in_iter3_pe_2
  hls::stream<Data<float>> from_in_iter3_pe_2_to_in_iter3_offset_1("from_in_iter3_pe_2_to_in_iter3_offset_1");
#pragma HLS stream variable = from_in_iter3_pe_2_to_in_iter3_offset_1 depth = 5
#pragma HLS aggregate bit variable = from_in_iter3_pe_2_to_in_iter3_offset_1
  hls::stream<Data<float>> from_in_iter3_offset_1_to_in_iter3_offset_16257("from_in_iter3_offset_1_to_in_iter3_offset_16257");
#pragma HLS stream variable = from_in_iter3_offset_1_to_in_iter3_offset_16257 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_1_to_in_iter3_offset_16257
  hls::stream<Data<float>> from_in_iter3_offset_1_to_in_iter4_pe_2("from_in_iter3_offset_1_to_in_iter4_pe_2");
#pragma HLS stream variable = from_in_iter3_offset_1_to_in_iter4_pe_2 depth = 11
#pragma HLS aggregate bit variable = from_in_iter3_offset_1_to_in_iter4_pe_2
  hls::stream<Data<float>> from_in_iter3_offset_16257_to_in_iter3_offset_16385("from_in_iter3_offset_16257_to_in_iter3_offset_16385");
#pragma HLS stream variable = from_in_iter3_offset_16257_to_in_iter3_offset_16385 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16257_to_in_iter3_offset_16385
  hls::stream<Data<float>> from_in_iter3_offset_16257_to_in_iter4_pe_2("from_in_iter3_offset_16257_to_in_iter4_pe_2");
#pragma HLS stream variable = from_in_iter3_offset_16257_to_in_iter4_pe_2 depth = 9
#pragma HLS aggregate bit variable = from_in_iter3_offset_16257_to_in_iter4_pe_2
  hls::stream<Data<float>> from_in_iter3_offset_16385_to_in_iter3_offset_16513("from_in_iter3_offset_16385_to_in_iter3_offset_16513");
#pragma HLS stream variable = from_in_iter3_offset_16385_to_in_iter3_offset_16513 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16385_to_in_iter3_offset_16513
  hls::stream<Data<float>> from_in_iter3_offset_16385_to_in_iter4_pe_3("from_in_iter3_offset_16385_to_in_iter4_pe_3");
#pragma HLS stream variable = from_in_iter3_offset_16385_to_in_iter4_pe_3 depth = 13
#pragma HLS aggregate bit variable = from_in_iter3_offset_16385_to_in_iter4_pe_3
  hls::stream<Data<float>> from_in_iter3_offset_16385_to_in_iter4_pe_2("from_in_iter3_offset_16385_to_in_iter4_pe_2");
#pragma HLS stream variable = from_in_iter3_offset_16385_to_in_iter4_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_in_iter3_offset_16385_to_in_iter4_pe_2
  hls::stream<Data<float>> from_in_iter3_offset_16385_to_in_iter4_pe_1("from_in_iter3_offset_16385_to_in_iter4_pe_1");
#pragma HLS stream variable = from_in_iter3_offset_16385_to_in_iter4_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_in_iter3_offset_16385_to_in_iter4_pe_1
  hls::stream<Data<float>> from_in_iter2_offset_32770_to_in_iter3_pe_1("from_in_iter2_offset_32770_to_in_iter3_pe_1");
#pragma HLS stream variable = from_in_iter2_offset_32770_to_in_iter3_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_32770_to_in_iter3_pe_1
  hls::stream<Data<float>> from_in_iter3_pe_1_to_in_iter3_offset_2("from_in_iter3_pe_1_to_in_iter3_offset_2");
#pragma HLS stream variable = from_in_iter3_pe_1_to_in_iter3_offset_2 depth = 5
#pragma HLS aggregate bit variable = from_in_iter3_pe_1_to_in_iter3_offset_2
  hls::stream<Data<float>> from_in_iter3_offset_2_to_in_iter3_offset_16258("from_in_iter3_offset_2_to_in_iter3_offset_16258");
#pragma HLS stream variable = from_in_iter3_offset_2_to_in_iter3_offset_16258 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_2_to_in_iter3_offset_16258
  hls::stream<Data<float>> from_in_iter3_offset_2_to_in_iter4_pe_1("from_in_iter3_offset_2_to_in_iter4_pe_1");
#pragma HLS stream variable = from_in_iter3_offset_2_to_in_iter4_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_in_iter3_offset_2_to_in_iter4_pe_1
  hls::stream<Data<float>> from_in_iter3_offset_16258_to_in_iter3_offset_16386("from_in_iter3_offset_16258_to_in_iter3_offset_16386");
#pragma HLS stream variable = from_in_iter3_offset_16258_to_in_iter3_offset_16386 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16258_to_in_iter3_offset_16386
  hls::stream<Data<float>> from_in_iter3_offset_16258_to_in_iter4_pe_1("from_in_iter3_offset_16258_to_in_iter4_pe_1");
#pragma HLS stream variable = from_in_iter3_offset_16258_to_in_iter4_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_in_iter3_offset_16258_to_in_iter4_pe_1
  hls::stream<Data<float>> from_in_iter3_offset_16386_to_in_iter3_offset_16514("from_in_iter3_offset_16386_to_in_iter3_offset_16514");
#pragma HLS stream variable = from_in_iter3_offset_16386_to_in_iter3_offset_16514 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16386_to_in_iter3_offset_16514
  hls::stream<Data<float>> from_in_iter3_offset_16386_to_in_iter4_pe_2("from_in_iter3_offset_16386_to_in_iter4_pe_2");
#pragma HLS stream variable = from_in_iter3_offset_16386_to_in_iter4_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_in_iter3_offset_16386_to_in_iter4_pe_2
  hls::stream<Data<float>> from_in_iter3_offset_16386_to_in_iter4_pe_1("from_in_iter3_offset_16386_to_in_iter4_pe_1");
#pragma HLS stream variable = from_in_iter3_offset_16386_to_in_iter4_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_in_iter3_offset_16386_to_in_iter4_pe_1
  hls::stream<Data<float>> from_in_iter3_offset_16386_to_in_iter4_pe_0("from_in_iter3_offset_16386_to_in_iter4_pe_0");
#pragma HLS stream variable = from_in_iter3_offset_16386_to_in_iter4_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_in_iter3_offset_16386_to_in_iter4_pe_0
  hls::stream<Data<float>> from_in_iter2_offset_16515_to_in_iter2_offset_32771("from_in_iter2_offset_16515_to_in_iter2_offset_32771");
#pragma HLS stream variable = from_in_iter2_offset_16515_to_in_iter2_offset_32771 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_16515_to_in_iter2_offset_32771
  hls::stream<Data<float>> from_in_iter2_offset_16515_to_in_iter3_pe_0("from_in_iter2_offset_16515_to_in_iter3_pe_0");
#pragma HLS stream variable = from_in_iter2_offset_16515_to_in_iter3_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_in_iter2_offset_16515_to_in_iter3_pe_0
  hls::stream<Data<float>> from_in_iter2_offset_32768_to_in_iter3_pe_3("from_in_iter2_offset_32768_to_in_iter3_pe_3");
#pragma HLS stream variable = from_in_iter2_offset_32768_to_in_iter3_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_32768_to_in_iter3_pe_3
  hls::stream<Data<float>> from_in_iter3_pe_3_to_in_iter3_offset_0("from_in_iter3_pe_3_to_in_iter3_offset_0");
#pragma HLS stream variable = from_in_iter3_pe_3_to_in_iter3_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_pe_3_to_in_iter3_offset_0
  hls::stream<Data<float>> from_in_iter3_offset_0_to_in_iter3_offset_16256("from_in_iter3_offset_0_to_in_iter3_offset_16256");
#pragma HLS stream variable = from_in_iter3_offset_0_to_in_iter3_offset_16256 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_0_to_in_iter3_offset_16256
  hls::stream<Data<float>> from_in_iter3_offset_0_to_in_iter4_pe_3("from_in_iter3_offset_0_to_in_iter4_pe_3");
#pragma HLS stream variable = from_in_iter3_offset_0_to_in_iter4_pe_3 depth = 13
#pragma HLS aggregate bit variable = from_in_iter3_offset_0_to_in_iter4_pe_3
  hls::stream<Data<float>> from_in_iter3_offset_16256_to_in_iter3_offset_16384("from_in_iter3_offset_16256_to_in_iter3_offset_16384");
#pragma HLS stream variable = from_in_iter3_offset_16256_to_in_iter3_offset_16384 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16256_to_in_iter3_offset_16384
  hls::stream<Data<float>> from_in_iter3_offset_16256_to_in_iter4_pe_3("from_in_iter3_offset_16256_to_in_iter4_pe_3");
#pragma HLS stream variable = from_in_iter3_offset_16256_to_in_iter4_pe_3 depth = 11
#pragma HLS aggregate bit variable = from_in_iter3_offset_16256_to_in_iter4_pe_3
  hls::stream<Data<float>> from_in_iter3_offset_16384_to_in_iter3_offset_16388("from_in_iter3_offset_16384_to_in_iter3_offset_16388");
#pragma HLS stream variable = from_in_iter3_offset_16384_to_in_iter3_offset_16388 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16384_to_in_iter3_offset_16388
  hls::stream<Data<float>> from_in_iter3_offset_16384_to_in_iter4_pe_3("from_in_iter3_offset_16384_to_in_iter4_pe_3");
#pragma HLS stream variable = from_in_iter3_offset_16384_to_in_iter4_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_in_iter3_offset_16384_to_in_iter4_pe_3
  hls::stream<Data<float>> from_in_iter3_offset_16384_to_in_iter4_pe_2("from_in_iter3_offset_16384_to_in_iter4_pe_2");
#pragma HLS stream variable = from_in_iter3_offset_16384_to_in_iter4_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16384_to_in_iter4_pe_2
  hls::stream<Data<float>> from_in_iter3_offset_16388_to_in_iter3_offset_16512("from_in_iter3_offset_16388_to_in_iter3_offset_16512");
#pragma HLS stream variable = from_in_iter3_offset_16388_to_in_iter3_offset_16512 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16388_to_in_iter3_offset_16512
  hls::stream<Data<float>> from_in_iter3_offset_16388_to_in_iter4_pe_0("from_in_iter3_offset_16388_to_in_iter4_pe_0");
#pragma HLS stream variable = from_in_iter3_offset_16388_to_in_iter4_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_in_iter3_offset_16388_to_in_iter4_pe_0
  hls::stream<Data<float>> from_in_iter3_offset_16513_to_in_iter3_offset_32769("from_in_iter3_offset_16513_to_in_iter3_offset_32769");
#pragma HLS stream variable = from_in_iter3_offset_16513_to_in_iter3_offset_32769 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16513_to_in_iter3_offset_32769
  hls::stream<Data<float>> from_in_iter3_offset_16513_to_in_iter4_pe_2("from_in_iter3_offset_16513_to_in_iter4_pe_2");
#pragma HLS stream variable = from_in_iter3_offset_16513_to_in_iter4_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_in_iter3_offset_16513_to_in_iter4_pe_2
  hls::stream<Data<float>> from_in_iter3_offset_16514_to_in_iter3_offset_32770("from_in_iter3_offset_16514_to_in_iter3_offset_32770");
#pragma HLS stream variable = from_in_iter3_offset_16514_to_in_iter3_offset_32770 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16514_to_in_iter3_offset_32770
  hls::stream<Data<float>> from_in_iter3_offset_16514_to_in_iter4_pe_1("from_in_iter3_offset_16514_to_in_iter4_pe_1");
#pragma HLS stream variable = from_in_iter3_offset_16514_to_in_iter4_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_in_iter3_offset_16514_to_in_iter4_pe_1
  hls::stream<Data<float>> from_in_iter2_offset_32771_to_in_iter3_pe_0("from_in_iter2_offset_32771_to_in_iter3_pe_0");
#pragma HLS stream variable = from_in_iter2_offset_32771_to_in_iter3_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_in_iter2_offset_32771_to_in_iter3_pe_0
  hls::stream<Data<float>> from_in_iter3_pe_0_to_in_iter3_offset_3("from_in_iter3_pe_0_to_in_iter3_offset_3");
#pragma HLS stream variable = from_in_iter3_pe_0_to_in_iter3_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_pe_0_to_in_iter3_offset_3
  hls::stream<Data<float>> from_in_iter3_offset_3_to_in_iter3_offset_16259("from_in_iter3_offset_3_to_in_iter3_offset_16259");
#pragma HLS stream variable = from_in_iter3_offset_3_to_in_iter3_offset_16259 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_3_to_in_iter3_offset_16259
  hls::stream<Data<float>> from_in_iter3_offset_3_to_in_iter4_pe_0("from_in_iter3_offset_3_to_in_iter4_pe_0");
#pragma HLS stream variable = from_in_iter3_offset_3_to_in_iter4_pe_0 depth = 13
#pragma HLS aggregate bit variable = from_in_iter3_offset_3_to_in_iter4_pe_0
  hls::stream<Data<float>> from_in_iter3_offset_16259_to_in_iter3_offset_16383("from_in_iter3_offset_16259_to_in_iter3_offset_16383");
#pragma HLS stream variable = from_in_iter3_offset_16259_to_in_iter3_offset_16383 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16259_to_in_iter3_offset_16383
  hls::stream<Data<float>> from_in_iter3_offset_16259_to_in_iter4_pe_0("from_in_iter3_offset_16259_to_in_iter4_pe_0");
#pragma HLS stream variable = from_in_iter3_offset_16259_to_in_iter4_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_in_iter3_offset_16259_to_in_iter4_pe_0
  hls::stream<Data<float>> from_in_iter3_offset_16383_to_in_iter3_offset_16387("from_in_iter3_offset_16383_to_in_iter3_offset_16387");
#pragma HLS stream variable = from_in_iter3_offset_16383_to_in_iter3_offset_16387 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16383_to_in_iter3_offset_16387
  hls::stream<Data<float>> from_in_iter3_offset_16383_to_in_iter4_pe_3("from_in_iter3_offset_16383_to_in_iter4_pe_3");
#pragma HLS stream variable = from_in_iter3_offset_16383_to_in_iter4_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_in_iter3_offset_16383_to_in_iter4_pe_3
  hls::stream<Data<float>> from_in_iter3_offset_16387_to_in_iter3_offset_16515("from_in_iter3_offset_16387_to_in_iter3_offset_16515");
#pragma HLS stream variable = from_in_iter3_offset_16387_to_in_iter3_offset_16515 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16387_to_in_iter3_offset_16515
  hls::stream<Data<float>> from_in_iter3_offset_16387_to_in_iter4_pe_1("from_in_iter3_offset_16387_to_in_iter4_pe_1");
#pragma HLS stream variable = from_in_iter3_offset_16387_to_in_iter4_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16387_to_in_iter4_pe_1
  hls::stream<Data<float>> from_in_iter3_offset_16387_to_in_iter4_pe_0("from_in_iter3_offset_16387_to_in_iter4_pe_0");
#pragma HLS stream variable = from_in_iter3_offset_16387_to_in_iter4_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_in_iter3_offset_16387_to_in_iter4_pe_0
  hls::stream<Data<float>> from_in_iter3_offset_16512_to_in_iter3_offset_32768("from_in_iter3_offset_16512_to_in_iter3_offset_32768");
#pragma HLS stream variable = from_in_iter3_offset_16512_to_in_iter3_offset_32768 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16512_to_in_iter3_offset_32768
  hls::stream<Data<float>> from_in_iter3_offset_16512_to_in_iter4_pe_3("from_in_iter3_offset_16512_to_in_iter4_pe_3");
#pragma HLS stream variable = from_in_iter3_offset_16512_to_in_iter4_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_in_iter3_offset_16512_to_in_iter4_pe_3
  hls::stream<Data<float>> from_in_iter3_offset_32769_to_in_iter4_pe_2("from_in_iter3_offset_32769_to_in_iter4_pe_2");
#pragma HLS stream variable = from_in_iter3_offset_32769_to_in_iter4_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_32769_to_in_iter4_pe_2
  hls::stream<Data<float>> from_in_iter4_pe_2_to_in_iter4_offset_1("from_in_iter4_pe_2_to_in_iter4_offset_1");
#pragma HLS stream variable = from_in_iter4_pe_2_to_in_iter4_offset_1 depth = 5
#pragma HLS aggregate bit variable = from_in_iter4_pe_2_to_in_iter4_offset_1
  hls::stream<Data<float>> from_in_iter4_offset_1_to_in_iter4_offset_16257("from_in_iter4_offset_1_to_in_iter4_offset_16257");
#pragma HLS stream variable = from_in_iter4_offset_1_to_in_iter4_offset_16257 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_1_to_in_iter4_offset_16257
  hls::stream<Data<float>> from_in_iter4_offset_1_to_out_pe_2("from_in_iter4_offset_1_to_out_pe_2");
#pragma HLS stream variable = from_in_iter4_offset_1_to_out_pe_2 depth = 11
#pragma HLS aggregate bit variable = from_in_iter4_offset_1_to_out_pe_2
  hls::stream<Data<float>> from_in_iter4_offset_16257_to_in_iter4_offset_16385("from_in_iter4_offset_16257_to_in_iter4_offset_16385");
#pragma HLS stream variable = from_in_iter4_offset_16257_to_in_iter4_offset_16385 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16257_to_in_iter4_offset_16385
  hls::stream<Data<float>> from_in_iter4_offset_16257_to_out_pe_2("from_in_iter4_offset_16257_to_out_pe_2");
#pragma HLS stream variable = from_in_iter4_offset_16257_to_out_pe_2 depth = 9
#pragma HLS aggregate bit variable = from_in_iter4_offset_16257_to_out_pe_2
  hls::stream<Data<float>> from_in_iter4_offset_16385_to_in_iter4_offset_16513("from_in_iter4_offset_16385_to_in_iter4_offset_16513");
#pragma HLS stream variable = from_in_iter4_offset_16385_to_in_iter4_offset_16513 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16385_to_in_iter4_offset_16513
  hls::stream<Data<float>> from_in_iter4_offset_16385_to_out_pe_3("from_in_iter4_offset_16385_to_out_pe_3");
#pragma HLS stream variable = from_in_iter4_offset_16385_to_out_pe_3 depth = 13
#pragma HLS aggregate bit variable = from_in_iter4_offset_16385_to_out_pe_3
  hls::stream<Data<float>> from_in_iter4_offset_16385_to_out_pe_2("from_in_iter4_offset_16385_to_out_pe_2");
#pragma HLS stream variable = from_in_iter4_offset_16385_to_out_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_in_iter4_offset_16385_to_out_pe_2
  hls::stream<Data<float>> from_in_iter4_offset_16385_to_out_pe_1("from_in_iter4_offset_16385_to_out_pe_1");
#pragma HLS stream variable = from_in_iter4_offset_16385_to_out_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_in_iter4_offset_16385_to_out_pe_1
  hls::stream<Data<float>> from_in_iter3_offset_32770_to_in_iter4_pe_1("from_in_iter3_offset_32770_to_in_iter4_pe_1");
#pragma HLS stream variable = from_in_iter3_offset_32770_to_in_iter4_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_32770_to_in_iter4_pe_1
  hls::stream<Data<float>> from_in_iter4_pe_1_to_in_iter4_offset_2("from_in_iter4_pe_1_to_in_iter4_offset_2");
#pragma HLS stream variable = from_in_iter4_pe_1_to_in_iter4_offset_2 depth = 5
#pragma HLS aggregate bit variable = from_in_iter4_pe_1_to_in_iter4_offset_2
  hls::stream<Data<float>> from_in_iter4_offset_2_to_in_iter4_offset_16258("from_in_iter4_offset_2_to_in_iter4_offset_16258");
#pragma HLS stream variable = from_in_iter4_offset_2_to_in_iter4_offset_16258 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_2_to_in_iter4_offset_16258
  hls::stream<Data<float>> from_in_iter4_offset_2_to_out_pe_1("from_in_iter4_offset_2_to_out_pe_1");
#pragma HLS stream variable = from_in_iter4_offset_2_to_out_pe_1 depth = 11
#pragma HLS aggregate bit variable = from_in_iter4_offset_2_to_out_pe_1
  hls::stream<Data<float>> from_in_iter4_offset_16258_to_in_iter4_offset_16386("from_in_iter4_offset_16258_to_in_iter4_offset_16386");
#pragma HLS stream variable = from_in_iter4_offset_16258_to_in_iter4_offset_16386 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16258_to_in_iter4_offset_16386
  hls::stream<Data<float>> from_in_iter4_offset_16258_to_out_pe_1("from_in_iter4_offset_16258_to_out_pe_1");
#pragma HLS stream variable = from_in_iter4_offset_16258_to_out_pe_1 depth = 9
#pragma HLS aggregate bit variable = from_in_iter4_offset_16258_to_out_pe_1
  hls::stream<Data<float>> from_in_iter4_offset_16386_to_in_iter4_offset_16514("from_in_iter4_offset_16386_to_in_iter4_offset_16514");
#pragma HLS stream variable = from_in_iter4_offset_16386_to_in_iter4_offset_16514 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16386_to_in_iter4_offset_16514
  hls::stream<Data<float>> from_in_iter4_offset_16386_to_out_pe_2("from_in_iter4_offset_16386_to_out_pe_2");
#pragma HLS stream variable = from_in_iter4_offset_16386_to_out_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_in_iter4_offset_16386_to_out_pe_2
  hls::stream<Data<float>> from_in_iter4_offset_16386_to_out_pe_1("from_in_iter4_offset_16386_to_out_pe_1");
#pragma HLS stream variable = from_in_iter4_offset_16386_to_out_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_in_iter4_offset_16386_to_out_pe_1
  hls::stream<Data<float>> from_in_iter4_offset_16386_to_out_pe_0("from_in_iter4_offset_16386_to_out_pe_0");
#pragma HLS stream variable = from_in_iter4_offset_16386_to_out_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_in_iter4_offset_16386_to_out_pe_0
  hls::stream<Data<float>> from_in_iter3_offset_16515_to_in_iter3_offset_32771("from_in_iter3_offset_16515_to_in_iter3_offset_32771");
#pragma HLS stream variable = from_in_iter3_offset_16515_to_in_iter3_offset_32771 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_16515_to_in_iter3_offset_32771
  hls::stream<Data<float>> from_in_iter3_offset_16515_to_in_iter4_pe_0("from_in_iter3_offset_16515_to_in_iter4_pe_0");
#pragma HLS stream variable = from_in_iter3_offset_16515_to_in_iter4_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_in_iter3_offset_16515_to_in_iter4_pe_0
  hls::stream<Data<float>> from_in_iter3_offset_32768_to_in_iter4_pe_3("from_in_iter3_offset_32768_to_in_iter4_pe_3");
#pragma HLS stream variable = from_in_iter3_offset_32768_to_in_iter4_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_32768_to_in_iter4_pe_3
  hls::stream<Data<float>> from_in_iter4_pe_3_to_in_iter4_offset_0("from_in_iter4_pe_3_to_in_iter4_offset_0");
#pragma HLS stream variable = from_in_iter4_pe_3_to_in_iter4_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_pe_3_to_in_iter4_offset_0
  hls::stream<Data<float>> from_in_iter4_offset_0_to_in_iter4_offset_16256("from_in_iter4_offset_0_to_in_iter4_offset_16256");
#pragma HLS stream variable = from_in_iter4_offset_0_to_in_iter4_offset_16256 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_0_to_in_iter4_offset_16256
  hls::stream<Data<float>> from_in_iter4_offset_0_to_out_pe_3("from_in_iter4_offset_0_to_out_pe_3");
#pragma HLS stream variable = from_in_iter4_offset_0_to_out_pe_3 depth = 13
#pragma HLS aggregate bit variable = from_in_iter4_offset_0_to_out_pe_3
  hls::stream<Data<float>> from_in_iter4_offset_16256_to_in_iter4_offset_16384("from_in_iter4_offset_16256_to_in_iter4_offset_16384");
#pragma HLS stream variable = from_in_iter4_offset_16256_to_in_iter4_offset_16384 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16256_to_in_iter4_offset_16384
  hls::stream<Data<float>> from_in_iter4_offset_16256_to_out_pe_3("from_in_iter4_offset_16256_to_out_pe_3");
#pragma HLS stream variable = from_in_iter4_offset_16256_to_out_pe_3 depth = 11
#pragma HLS aggregate bit variable = from_in_iter4_offset_16256_to_out_pe_3
  hls::stream<Data<float>> from_in_iter4_offset_16384_to_in_iter4_offset_16388("from_in_iter4_offset_16384_to_in_iter4_offset_16388");
#pragma HLS stream variable = from_in_iter4_offset_16384_to_in_iter4_offset_16388 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16384_to_in_iter4_offset_16388
  hls::stream<Data<float>> from_in_iter4_offset_16384_to_out_pe_3("from_in_iter4_offset_16384_to_out_pe_3");
#pragma HLS stream variable = from_in_iter4_offset_16384_to_out_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_in_iter4_offset_16384_to_out_pe_3
  hls::stream<Data<float>> from_in_iter4_offset_16384_to_out_pe_2("from_in_iter4_offset_16384_to_out_pe_2");
#pragma HLS stream variable = from_in_iter4_offset_16384_to_out_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16384_to_out_pe_2
  hls::stream<Data<float>> from_in_iter4_offset_16388_to_in_iter4_offset_16512("from_in_iter4_offset_16388_to_in_iter4_offset_16512");
#pragma HLS stream variable = from_in_iter4_offset_16388_to_in_iter4_offset_16512 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16388_to_in_iter4_offset_16512
  hls::stream<Data<float>> from_in_iter4_offset_16388_to_out_pe_0("from_in_iter4_offset_16388_to_out_pe_0");
#pragma HLS stream variable = from_in_iter4_offset_16388_to_out_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_in_iter4_offset_16388_to_out_pe_0
  hls::stream<Data<float>> from_in_iter4_offset_16513_to_in_iter4_offset_32769("from_in_iter4_offset_16513_to_in_iter4_offset_32769");
#pragma HLS stream variable = from_in_iter4_offset_16513_to_in_iter4_offset_32769 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16513_to_in_iter4_offset_32769
  hls::stream<Data<float>> from_in_iter4_offset_16513_to_out_pe_2("from_in_iter4_offset_16513_to_out_pe_2");
#pragma HLS stream variable = from_in_iter4_offset_16513_to_out_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_in_iter4_offset_16513_to_out_pe_2
  hls::stream<Data<float>> from_in_iter4_offset_16514_to_in_iter4_offset_32770("from_in_iter4_offset_16514_to_in_iter4_offset_32770");
#pragma HLS stream variable = from_in_iter4_offset_16514_to_in_iter4_offset_32770 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16514_to_in_iter4_offset_32770
  hls::stream<Data<float>> from_in_iter4_offset_16514_to_out_pe_1("from_in_iter4_offset_16514_to_out_pe_1");
#pragma HLS stream variable = from_in_iter4_offset_16514_to_out_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_in_iter4_offset_16514_to_out_pe_1
  hls::stream<Data<float>> from_in_iter3_offset_32771_to_in_iter4_pe_0("from_in_iter3_offset_32771_to_in_iter4_pe_0");
#pragma HLS stream variable = from_in_iter3_offset_32771_to_in_iter4_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_in_iter3_offset_32771_to_in_iter4_pe_0
  hls::stream<Data<float>> from_in_iter4_pe_0_to_in_iter4_offset_3("from_in_iter4_pe_0_to_in_iter4_offset_3");
#pragma HLS stream variable = from_in_iter4_pe_0_to_in_iter4_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_pe_0_to_in_iter4_offset_3
  hls::stream<Data<float>> from_in_iter4_offset_3_to_in_iter4_offset_16259("from_in_iter4_offset_3_to_in_iter4_offset_16259");
#pragma HLS stream variable = from_in_iter4_offset_3_to_in_iter4_offset_16259 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_3_to_in_iter4_offset_16259
  hls::stream<Data<float>> from_in_iter4_offset_3_to_out_pe_0("from_in_iter4_offset_3_to_out_pe_0");
#pragma HLS stream variable = from_in_iter4_offset_3_to_out_pe_0 depth = 13
#pragma HLS aggregate bit variable = from_in_iter4_offset_3_to_out_pe_0
  hls::stream<Data<float>> from_in_iter4_offset_16259_to_in_iter4_offset_16383("from_in_iter4_offset_16259_to_in_iter4_offset_16383");
#pragma HLS stream variable = from_in_iter4_offset_16259_to_in_iter4_offset_16383 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16259_to_in_iter4_offset_16383
  hls::stream<Data<float>> from_in_iter4_offset_16259_to_out_pe_0("from_in_iter4_offset_16259_to_out_pe_0");
#pragma HLS stream variable = from_in_iter4_offset_16259_to_out_pe_0 depth = 11
#pragma HLS aggregate bit variable = from_in_iter4_offset_16259_to_out_pe_0
  hls::stream<Data<float>> from_in_iter4_offset_16383_to_in_iter4_offset_16387("from_in_iter4_offset_16383_to_in_iter4_offset_16387");
#pragma HLS stream variable = from_in_iter4_offset_16383_to_in_iter4_offset_16387 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16383_to_in_iter4_offset_16387
  hls::stream<Data<float>> from_in_iter4_offset_16383_to_out_pe_3("from_in_iter4_offset_16383_to_out_pe_3");
#pragma HLS stream variable = from_in_iter4_offset_16383_to_out_pe_3 depth = 9
#pragma HLS aggregate bit variable = from_in_iter4_offset_16383_to_out_pe_3
  hls::stream<Data<float>> from_in_iter4_offset_16387_to_in_iter4_offset_16515("from_in_iter4_offset_16387_to_in_iter4_offset_16515");
#pragma HLS stream variable = from_in_iter4_offset_16387_to_in_iter4_offset_16515 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16387_to_in_iter4_offset_16515
  hls::stream<Data<float>> from_in_iter4_offset_16387_to_out_pe_1("from_in_iter4_offset_16387_to_out_pe_1");
#pragma HLS stream variable = from_in_iter4_offset_16387_to_out_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16387_to_out_pe_1
  hls::stream<Data<float>> from_in_iter4_offset_16387_to_out_pe_0("from_in_iter4_offset_16387_to_out_pe_0");
#pragma HLS stream variable = from_in_iter4_offset_16387_to_out_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_in_iter4_offset_16387_to_out_pe_0
  hls::stream<Data<float>> from_in_iter4_offset_16512_to_in_iter4_offset_32768("from_in_iter4_offset_16512_to_in_iter4_offset_32768");
#pragma HLS stream variable = from_in_iter4_offset_16512_to_in_iter4_offset_32768 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16512_to_in_iter4_offset_32768
  hls::stream<Data<float>> from_in_iter4_offset_16512_to_out_pe_3("from_in_iter4_offset_16512_to_out_pe_3");
#pragma HLS stream variable = from_in_iter4_offset_16512_to_out_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_in_iter4_offset_16512_to_out_pe_3
  hls::stream<Data<float>> from_in_iter4_offset_32769_to_out_pe_2("from_in_iter4_offset_32769_to_out_pe_2");
#pragma HLS stream variable = from_in_iter4_offset_32769_to_out_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_32769_to_out_pe_2
  hls::stream<Data<float>> from_out_pe_2_to_out_bank_1("from_out_pe_2_to_out_bank_1");
#pragma HLS stream variable = from_out_pe_2_to_out_bank_1 depth = 9
#pragma HLS aggregate bit variable = from_out_pe_2_to_out_bank_1
  hls::stream<Data<float>> from_in_iter4_offset_32770_to_out_pe_1("from_in_iter4_offset_32770_to_out_pe_1");
#pragma HLS stream variable = from_in_iter4_offset_32770_to_out_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_32770_to_out_pe_1
  hls::stream<Data<float>> from_out_pe_1_to_out_bank_1("from_out_pe_1_to_out_bank_1");
#pragma HLS stream variable = from_out_pe_1_to_out_bank_1 depth = 7
#pragma HLS aggregate bit variable = from_out_pe_1_to_out_bank_1
  hls::stream<Data<float>> from_in_iter4_offset_16515_to_in_iter4_offset_32771("from_in_iter4_offset_16515_to_in_iter4_offset_32771");
#pragma HLS stream variable = from_in_iter4_offset_16515_to_in_iter4_offset_32771 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_16515_to_in_iter4_offset_32771
  hls::stream<Data<float>> from_in_iter4_offset_16515_to_out_pe_0("from_in_iter4_offset_16515_to_out_pe_0");
#pragma HLS stream variable = from_in_iter4_offset_16515_to_out_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_in_iter4_offset_16515_to_out_pe_0
  hls::stream<Data<float>> from_in_iter4_offset_32768_to_out_pe_3("from_in_iter4_offset_32768_to_out_pe_3");
#pragma HLS stream variable = from_in_iter4_offset_32768_to_out_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_32768_to_out_pe_3
  hls::stream<Data<float>> from_out_pe_3_to_out_bank_1("from_out_pe_3_to_out_bank_1");
#pragma HLS stream variable = from_out_pe_3_to_out_bank_1 depth = 3
#pragma HLS aggregate bit variable = from_out_pe_3_to_out_bank_1
  hls::stream<Data<float>> from_in_iter4_offset_32771_to_out_pe_0("from_in_iter4_offset_32771_to_out_pe_0");
#pragma HLS stream variable = from_in_iter4_offset_32771_to_out_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_in_iter4_offset_32771_to_out_pe_0
  hls::stream<Data<float>> from_out_pe_0_to_out_bank_1("from_out_pe_0_to_out_bank_1");
#pragma HLS stream variable = from_out_pe_0_to_out_bank_1 depth = 3
#pragma HLS aggregate bit variable = from_out_pe_0_to_out_bank_1

#pragma HLS dataflow
  BurstRead(bank_0_in_buf, bank_0_in, coalesced_data_num);
  Module0Func(
      /*output*/ from_in_bank_0_to_in_offset_0, 
      /*output*/ from_in_bank_0_to_in_offset_1, 
      /*output*/ from_in_bank_0_to_in_offset_2, 
      /*output*/ from_in_bank_0_to_in_offset_3, 
      /* input*/ bank_0_in_buf);
  Module1Func(
      /*output*/ from_in_offset_0_to_in_offset_16256, 
      /*output*/ from_in_offset_0_to_in_iter1_pe_3, 
      /* input*/ from_in_bank_0_to_in_offset_0);
  Module1Func(
      /*output*/ from_in_offset_1_to_in_offset_16257, 
      /*output*/ from_in_offset_1_to_in_iter1_pe_2, 
      /* input*/ from_in_bank_0_to_in_offset_1);
  Module1Func(
      /*output*/ from_in_offset_2_to_in_offset_16258, 
      /*output*/ from_in_offset_2_to_in_iter1_pe_1, 
      /* input*/ from_in_bank_0_to_in_offset_2);
  Module1Func(
      /*output*/ from_in_offset_3_to_in_offset_16259, 
      /*output*/ from_in_offset_3_to_in_iter1_pe_0, 
      /* input*/ from_in_bank_0_to_in_offset_3);
  Module2Func(
      /*output*/ from_in_offset_16256_to_in_offset_16384, 
      /*output*/ from_in_offset_16256_to_in_iter1_pe_3, 
      /* input*/ from_in_offset_0_to_in_offset_16256);
  Module2Func(
      /*output*/ from_in_offset_16257_to_in_offset_16385, 
      /*output*/ from_in_offset_16257_to_in_iter1_pe_2, 
      /* input*/ from_in_offset_1_to_in_offset_16257);
  Module2Func(
      /*output*/ from_in_offset_16258_to_in_offset_16386, 
      /*output*/ from_in_offset_16258_to_in_iter1_pe_1, 
      /* input*/ from_in_offset_2_to_in_offset_16258);
  Module2Func(
      /*output*/ from_in_offset_16259_to_in_offset_16383, 
      /*output*/ from_in_offset_16259_to_in_iter1_pe_0, 
      /* input*/ from_in_offset_3_to_in_offset_16259);
  Module3Func(
      /*output*/ from_in_offset_16384_to_in_offset_16388, 
      /*output*/ from_in_offset_16384_to_in_iter1_pe_3, 
      /*output*/ from_in_offset_16384_to_in_iter1_pe_2, 
      /* input*/ from_in_offset_16256_to_in_offset_16384);
  Module4Func(
      /*output*/ from_in_offset_16385_to_in_offset_16513, 
      /*output*/ from_in_offset_16385_to_in_iter1_pe_3, 
      /*output*/ from_in_offset_16385_to_in_iter1_pe_2, 
      /*output*/ from_in_offset_16385_to_in_iter1_pe_1, 
      /* input*/ from_in_offset_16257_to_in_offset_16385);
  Module4Func(
      /*output*/ from_in_offset_16386_to_in_offset_16514, 
      /*output*/ from_in_offset_16386_to_in_iter1_pe_2, 
      /*output*/ from_in_offset_16386_to_in_iter1_pe_1, 
      /*output*/ from_in_offset_16386_to_in_iter1_pe_0, 
      /* input*/ from_in_offset_16258_to_in_offset_16386);
  Module5Func(
      /*output*/ from_in_offset_16383_to_in_offset_16387, 
      /*output*/ from_in_offset_16383_to_in_iter1_pe_3, 
      /* input*/ from_in_offset_16259_to_in_offset_16383);
  Module6Func(
      /*output*/ from_in_offset_16388_to_in_offset_16512, 
      /*output*/ from_in_offset_16388_to_in_iter1_pe_0, 
      /* input*/ from_in_offset_16384_to_in_offset_16388);
  Module7Func(
      /*output*/ from_in_offset_16513_to_in_offset_32769, 
      /*output*/ from_in_offset_16513_to_in_iter1_pe_2, 
      /* input*/ from_in_offset_16385_to_in_offset_16513);
  Module7Func(
      /*output*/ from_in_offset_16514_to_in_offset_32770, 
      /*output*/ from_in_offset_16514_to_in_iter1_pe_1, 
      /* input*/ from_in_offset_16386_to_in_offset_16514);
  Module8Func(
      /*output*/ from_in_offset_16387_to_in_offset_16515, 
      /*output*/ from_in_offset_16387_to_in_iter1_pe_1, 
      /*output*/ from_in_offset_16387_to_in_iter1_pe_0, 
      /* input*/ from_in_offset_16383_to_in_offset_16387);
  Module5Func(
      /*output*/ from_in_offset_16512_to_in_offset_32768, 
      /*output*/ from_in_offset_16512_to_in_iter1_pe_3, 
      /* input*/ from_in_offset_16388_to_in_offset_16512);
  Module9Func(
      /*output*/ from_in_offset_32769_to_in_iter1_pe_2, 
      /* input*/ from_in_offset_16513_to_in_offset_32769);
  Module10Func(
      /*output*/ from_in_iter1_pe_2_to_in_iter1_offset_1, 
      /* input*/ from_in_offset_16384_to_in_iter1_pe_2, 
      /* input*/ from_in_offset_16386_to_in_iter1_pe_2, 
      /* input*/ from_in_offset_16257_to_in_iter1_pe_2, 
      /* input*/ from_in_offset_16513_to_in_iter1_pe_2, 
      /* input*/ from_in_offset_1_to_in_iter1_pe_2, 
      /* input*/ from_in_offset_32769_to_in_iter1_pe_2, 
      /* input*/ from_in_offset_16385_to_in_iter1_pe_2);
  Module1Func(
      /*output*/ from_in_iter1_offset_1_to_in_iter1_offset_16257, 
      /*output*/ from_in_iter1_offset_1_to_in_iter2_pe_2, 
      /* input*/ from_in_iter1_pe_2_to_in_iter1_offset_1);
  Module2Func(
      /*output*/ from_in_iter1_offset_16257_to_in_iter1_offset_16385, 
      /*output*/ from_in_iter1_offset_16257_to_in_iter2_pe_2, 
      /* input*/ from_in_iter1_offset_1_to_in_iter1_offset_16257);
  Module4Func(
      /*output*/ from_in_iter1_offset_16385_to_in_iter1_offset_16513, 
      /*output*/ from_in_iter1_offset_16385_to_in_iter2_pe_3, 
      /*output*/ from_in_iter1_offset_16385_to_in_iter2_pe_2, 
      /*output*/ from_in_iter1_offset_16385_to_in_iter2_pe_1, 
      /* input*/ from_in_iter1_offset_16257_to_in_iter1_offset_16385);
  Module9Func(
      /*output*/ from_in_offset_32770_to_in_iter1_pe_1, 
      /* input*/ from_in_offset_16514_to_in_offset_32770);
  Module10Func(
      /*output*/ from_in_iter1_pe_1_to_in_iter1_offset_2, 
      /* input*/ from_in_offset_16385_to_in_iter1_pe_1, 
      /* input*/ from_in_offset_16387_to_in_iter1_pe_1, 
      /* input*/ from_in_offset_16258_to_in_iter1_pe_1, 
      /* input*/ from_in_offset_16514_to_in_iter1_pe_1, 
      /* input*/ from_in_offset_2_to_in_iter1_pe_1, 
      /* input*/ from_in_offset_32770_to_in_iter1_pe_1, 
      /* input*/ from_in_offset_16386_to_in_iter1_pe_1);
  Module1Func(
      /*output*/ from_in_iter1_offset_2_to_in_iter1_offset_16258, 
      /*output*/ from_in_iter1_offset_2_to_in_iter2_pe_1, 
      /* input*/ from_in_iter1_pe_1_to_in_iter1_offset_2);
  Module2Func(
      /*output*/ from_in_iter1_offset_16258_to_in_iter1_offset_16386, 
      /*output*/ from_in_iter1_offset_16258_to_in_iter2_pe_1, 
      /* input*/ from_in_iter1_offset_2_to_in_iter1_offset_16258);
  Module4Func(
      /*output*/ from_in_iter1_offset_16386_to_in_iter1_offset_16514, 
      /*output*/ from_in_iter1_offset_16386_to_in_iter2_pe_2, 
      /*output*/ from_in_iter1_offset_16386_to_in_iter2_pe_1, 
      /*output*/ from_in_iter1_offset_16386_to_in_iter2_pe_0, 
      /* input*/ from_in_iter1_offset_16258_to_in_iter1_offset_16386);
  Module7Func(
      /*output*/ from_in_offset_16515_to_in_offset_32771, 
      /*output*/ from_in_offset_16515_to_in_iter1_pe_0, 
      /* input*/ from_in_offset_16387_to_in_offset_16515);
  Module9Func(
      /*output*/ from_in_offset_32768_to_in_iter1_pe_3, 
      /* input*/ from_in_offset_16512_to_in_offset_32768);
  Module10Func(
      /*output*/ from_in_iter1_pe_3_to_in_iter1_offset_0, 
      /* input*/ from_in_offset_16383_to_in_iter1_pe_3, 
      /* input*/ from_in_offset_16385_to_in_iter1_pe_3, 
      /* input*/ from_in_offset_16256_to_in_iter1_pe_3, 
      /* input*/ from_in_offset_16512_to_in_iter1_pe_3, 
      /* input*/ from_in_offset_0_to_in_iter1_pe_3, 
      /* input*/ from_in_offset_32768_to_in_iter1_pe_3, 
      /* input*/ from_in_offset_16384_to_in_iter1_pe_3);
  Module1Func(
      /*output*/ from_in_iter1_offset_0_to_in_iter1_offset_16256, 
      /*output*/ from_in_iter1_offset_0_to_in_iter2_pe_3, 
      /* input*/ from_in_iter1_pe_3_to_in_iter1_offset_0);
  Module2Func(
      /*output*/ from_in_iter1_offset_16256_to_in_iter1_offset_16384, 
      /*output*/ from_in_iter1_offset_16256_to_in_iter2_pe_3, 
      /* input*/ from_in_iter1_offset_0_to_in_iter1_offset_16256);
  Module3Func(
      /*output*/ from_in_iter1_offset_16384_to_in_iter1_offset_16388, 
      /*output*/ from_in_iter1_offset_16384_to_in_iter2_pe_3, 
      /*output*/ from_in_iter1_offset_16384_to_in_iter2_pe_2, 
      /* input*/ from_in_iter1_offset_16256_to_in_iter1_offset_16384);
  Module6Func(
      /*output*/ from_in_iter1_offset_16388_to_in_iter1_offset_16512, 
      /*output*/ from_in_iter1_offset_16388_to_in_iter2_pe_0, 
      /* input*/ from_in_iter1_offset_16384_to_in_iter1_offset_16388);
  Module7Func(
      /*output*/ from_in_iter1_offset_16513_to_in_iter1_offset_32769, 
      /*output*/ from_in_iter1_offset_16513_to_in_iter2_pe_2, 
      /* input*/ from_in_iter1_offset_16385_to_in_iter1_offset_16513);
  Module7Func(
      /*output*/ from_in_iter1_offset_16514_to_in_iter1_offset_32770, 
      /*output*/ from_in_iter1_offset_16514_to_in_iter2_pe_1, 
      /* input*/ from_in_iter1_offset_16386_to_in_iter1_offset_16514);
  Module9Func(
      /*output*/ from_in_offset_32771_to_in_iter1_pe_0, 
      /* input*/ from_in_offset_16515_to_in_offset_32771);
  Module10Func(
      /*output*/ from_in_iter1_pe_0_to_in_iter1_offset_3, 
      /* input*/ from_in_offset_16386_to_in_iter1_pe_0, 
      /* input*/ from_in_offset_16388_to_in_iter1_pe_0, 
      /* input*/ from_in_offset_16259_to_in_iter1_pe_0, 
      /* input*/ from_in_offset_16515_to_in_iter1_pe_0, 
      /* input*/ from_in_offset_3_to_in_iter1_pe_0, 
      /* input*/ from_in_offset_32771_to_in_iter1_pe_0, 
      /* input*/ from_in_offset_16387_to_in_iter1_pe_0);
  Module1Func(
      /*output*/ from_in_iter1_offset_3_to_in_iter1_offset_16259, 
      /*output*/ from_in_iter1_offset_3_to_in_iter2_pe_0, 
      /* input*/ from_in_iter1_pe_0_to_in_iter1_offset_3);
  Module2Func(
      /*output*/ from_in_iter1_offset_16259_to_in_iter1_offset_16383, 
      /*output*/ from_in_iter1_offset_16259_to_in_iter2_pe_0, 
      /* input*/ from_in_iter1_offset_3_to_in_iter1_offset_16259);
  Module5Func(
      /*output*/ from_in_iter1_offset_16383_to_in_iter1_offset_16387, 
      /*output*/ from_in_iter1_offset_16383_to_in_iter2_pe_3, 
      /* input*/ from_in_iter1_offset_16259_to_in_iter1_offset_16383);
  Module8Func(
      /*output*/ from_in_iter1_offset_16387_to_in_iter1_offset_16515, 
      /*output*/ from_in_iter1_offset_16387_to_in_iter2_pe_1, 
      /*output*/ from_in_iter1_offset_16387_to_in_iter2_pe_0, 
      /* input*/ from_in_iter1_offset_16383_to_in_iter1_offset_16387);
  Module5Func(
      /*output*/ from_in_iter1_offset_16512_to_in_iter1_offset_32768, 
      /*output*/ from_in_iter1_offset_16512_to_in_iter2_pe_3, 
      /* input*/ from_in_iter1_offset_16388_to_in_iter1_offset_16512);
  Module9Func(
      /*output*/ from_in_iter1_offset_32769_to_in_iter2_pe_2, 
      /* input*/ from_in_iter1_offset_16513_to_in_iter1_offset_32769);
  Module10Func(
      /*output*/ from_in_iter2_pe_2_to_in_iter2_offset_1, 
      /* input*/ from_in_iter1_offset_16384_to_in_iter2_pe_2, 
      /* input*/ from_in_iter1_offset_16386_to_in_iter2_pe_2, 
      /* input*/ from_in_iter1_offset_16257_to_in_iter2_pe_2, 
      /* input*/ from_in_iter1_offset_16513_to_in_iter2_pe_2, 
      /* input*/ from_in_iter1_offset_1_to_in_iter2_pe_2, 
      /* input*/ from_in_iter1_offset_32769_to_in_iter2_pe_2, 
      /* input*/ from_in_iter1_offset_16385_to_in_iter2_pe_2);
  Module1Func(
      /*output*/ from_in_iter2_offset_1_to_in_iter2_offset_16257, 
      /*output*/ from_in_iter2_offset_1_to_in_iter3_pe_2, 
      /* input*/ from_in_iter2_pe_2_to_in_iter2_offset_1);
  Module2Func(
      /*output*/ from_in_iter2_offset_16257_to_in_iter2_offset_16385, 
      /*output*/ from_in_iter2_offset_16257_to_in_iter3_pe_2, 
      /* input*/ from_in_iter2_offset_1_to_in_iter2_offset_16257);
  Module4Func(
      /*output*/ from_in_iter2_offset_16385_to_in_iter2_offset_16513, 
      /*output*/ from_in_iter2_offset_16385_to_in_iter3_pe_3, 
      /*output*/ from_in_iter2_offset_16385_to_in_iter3_pe_2, 
      /*output*/ from_in_iter2_offset_16385_to_in_iter3_pe_1, 
      /* input*/ from_in_iter2_offset_16257_to_in_iter2_offset_16385);
  Module9Func(
      /*output*/ from_in_iter1_offset_32770_to_in_iter2_pe_1, 
      /* input*/ from_in_iter1_offset_16514_to_in_iter1_offset_32770);
  Module10Func(
      /*output*/ from_in_iter2_pe_1_to_in_iter2_offset_2, 
      /* input*/ from_in_iter1_offset_16385_to_in_iter2_pe_1, 
      /* input*/ from_in_iter1_offset_16387_to_in_iter2_pe_1, 
      /* input*/ from_in_iter1_offset_16258_to_in_iter2_pe_1, 
      /* input*/ from_in_iter1_offset_16514_to_in_iter2_pe_1, 
      /* input*/ from_in_iter1_offset_2_to_in_iter2_pe_1, 
      /* input*/ from_in_iter1_offset_32770_to_in_iter2_pe_1, 
      /* input*/ from_in_iter1_offset_16386_to_in_iter2_pe_1);
  Module1Func(
      /*output*/ from_in_iter2_offset_2_to_in_iter2_offset_16258, 
      /*output*/ from_in_iter2_offset_2_to_in_iter3_pe_1, 
      /* input*/ from_in_iter2_pe_1_to_in_iter2_offset_2);
  Module2Func(
      /*output*/ from_in_iter2_offset_16258_to_in_iter2_offset_16386, 
      /*output*/ from_in_iter2_offset_16258_to_in_iter3_pe_1, 
      /* input*/ from_in_iter2_offset_2_to_in_iter2_offset_16258);
  Module4Func(
      /*output*/ from_in_iter2_offset_16386_to_in_iter2_offset_16514, 
      /*output*/ from_in_iter2_offset_16386_to_in_iter3_pe_2, 
      /*output*/ from_in_iter2_offset_16386_to_in_iter3_pe_1, 
      /*output*/ from_in_iter2_offset_16386_to_in_iter3_pe_0, 
      /* input*/ from_in_iter2_offset_16258_to_in_iter2_offset_16386);
  Module7Func(
      /*output*/ from_in_iter1_offset_16515_to_in_iter1_offset_32771, 
      /*output*/ from_in_iter1_offset_16515_to_in_iter2_pe_0, 
      /* input*/ from_in_iter1_offset_16387_to_in_iter1_offset_16515);
  Module9Func(
      /*output*/ from_in_iter1_offset_32768_to_in_iter2_pe_3, 
      /* input*/ from_in_iter1_offset_16512_to_in_iter1_offset_32768);
  Module10Func(
      /*output*/ from_in_iter2_pe_3_to_in_iter2_offset_0, 
      /* input*/ from_in_iter1_offset_16383_to_in_iter2_pe_3, 
      /* input*/ from_in_iter1_offset_16385_to_in_iter2_pe_3, 
      /* input*/ from_in_iter1_offset_16256_to_in_iter2_pe_3, 
      /* input*/ from_in_iter1_offset_16512_to_in_iter2_pe_3, 
      /* input*/ from_in_iter1_offset_0_to_in_iter2_pe_3, 
      /* input*/ from_in_iter1_offset_32768_to_in_iter2_pe_3, 
      /* input*/ from_in_iter1_offset_16384_to_in_iter2_pe_3);
  Module1Func(
      /*output*/ from_in_iter2_offset_0_to_in_iter2_offset_16256, 
      /*output*/ from_in_iter2_offset_0_to_in_iter3_pe_3, 
      /* input*/ from_in_iter2_pe_3_to_in_iter2_offset_0);
  Module2Func(
      /*output*/ from_in_iter2_offset_16256_to_in_iter2_offset_16384, 
      /*output*/ from_in_iter2_offset_16256_to_in_iter3_pe_3, 
      /* input*/ from_in_iter2_offset_0_to_in_iter2_offset_16256);
  Module3Func(
      /*output*/ from_in_iter2_offset_16384_to_in_iter2_offset_16388, 
      /*output*/ from_in_iter2_offset_16384_to_in_iter3_pe_3, 
      /*output*/ from_in_iter2_offset_16384_to_in_iter3_pe_2, 
      /* input*/ from_in_iter2_offset_16256_to_in_iter2_offset_16384);
  Module6Func(
      /*output*/ from_in_iter2_offset_16388_to_in_iter2_offset_16512, 
      /*output*/ from_in_iter2_offset_16388_to_in_iter3_pe_0, 
      /* input*/ from_in_iter2_offset_16384_to_in_iter2_offset_16388);
  Module7Func(
      /*output*/ from_in_iter2_offset_16513_to_in_iter2_offset_32769, 
      /*output*/ from_in_iter2_offset_16513_to_in_iter3_pe_2, 
      /* input*/ from_in_iter2_offset_16385_to_in_iter2_offset_16513);
  Module7Func(
      /*output*/ from_in_iter2_offset_16514_to_in_iter2_offset_32770, 
      /*output*/ from_in_iter2_offset_16514_to_in_iter3_pe_1, 
      /* input*/ from_in_iter2_offset_16386_to_in_iter2_offset_16514);
  Module9Func(
      /*output*/ from_in_iter1_offset_32771_to_in_iter2_pe_0, 
      /* input*/ from_in_iter1_offset_16515_to_in_iter1_offset_32771);
  Module10Func(
      /*output*/ from_in_iter2_pe_0_to_in_iter2_offset_3, 
      /* input*/ from_in_iter1_offset_16386_to_in_iter2_pe_0, 
      /* input*/ from_in_iter1_offset_16388_to_in_iter2_pe_0, 
      /* input*/ from_in_iter1_offset_16259_to_in_iter2_pe_0, 
      /* input*/ from_in_iter1_offset_16515_to_in_iter2_pe_0, 
      /* input*/ from_in_iter1_offset_3_to_in_iter2_pe_0, 
      /* input*/ from_in_iter1_offset_32771_to_in_iter2_pe_0, 
      /* input*/ from_in_iter1_offset_16387_to_in_iter2_pe_0);
  Module1Func(
      /*output*/ from_in_iter2_offset_3_to_in_iter2_offset_16259, 
      /*output*/ from_in_iter2_offset_3_to_in_iter3_pe_0, 
      /* input*/ from_in_iter2_pe_0_to_in_iter2_offset_3);
  Module2Func(
      /*output*/ from_in_iter2_offset_16259_to_in_iter2_offset_16383, 
      /*output*/ from_in_iter2_offset_16259_to_in_iter3_pe_0, 
      /* input*/ from_in_iter2_offset_3_to_in_iter2_offset_16259);
  Module5Func(
      /*output*/ from_in_iter2_offset_16383_to_in_iter2_offset_16387, 
      /*output*/ from_in_iter2_offset_16383_to_in_iter3_pe_3, 
      /* input*/ from_in_iter2_offset_16259_to_in_iter2_offset_16383);
  Module8Func(
      /*output*/ from_in_iter2_offset_16387_to_in_iter2_offset_16515, 
      /*output*/ from_in_iter2_offset_16387_to_in_iter3_pe_1, 
      /*output*/ from_in_iter2_offset_16387_to_in_iter3_pe_0, 
      /* input*/ from_in_iter2_offset_16383_to_in_iter2_offset_16387);
  Module5Func(
      /*output*/ from_in_iter2_offset_16512_to_in_iter2_offset_32768, 
      /*output*/ from_in_iter2_offset_16512_to_in_iter3_pe_3, 
      /* input*/ from_in_iter2_offset_16388_to_in_iter2_offset_16512);
  Module9Func(
      /*output*/ from_in_iter2_offset_32769_to_in_iter3_pe_2, 
      /* input*/ from_in_iter2_offset_16513_to_in_iter2_offset_32769);
  Module10Func(
      /*output*/ from_in_iter3_pe_2_to_in_iter3_offset_1, 
      /* input*/ from_in_iter2_offset_16384_to_in_iter3_pe_2, 
      /* input*/ from_in_iter2_offset_16386_to_in_iter3_pe_2, 
      /* input*/ from_in_iter2_offset_16257_to_in_iter3_pe_2, 
      /* input*/ from_in_iter2_offset_16513_to_in_iter3_pe_2, 
      /* input*/ from_in_iter2_offset_1_to_in_iter3_pe_2, 
      /* input*/ from_in_iter2_offset_32769_to_in_iter3_pe_2, 
      /* input*/ from_in_iter2_offset_16385_to_in_iter3_pe_2);
  Module1Func(
      /*output*/ from_in_iter3_offset_1_to_in_iter3_offset_16257, 
      /*output*/ from_in_iter3_offset_1_to_in_iter4_pe_2, 
      /* input*/ from_in_iter3_pe_2_to_in_iter3_offset_1);
  Module2Func(
      /*output*/ from_in_iter3_offset_16257_to_in_iter3_offset_16385, 
      /*output*/ from_in_iter3_offset_16257_to_in_iter4_pe_2, 
      /* input*/ from_in_iter3_offset_1_to_in_iter3_offset_16257);
  Module4Func(
      /*output*/ from_in_iter3_offset_16385_to_in_iter3_offset_16513, 
      /*output*/ from_in_iter3_offset_16385_to_in_iter4_pe_3, 
      /*output*/ from_in_iter3_offset_16385_to_in_iter4_pe_2, 
      /*output*/ from_in_iter3_offset_16385_to_in_iter4_pe_1, 
      /* input*/ from_in_iter3_offset_16257_to_in_iter3_offset_16385);
  Module9Func(
      /*output*/ from_in_iter2_offset_32770_to_in_iter3_pe_1, 
      /* input*/ from_in_iter2_offset_16514_to_in_iter2_offset_32770);
  Module10Func(
      /*output*/ from_in_iter3_pe_1_to_in_iter3_offset_2, 
      /* input*/ from_in_iter2_offset_16385_to_in_iter3_pe_1, 
      /* input*/ from_in_iter2_offset_16387_to_in_iter3_pe_1, 
      /* input*/ from_in_iter2_offset_16258_to_in_iter3_pe_1, 
      /* input*/ from_in_iter2_offset_16514_to_in_iter3_pe_1, 
      /* input*/ from_in_iter2_offset_2_to_in_iter3_pe_1, 
      /* input*/ from_in_iter2_offset_32770_to_in_iter3_pe_1, 
      /* input*/ from_in_iter2_offset_16386_to_in_iter3_pe_1);
  Module1Func(
      /*output*/ from_in_iter3_offset_2_to_in_iter3_offset_16258, 
      /*output*/ from_in_iter3_offset_2_to_in_iter4_pe_1, 
      /* input*/ from_in_iter3_pe_1_to_in_iter3_offset_2);
  Module2Func(
      /*output*/ from_in_iter3_offset_16258_to_in_iter3_offset_16386, 
      /*output*/ from_in_iter3_offset_16258_to_in_iter4_pe_1, 
      /* input*/ from_in_iter3_offset_2_to_in_iter3_offset_16258);
  Module4Func(
      /*output*/ from_in_iter3_offset_16386_to_in_iter3_offset_16514, 
      /*output*/ from_in_iter3_offset_16386_to_in_iter4_pe_2, 
      /*output*/ from_in_iter3_offset_16386_to_in_iter4_pe_1, 
      /*output*/ from_in_iter3_offset_16386_to_in_iter4_pe_0, 
      /* input*/ from_in_iter3_offset_16258_to_in_iter3_offset_16386);
  Module7Func(
      /*output*/ from_in_iter2_offset_16515_to_in_iter2_offset_32771, 
      /*output*/ from_in_iter2_offset_16515_to_in_iter3_pe_0, 
      /* input*/ from_in_iter2_offset_16387_to_in_iter2_offset_16515);
  Module9Func(
      /*output*/ from_in_iter2_offset_32768_to_in_iter3_pe_3, 
      /* input*/ from_in_iter2_offset_16512_to_in_iter2_offset_32768);
  Module10Func(
      /*output*/ from_in_iter3_pe_3_to_in_iter3_offset_0, 
      /* input*/ from_in_iter2_offset_16383_to_in_iter3_pe_3, 
      /* input*/ from_in_iter2_offset_16385_to_in_iter3_pe_3, 
      /* input*/ from_in_iter2_offset_16256_to_in_iter3_pe_3, 
      /* input*/ from_in_iter2_offset_16512_to_in_iter3_pe_3, 
      /* input*/ from_in_iter2_offset_0_to_in_iter3_pe_3, 
      /* input*/ from_in_iter2_offset_32768_to_in_iter3_pe_3, 
      /* input*/ from_in_iter2_offset_16384_to_in_iter3_pe_3);
  Module1Func(
      /*output*/ from_in_iter3_offset_0_to_in_iter3_offset_16256, 
      /*output*/ from_in_iter3_offset_0_to_in_iter4_pe_3, 
      /* input*/ from_in_iter3_pe_3_to_in_iter3_offset_0);
  Module2Func(
      /*output*/ from_in_iter3_offset_16256_to_in_iter3_offset_16384, 
      /*output*/ from_in_iter3_offset_16256_to_in_iter4_pe_3, 
      /* input*/ from_in_iter3_offset_0_to_in_iter3_offset_16256);
  Module3Func(
      /*output*/ from_in_iter3_offset_16384_to_in_iter3_offset_16388, 
      /*output*/ from_in_iter3_offset_16384_to_in_iter4_pe_3, 
      /*output*/ from_in_iter3_offset_16384_to_in_iter4_pe_2, 
      /* input*/ from_in_iter3_offset_16256_to_in_iter3_offset_16384);
  Module6Func(
      /*output*/ from_in_iter3_offset_16388_to_in_iter3_offset_16512, 
      /*output*/ from_in_iter3_offset_16388_to_in_iter4_pe_0, 
      /* input*/ from_in_iter3_offset_16384_to_in_iter3_offset_16388);
  Module7Func(
      /*output*/ from_in_iter3_offset_16513_to_in_iter3_offset_32769, 
      /*output*/ from_in_iter3_offset_16513_to_in_iter4_pe_2, 
      /* input*/ from_in_iter3_offset_16385_to_in_iter3_offset_16513);
  Module7Func(
      /*output*/ from_in_iter3_offset_16514_to_in_iter3_offset_32770, 
      /*output*/ from_in_iter3_offset_16514_to_in_iter4_pe_1, 
      /* input*/ from_in_iter3_offset_16386_to_in_iter3_offset_16514);
  Module9Func(
      /*output*/ from_in_iter2_offset_32771_to_in_iter3_pe_0, 
      /* input*/ from_in_iter2_offset_16515_to_in_iter2_offset_32771);
  Module10Func(
      /*output*/ from_in_iter3_pe_0_to_in_iter3_offset_3, 
      /* input*/ from_in_iter2_offset_16386_to_in_iter3_pe_0, 
      /* input*/ from_in_iter2_offset_16388_to_in_iter3_pe_0, 
      /* input*/ from_in_iter2_offset_16259_to_in_iter3_pe_0, 
      /* input*/ from_in_iter2_offset_16515_to_in_iter3_pe_0, 
      /* input*/ from_in_iter2_offset_3_to_in_iter3_pe_0, 
      /* input*/ from_in_iter2_offset_32771_to_in_iter3_pe_0, 
      /* input*/ from_in_iter2_offset_16387_to_in_iter3_pe_0);
  Module1Func(
      /*output*/ from_in_iter3_offset_3_to_in_iter3_offset_16259, 
      /*output*/ from_in_iter3_offset_3_to_in_iter4_pe_0, 
      /* input*/ from_in_iter3_pe_0_to_in_iter3_offset_3);
  Module2Func(
      /*output*/ from_in_iter3_offset_16259_to_in_iter3_offset_16383, 
      /*output*/ from_in_iter3_offset_16259_to_in_iter4_pe_0, 
      /* input*/ from_in_iter3_offset_3_to_in_iter3_offset_16259);
  Module5Func(
      /*output*/ from_in_iter3_offset_16383_to_in_iter3_offset_16387, 
      /*output*/ from_in_iter3_offset_16383_to_in_iter4_pe_3, 
      /* input*/ from_in_iter3_offset_16259_to_in_iter3_offset_16383);
  Module8Func(
      /*output*/ from_in_iter3_offset_16387_to_in_iter3_offset_16515, 
      /*output*/ from_in_iter3_offset_16387_to_in_iter4_pe_1, 
      /*output*/ from_in_iter3_offset_16387_to_in_iter4_pe_0, 
      /* input*/ from_in_iter3_offset_16383_to_in_iter3_offset_16387);
  Module5Func(
      /*output*/ from_in_iter3_offset_16512_to_in_iter3_offset_32768, 
      /*output*/ from_in_iter3_offset_16512_to_in_iter4_pe_3, 
      /* input*/ from_in_iter3_offset_16388_to_in_iter3_offset_16512);
  Module9Func(
      /*output*/ from_in_iter3_offset_32769_to_in_iter4_pe_2, 
      /* input*/ from_in_iter3_offset_16513_to_in_iter3_offset_32769);
  Module10Func(
      /*output*/ from_in_iter4_pe_2_to_in_iter4_offset_1, 
      /* input*/ from_in_iter3_offset_16384_to_in_iter4_pe_2, 
      /* input*/ from_in_iter3_offset_16386_to_in_iter4_pe_2, 
      /* input*/ from_in_iter3_offset_16257_to_in_iter4_pe_2, 
      /* input*/ from_in_iter3_offset_16513_to_in_iter4_pe_2, 
      /* input*/ from_in_iter3_offset_1_to_in_iter4_pe_2, 
      /* input*/ from_in_iter3_offset_32769_to_in_iter4_pe_2, 
      /* input*/ from_in_iter3_offset_16385_to_in_iter4_pe_2);
  Module1Func(
      /*output*/ from_in_iter4_offset_1_to_in_iter4_offset_16257, 
      /*output*/ from_in_iter4_offset_1_to_out_pe_2, 
      /* input*/ from_in_iter4_pe_2_to_in_iter4_offset_1);
  Module2Func(
      /*output*/ from_in_iter4_offset_16257_to_in_iter4_offset_16385, 
      /*output*/ from_in_iter4_offset_16257_to_out_pe_2, 
      /* input*/ from_in_iter4_offset_1_to_in_iter4_offset_16257);
  Module4Func(
      /*output*/ from_in_iter4_offset_16385_to_in_iter4_offset_16513, 
      /*output*/ from_in_iter4_offset_16385_to_out_pe_3, 
      /*output*/ from_in_iter4_offset_16385_to_out_pe_2, 
      /*output*/ from_in_iter4_offset_16385_to_out_pe_1, 
      /* input*/ from_in_iter4_offset_16257_to_in_iter4_offset_16385);
  Module9Func(
      /*output*/ from_in_iter3_offset_32770_to_in_iter4_pe_1, 
      /* input*/ from_in_iter3_offset_16514_to_in_iter3_offset_32770);
  Module10Func(
      /*output*/ from_in_iter4_pe_1_to_in_iter4_offset_2, 
      /* input*/ from_in_iter3_offset_16385_to_in_iter4_pe_1, 
      /* input*/ from_in_iter3_offset_16387_to_in_iter4_pe_1, 
      /* input*/ from_in_iter3_offset_16258_to_in_iter4_pe_1, 
      /* input*/ from_in_iter3_offset_16514_to_in_iter4_pe_1, 
      /* input*/ from_in_iter3_offset_2_to_in_iter4_pe_1, 
      /* input*/ from_in_iter3_offset_32770_to_in_iter4_pe_1, 
      /* input*/ from_in_iter3_offset_16386_to_in_iter4_pe_1);
  Module1Func(
      /*output*/ from_in_iter4_offset_2_to_in_iter4_offset_16258, 
      /*output*/ from_in_iter4_offset_2_to_out_pe_1, 
      /* input*/ from_in_iter4_pe_1_to_in_iter4_offset_2);
  Module2Func(
      /*output*/ from_in_iter4_offset_16258_to_in_iter4_offset_16386, 
      /*output*/ from_in_iter4_offset_16258_to_out_pe_1, 
      /* input*/ from_in_iter4_offset_2_to_in_iter4_offset_16258);
  Module4Func(
      /*output*/ from_in_iter4_offset_16386_to_in_iter4_offset_16514, 
      /*output*/ from_in_iter4_offset_16386_to_out_pe_2, 
      /*output*/ from_in_iter4_offset_16386_to_out_pe_1, 
      /*output*/ from_in_iter4_offset_16386_to_out_pe_0, 
      /* input*/ from_in_iter4_offset_16258_to_in_iter4_offset_16386);
  Module7Func(
      /*output*/ from_in_iter3_offset_16515_to_in_iter3_offset_32771, 
      /*output*/ from_in_iter3_offset_16515_to_in_iter4_pe_0, 
      /* input*/ from_in_iter3_offset_16387_to_in_iter3_offset_16515);
  Module9Func(
      /*output*/ from_in_iter3_offset_32768_to_in_iter4_pe_3, 
      /* input*/ from_in_iter3_offset_16512_to_in_iter3_offset_32768);
  Module10Func(
      /*output*/ from_in_iter4_pe_3_to_in_iter4_offset_0, 
      /* input*/ from_in_iter3_offset_16383_to_in_iter4_pe_3, 
      /* input*/ from_in_iter3_offset_16385_to_in_iter4_pe_3, 
      /* input*/ from_in_iter3_offset_16256_to_in_iter4_pe_3, 
      /* input*/ from_in_iter3_offset_16512_to_in_iter4_pe_3, 
      /* input*/ from_in_iter3_offset_0_to_in_iter4_pe_3, 
      /* input*/ from_in_iter3_offset_32768_to_in_iter4_pe_3, 
      /* input*/ from_in_iter3_offset_16384_to_in_iter4_pe_3);
  Module1Func(
      /*output*/ from_in_iter4_offset_0_to_in_iter4_offset_16256, 
      /*output*/ from_in_iter4_offset_0_to_out_pe_3, 
      /* input*/ from_in_iter4_pe_3_to_in_iter4_offset_0);
  Module2Func(
      /*output*/ from_in_iter4_offset_16256_to_in_iter4_offset_16384, 
      /*output*/ from_in_iter4_offset_16256_to_out_pe_3, 
      /* input*/ from_in_iter4_offset_0_to_in_iter4_offset_16256);
  Module3Func(
      /*output*/ from_in_iter4_offset_16384_to_in_iter4_offset_16388, 
      /*output*/ from_in_iter4_offset_16384_to_out_pe_3, 
      /*output*/ from_in_iter4_offset_16384_to_out_pe_2, 
      /* input*/ from_in_iter4_offset_16256_to_in_iter4_offset_16384);
  Module6Func(
      /*output*/ from_in_iter4_offset_16388_to_in_iter4_offset_16512, 
      /*output*/ from_in_iter4_offset_16388_to_out_pe_0, 
      /* input*/ from_in_iter4_offset_16384_to_in_iter4_offset_16388);
  Module7Func(
      /*output*/ from_in_iter4_offset_16513_to_in_iter4_offset_32769, 
      /*output*/ from_in_iter4_offset_16513_to_out_pe_2, 
      /* input*/ from_in_iter4_offset_16385_to_in_iter4_offset_16513);
  Module7Func(
      /*output*/ from_in_iter4_offset_16514_to_in_iter4_offset_32770, 
      /*output*/ from_in_iter4_offset_16514_to_out_pe_1, 
      /* input*/ from_in_iter4_offset_16386_to_in_iter4_offset_16514);
  Module9Func(
      /*output*/ from_in_iter3_offset_32771_to_in_iter4_pe_0, 
      /* input*/ from_in_iter3_offset_16515_to_in_iter3_offset_32771);
  Module10Func(
      /*output*/ from_in_iter4_pe_0_to_in_iter4_offset_3, 
      /* input*/ from_in_iter3_offset_16386_to_in_iter4_pe_0, 
      /* input*/ from_in_iter3_offset_16388_to_in_iter4_pe_0, 
      /* input*/ from_in_iter3_offset_16259_to_in_iter4_pe_0, 
      /* input*/ from_in_iter3_offset_16515_to_in_iter4_pe_0, 
      /* input*/ from_in_iter3_offset_3_to_in_iter4_pe_0, 
      /* input*/ from_in_iter3_offset_32771_to_in_iter4_pe_0, 
      /* input*/ from_in_iter3_offset_16387_to_in_iter4_pe_0);
  Module1Func(
      /*output*/ from_in_iter4_offset_3_to_in_iter4_offset_16259, 
      /*output*/ from_in_iter4_offset_3_to_out_pe_0, 
      /* input*/ from_in_iter4_pe_0_to_in_iter4_offset_3);
  Module2Func(
      /*output*/ from_in_iter4_offset_16259_to_in_iter4_offset_16383, 
      /*output*/ from_in_iter4_offset_16259_to_out_pe_0, 
      /* input*/ from_in_iter4_offset_3_to_in_iter4_offset_16259);
  Module5Func(
      /*output*/ from_in_iter4_offset_16383_to_in_iter4_offset_16387, 
      /*output*/ from_in_iter4_offset_16383_to_out_pe_3, 
      /* input*/ from_in_iter4_offset_16259_to_in_iter4_offset_16383);
  Module8Func(
      /*output*/ from_in_iter4_offset_16387_to_in_iter4_offset_16515, 
      /*output*/ from_in_iter4_offset_16387_to_out_pe_1, 
      /*output*/ from_in_iter4_offset_16387_to_out_pe_0, 
      /* input*/ from_in_iter4_offset_16383_to_in_iter4_offset_16387);
  Module5Func(
      /*output*/ from_in_iter4_offset_16512_to_in_iter4_offset_32768, 
      /*output*/ from_in_iter4_offset_16512_to_out_pe_3, 
      /* input*/ from_in_iter4_offset_16388_to_in_iter4_offset_16512);
  Module9Func(
      /*output*/ from_in_iter4_offset_32769_to_out_pe_2, 
      /* input*/ from_in_iter4_offset_16513_to_in_iter4_offset_32769);
  Module10Func(
      /*output*/ from_out_pe_2_to_out_bank_1, 
      /* input*/ from_in_iter4_offset_16384_to_out_pe_2, 
      /* input*/ from_in_iter4_offset_16386_to_out_pe_2, 
      /* input*/ from_in_iter4_offset_16257_to_out_pe_2, 
      /* input*/ from_in_iter4_offset_16513_to_out_pe_2, 
      /* input*/ from_in_iter4_offset_1_to_out_pe_2, 
      /* input*/ from_in_iter4_offset_32769_to_out_pe_2, 
      /* input*/ from_in_iter4_offset_16385_to_out_pe_2);
  Module9Func(
      /*output*/ from_in_iter4_offset_32770_to_out_pe_1, 
      /* input*/ from_in_iter4_offset_16514_to_in_iter4_offset_32770);
  Module10Func(
      /*output*/ from_out_pe_1_to_out_bank_1, 
      /* input*/ from_in_iter4_offset_16385_to_out_pe_1, 
      /* input*/ from_in_iter4_offset_16387_to_out_pe_1, 
      /* input*/ from_in_iter4_offset_16258_to_out_pe_1, 
      /* input*/ from_in_iter4_offset_16514_to_out_pe_1, 
      /* input*/ from_in_iter4_offset_2_to_out_pe_1, 
      /* input*/ from_in_iter4_offset_32770_to_out_pe_1, 
      /* input*/ from_in_iter4_offset_16386_to_out_pe_1);
  Module7Func(
      /*output*/ from_in_iter4_offset_16515_to_in_iter4_offset_32771, 
      /*output*/ from_in_iter4_offset_16515_to_out_pe_0, 
      /* input*/ from_in_iter4_offset_16387_to_in_iter4_offset_16515);
  Module9Func(
      /*output*/ from_in_iter4_offset_32768_to_out_pe_3, 
      /* input*/ from_in_iter4_offset_16512_to_in_iter4_offset_32768);
  Module10Func(
      /*output*/ from_out_pe_3_to_out_bank_1, 
      /* input*/ from_in_iter4_offset_16383_to_out_pe_3, 
      /* input*/ from_in_iter4_offset_16385_to_out_pe_3, 
      /* input*/ from_in_iter4_offset_16256_to_out_pe_3, 
      /* input*/ from_in_iter4_offset_16512_to_out_pe_3, 
      /* input*/ from_in_iter4_offset_0_to_out_pe_3, 
      /* input*/ from_in_iter4_offset_32768_to_out_pe_3, 
      /* input*/ from_in_iter4_offset_16384_to_out_pe_3);
  Module9Func(
      /*output*/ from_in_iter4_offset_32771_to_out_pe_0, 
      /* input*/ from_in_iter4_offset_16515_to_in_iter4_offset_32771);
  Module10Func(
      /*output*/ from_out_pe_0_to_out_bank_1, 
      /* input*/ from_in_iter4_offset_16386_to_out_pe_0, 
      /* input*/ from_in_iter4_offset_16388_to_out_pe_0, 
      /* input*/ from_in_iter4_offset_16259_to_out_pe_0, 
      /* input*/ from_in_iter4_offset_16515_to_out_pe_0, 
      /* input*/ from_in_iter4_offset_3_to_out_pe_0, 
      /* input*/ from_in_iter4_offset_32771_to_out_pe_0, 
      /* input*/ from_in_iter4_offset_16387_to_out_pe_0);
  Module11Func(
      /*output*/ bank_1_out_buf, 
      /* input*/ from_out_pe_2_to_out_bank_1, 
      /* input*/ from_out_pe_1_to_out_bank_1, 
      /* input*/ from_out_pe_3_to_out_bank_1, 
      /* input*/ from_out_pe_0_to_out_bank_1);
  BurstWrite(bank_1_out, bank_1_out_buf, coalesced_data_num);
}

}  // extern "C"
