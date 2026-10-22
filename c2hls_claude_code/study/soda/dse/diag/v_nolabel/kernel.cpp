// this file can be generated from the following SODA DSL
/*
kernel: blur
burst width: 128
iterate: 1
unroll factor: 4
input dram 0 uint16: input(2000, *)
local uint16: blur_x(0, 0) = (input(0, 0) + input(0, 1) + input(0, 2)) / 3
output dram 1 uint16: blur_y(0, 0) = (blur_x(0, 0) + blur_x(1, 0) + blur_x(2, 0)) / 3
border: ignore
cluster: none
*/

// stencil window size: (3, 3)
// stencil distace: 2050
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
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 131329 max = 131329
    const uint64_t next_i = i + 1;
    WriteData(to, from[i], next_i < data_num);
    i = next_i;
  }
}

template <typename T>
void BurstWrite(T* to, hls::stream<Data<T>>& from, uint64_t data_num) {
store:
  for (uint64_t i = 0; i < data_num; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 131329 max = 131329
    T buf;
    ReadData(buf, from);
    to[i] = buf;
  }
}

void Module0Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_1, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_2, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_3, 
  /* input*/ hls::stream<Data<ap_uint<128>>>& dram_input_bank_0_fifo)
{
#pragma HLS aggregate bit variable = dram_input_bank_0_fifo
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_st_3
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 2
#pragma HLS loop_tripcount min = 262657 max = 262657
    if (!dram_input_bank_0_fifo.empty())
    {
      ap_uint<128> dram_input_bank_0_buf;
      const bool dram_input_bank_0_buf_enable = ReadData(dram_input_bank_0_buf, dram_input_bank_0_fifo);
      const bool enabled = dram_input_bank_0_buf_enable;
      enable = enabled;
      WriteData(fifo_st_0, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(63, 48))), true);
      WriteData(fifo_st_1, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(47, 32))), true);
      WriteData(fifo_st_2, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(31, 16))), true);
      WriteData(fifo_st_3, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(15, 0))), true);
      WriteData(fifo_st_0, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(127, 112))), enabled);
      WriteData(fifo_st_1, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(111, 96))), enabled);
      WriteData(fifo_st_2, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(95, 80))), enabled);
      WriteData(fifo_st_3, Reinterpret<uint16_t>(static_cast<ap_uint<16>>(dram_input_bank_0_buf(79, 64))), enabled);
    } // if not empty
  } // for module_0
} // Module0Func

void Module1Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_1, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 262657 max = 262657
    if (!fifo_ld_0.empty())
    {
      uint16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, uint16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_1, uint16_t(fifo_ref_0), enabled);
    } // if not empty
  } // for module_1
} // Module1Func

void Module2Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_1, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
  uint8_t ptr_delay_256 = 0;
  uint16_t fifo_ref_0_delayed_256_buf[256];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_256_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 262657 max = 262657
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_256_buf distance = 256
    if (!fifo_ld_0.empty())
    {
      uint16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const uint16_t fifo_ref_0_delayed_256 = fifo_ref_0_delayed_256_buf[ptr_delay_256];;
      const uint16_t let_0 = fifo_ref_0_delayed_256;
      WriteData(fifo_st_0, uint16_t(let_0), enabled);
      WriteData(fifo_st_1, uint16_t(let_0), enabled);
      fifo_ref_0_delayed_256_buf[ptr_delay_256] = fifo_ref_0;
      ptr_delay_256 < 255 ? (++ptr_delay_256) : (ptr_delay_256 = 0);
    } // if not empty
  } // for module_2
} // Module2Func

void Module3Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
  uint8_t ptr_delay_256 = 0;
  uint16_t fifo_ref_0_delayed_256_buf[256];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_256_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 262657 max = 262657
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_256_buf distance = 256
    if (!fifo_ld_0.empty())
    {
      uint16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const uint16_t fifo_ref_0_delayed_256 = fifo_ref_0_delayed_256_buf[ptr_delay_256];;
      const uint16_t let_0 = fifo_ref_0_delayed_256;
      WriteData(fifo_st_0, uint16_t(let_0), enabled);
      fifo_ref_0_delayed_256_buf[ptr_delay_256] = fifo_ref_0;
      ptr_delay_256 < 255 ? (++ptr_delay_256) : (ptr_delay_256 = 0);
    } // if not empty
  } // for module_3
} // Module3Func

void Module4Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_2)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#pragma HLS aggregate bit variable = fifo_ld_2
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 262657 max = 262657
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty())
    {
      uint16_t fifo_ref_0;
      uint16_t fifo_ref_1;
      uint16_t fifo_ref_2;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable;
      enable = enabled;
      WriteData(fifo_st_0, uint16_t((uint16_t)((fifo_ref_0 + fifo_ref_1 + fifo_ref_2) / 3)), enabled);
    } // if not empty
  } // for module_4
} // Module4Func

void Module5Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_1, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_2, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_st_2
#pragma HLS aggregate bit variable = fifo_ld_0
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 262657 max = 262657
    if (!fifo_ld_0.empty())
    {
      uint16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, uint16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_1, uint16_t(fifo_ref_0), enabled);
      WriteData(fifo_st_2, uint16_t(fifo_ref_0), enabled);
    } // if not empty
  } // for module_5
} // Module5Func

void Module6Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_1, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_st_1
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<1> ptr_delay_1 = 0;
  uint16_t fifo_ref_0_delayed_1_buf[1];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_1_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 262657 max = 262657
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_1_buf distance = 1
    if (!fifo_ld_0.empty())
    {
      uint16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const uint16_t fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const uint16_t let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, uint16_t(let_0), enabled);
      WriteData(fifo_st_1, uint16_t(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_6
} // Module6Func

void Module7Func(
  /*output*/ hls::stream<Data<uint16_t>>& fifo_st_0, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0)
{
#pragma HLS aggregate bit variable = fifo_st_0
#pragma HLS aggregate bit variable = fifo_ld_0
  ap_uint<1> ptr_delay_1 = 0;
  uint16_t fifo_ref_0_delayed_1_buf[1];
#ifdef __VITIS_HLS__
#pragma HLS bind_storage variable = fifo_ref_0_delayed_1_buf type = ram_s2p latency = 3
#endif  // __VITIS_HLS__
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 262657 max = 262657
#pragma HLS dependence inter true variable = fifo_ref_0_delayed_1_buf distance = 1
    if (!fifo_ld_0.empty())
    {
      uint16_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const uint16_t fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const uint16_t let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, uint16_t(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_7
} // Module7Func

void Module8Func(
  /*output*/ hls::stream<Data<ap_uint<128>>>& dram_blur_y_bank_1_fifo, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<uint16_t>>& fifo_ld_3)
{
#pragma HLS aggregate bit variable = dram_blur_y_bank_1_fifo
#pragma HLS aggregate bit variable = fifo_ld_0
#pragma HLS aggregate bit variable = fifo_ld_1
#pragma HLS aggregate bit variable = fifo_ld_2
#pragma HLS aggregate bit variable = fifo_ld_3
#ifdef __VITIS_HLS__
#endif  // __VITIS_HLS__
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 2
#pragma HLS loop_tripcount min = 262657 max = 262657
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty())
    {
      uint16_t fifo_ref_0;
      uint16_t fifo_ref_1;
      uint16_t fifo_ref_2;
      uint16_t fifo_ref_3;
      ap_uint<128> dram_blur_y_bank_1_buf;
      ReadData(fifo_ref_0, fifo_ld_0);
      ReadData(fifo_ref_1, fifo_ld_1);
      ReadData(fifo_ref_2, fifo_ld_2);
      ReadData(fifo_ref_3, fifo_ld_3);
      dram_blur_y_bank_1_buf(63, 48) = Reinterpret<ap_uint<16>>(fifo_ref_0);
      dram_blur_y_bank_1_buf(47, 32) = Reinterpret<ap_uint<16>>(fifo_ref_1);
      dram_blur_y_bank_1_buf(31, 16) = Reinterpret<ap_uint<16>>(fifo_ref_2);
      dram_blur_y_bank_1_buf(15, 0) = Reinterpret<ap_uint<16>>(fifo_ref_3);
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool fifo_ref_3_enable = ReadData(fifo_ref_3, fifo_ld_3);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable;
      enable = enabled;
      dram_blur_y_bank_1_buf(127, 112) = Reinterpret<ap_uint<16>>(fifo_ref_0);
      dram_blur_y_bank_1_buf(111, 96) = Reinterpret<ap_uint<16>>(fifo_ref_1);
      dram_blur_y_bank_1_buf(95, 80) = Reinterpret<ap_uint<16>>(fifo_ref_2);
      dram_blur_y_bank_1_buf(79, 64) = Reinterpret<ap_uint<16>>(fifo_ref_3);
      WriteData(dram_blur_y_bank_1_fifo, dram_blur_y_bank_1_buf, enabled);
    } // if not empty
  } // for module_8
} // Module8Func

extern "C" {

void blur_kernel(
  ap_uint<128>* bank_1_blur_y, 
  ap_uint<128>* bank_0_input, 
  uint64_t coalesced_data_num)
{
#pragma HLS interface m_axi port = bank_1_blur_y offset = slave bundle = blur_y_bank_1 depth = 131330
#pragma HLS interface m_axi port = bank_0_input offset = slave bundle = input_bank_0 depth = 131330
#pragma HLS interface s_axilite port = bank_1_blur_y bundle = control
#pragma HLS interface s_axilite port = bank_0_input bundle = control
#pragma HLS interface s_axilite port = coalesced_data_num bundle = control
#pragma HLS interface s_axilite port = return bundle = control

  hls::stream<Data<ap_uint<128>>> bank_0_input_buf("bank_0_input_buf");
#pragma HLS stream variable = bank_0_input_buf depth = 32
#pragma HLS aggregate bit variable = bank_0_input_buf
  hls::stream<Data<ap_uint<128>>> bank_1_blur_y_buf("bank_1_blur_y_buf");
#pragma HLS stream variable = bank_1_blur_y_buf depth = 32
#pragma HLS aggregate bit variable = bank_1_blur_y_buf

  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_0("from_input_bank_0_to_input_offset_0");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_0
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_1("from_input_bank_0_to_input_offset_1");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_1
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_2("from_input_bank_0_to_input_offset_2");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_2
  hls::stream<Data<uint16_t>> from_input_bank_0_to_input_offset_3("from_input_bank_0_to_input_offset_3");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_input_bank_0_to_input_offset_3
  hls::stream<Data<uint16_t>> from_input_offset_0_to_input_offset_1024("from_input_offset_0_to_input_offset_1024");
#pragma HLS stream variable = from_input_offset_0_to_input_offset_1024 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_0_to_input_offset_1024
  hls::stream<Data<uint16_t>> from_input_offset_0_to_blur_x_pe_3("from_input_offset_0_to_blur_x_pe_3");
#pragma HLS stream variable = from_input_offset_0_to_blur_x_pe_3 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_0_to_blur_x_pe_3
  hls::stream<Data<uint16_t>> from_input_offset_1_to_input_offset_1025("from_input_offset_1_to_input_offset_1025");
#pragma HLS stream variable = from_input_offset_1_to_input_offset_1025 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_1_to_input_offset_1025
  hls::stream<Data<uint16_t>> from_input_offset_1_to_blur_x_pe_2("from_input_offset_1_to_blur_x_pe_2");
#pragma HLS stream variable = from_input_offset_1_to_blur_x_pe_2 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_1_to_blur_x_pe_2
  hls::stream<Data<uint16_t>> from_input_offset_2_to_input_offset_1026("from_input_offset_2_to_input_offset_1026");
#pragma HLS stream variable = from_input_offset_2_to_input_offset_1026 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2_to_input_offset_1026
  hls::stream<Data<uint16_t>> from_input_offset_2_to_blur_x_pe_1("from_input_offset_2_to_blur_x_pe_1");
#pragma HLS stream variable = from_input_offset_2_to_blur_x_pe_1 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_2_to_blur_x_pe_1
  hls::stream<Data<uint16_t>> from_input_offset_3_to_input_offset_1027("from_input_offset_3_to_input_offset_1027");
#pragma HLS stream variable = from_input_offset_3_to_input_offset_1027 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_3_to_input_offset_1027
  hls::stream<Data<uint16_t>> from_input_offset_3_to_blur_x_pe_0("from_input_offset_3_to_blur_x_pe_0");
#pragma HLS stream variable = from_input_offset_3_to_blur_x_pe_0 depth = 7
#pragma HLS aggregate bit variable = from_input_offset_3_to_blur_x_pe_0
  hls::stream<Data<uint16_t>> from_input_offset_1024_to_input_offset_2048("from_input_offset_1024_to_input_offset_2048");
#pragma HLS stream variable = from_input_offset_1024_to_input_offset_2048 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_1024_to_input_offset_2048
  hls::stream<Data<uint16_t>> from_input_offset_1024_to_blur_x_pe_3("from_input_offset_1024_to_blur_x_pe_3");
#pragma HLS stream variable = from_input_offset_1024_to_blur_x_pe_3 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_1024_to_blur_x_pe_3
  hls::stream<Data<uint16_t>> from_input_offset_1025_to_input_offset_2049("from_input_offset_1025_to_input_offset_2049");
#pragma HLS stream variable = from_input_offset_1025_to_input_offset_2049 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_1025_to_input_offset_2049
  hls::stream<Data<uint16_t>> from_input_offset_1025_to_blur_x_pe_2("from_input_offset_1025_to_blur_x_pe_2");
#pragma HLS stream variable = from_input_offset_1025_to_blur_x_pe_2 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_1025_to_blur_x_pe_2
  hls::stream<Data<uint16_t>> from_input_offset_1026_to_input_offset_2050("from_input_offset_1026_to_input_offset_2050");
#pragma HLS stream variable = from_input_offset_1026_to_input_offset_2050 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_1026_to_input_offset_2050
  hls::stream<Data<uint16_t>> from_input_offset_1026_to_blur_x_pe_1("from_input_offset_1026_to_blur_x_pe_1");
#pragma HLS stream variable = from_input_offset_1026_to_blur_x_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_1026_to_blur_x_pe_1
  hls::stream<Data<uint16_t>> from_input_offset_1027_to_input_offset_2051("from_input_offset_1027_to_input_offset_2051");
#pragma HLS stream variable = from_input_offset_1027_to_input_offset_2051 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_1027_to_input_offset_2051
  hls::stream<Data<uint16_t>> from_input_offset_1027_to_blur_x_pe_0("from_input_offset_1027_to_blur_x_pe_0");
#pragma HLS stream variable = from_input_offset_1027_to_blur_x_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_input_offset_1027_to_blur_x_pe_0
  hls::stream<Data<uint16_t>> from_input_offset_2048_to_blur_x_pe_3("from_input_offset_2048_to_blur_x_pe_3");
#pragma HLS stream variable = from_input_offset_2048_to_blur_x_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2048_to_blur_x_pe_3
  hls::stream<Data<uint16_t>> from_blur_x_pe_3_to_blur_x_offset_0("from_blur_x_pe_3_to_blur_x_offset_0");
#pragma HLS stream variable = from_blur_x_pe_3_to_blur_x_offset_0 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_3_to_blur_x_offset_0
  hls::stream<Data<uint16_t>> from_blur_x_offset_0_to_blur_x_offset_4("from_blur_x_offset_0_to_blur_x_offset_4");
#pragma HLS stream variable = from_blur_x_offset_0_to_blur_x_offset_4 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_0_to_blur_x_offset_4
  hls::stream<Data<uint16_t>> from_blur_x_offset_0_to_blur_y_pe_3("from_blur_x_offset_0_to_blur_y_pe_3");
#pragma HLS stream variable = from_blur_x_offset_0_to_blur_y_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_0_to_blur_y_pe_3
  hls::stream<Data<uint16_t>> from_input_offset_2049_to_blur_x_pe_2("from_input_offset_2049_to_blur_x_pe_2");
#pragma HLS stream variable = from_input_offset_2049_to_blur_x_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2049_to_blur_x_pe_2
  hls::stream<Data<uint16_t>> from_blur_x_pe_2_to_blur_x_offset_1("from_blur_x_pe_2_to_blur_x_offset_1");
#pragma HLS stream variable = from_blur_x_pe_2_to_blur_x_offset_1 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_2_to_blur_x_offset_1
  hls::stream<Data<uint16_t>> from_blur_x_offset_1_to_blur_x_offset_5("from_blur_x_offset_1_to_blur_x_offset_5");
#pragma HLS stream variable = from_blur_x_offset_1_to_blur_x_offset_5 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_1_to_blur_x_offset_5
  hls::stream<Data<uint16_t>> from_blur_x_offset_1_to_blur_y_pe_3("from_blur_x_offset_1_to_blur_y_pe_3");
#pragma HLS stream variable = from_blur_x_offset_1_to_blur_y_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_1_to_blur_y_pe_3
  hls::stream<Data<uint16_t>> from_blur_x_offset_1_to_blur_y_pe_2("from_blur_x_offset_1_to_blur_y_pe_2");
#pragma HLS stream variable = from_blur_x_offset_1_to_blur_y_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_1_to_blur_y_pe_2
  hls::stream<Data<uint16_t>> from_input_offset_2050_to_blur_x_pe_1("from_input_offset_2050_to_blur_x_pe_1");
#pragma HLS stream variable = from_input_offset_2050_to_blur_x_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2050_to_blur_x_pe_1
  hls::stream<Data<uint16_t>> from_blur_x_pe_1_to_blur_x_offset_2("from_blur_x_pe_1_to_blur_x_offset_2");
#pragma HLS stream variable = from_blur_x_pe_1_to_blur_x_offset_2 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_1_to_blur_x_offset_2
  hls::stream<Data<uint16_t>> from_blur_x_offset_2_to_blur_y_pe_3("from_blur_x_offset_2_to_blur_y_pe_3");
#pragma HLS stream variable = from_blur_x_offset_2_to_blur_y_pe_3 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_2_to_blur_y_pe_3
  hls::stream<Data<uint16_t>> from_blur_x_offset_2_to_blur_y_pe_2("from_blur_x_offset_2_to_blur_y_pe_2");
#pragma HLS stream variable = from_blur_x_offset_2_to_blur_y_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_2_to_blur_y_pe_2
  hls::stream<Data<uint16_t>> from_blur_x_offset_2_to_blur_y_pe_1("from_blur_x_offset_2_to_blur_y_pe_1");
#pragma HLS stream variable = from_blur_x_offset_2_to_blur_y_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_x_offset_2_to_blur_y_pe_1
  hls::stream<Data<uint16_t>> from_input_offset_2051_to_blur_x_pe_0("from_input_offset_2051_to_blur_x_pe_0");
#pragma HLS stream variable = from_input_offset_2051_to_blur_x_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_input_offset_2051_to_blur_x_pe_0
  hls::stream<Data<uint16_t>> from_blur_x_pe_0_to_blur_x_offset_3("from_blur_x_pe_0_to_blur_x_offset_3");
#pragma HLS stream variable = from_blur_x_pe_0_to_blur_x_offset_3 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_pe_0_to_blur_x_offset_3
  hls::stream<Data<uint16_t>> from_blur_x_offset_3_to_blur_y_pe_2("from_blur_x_offset_3_to_blur_y_pe_2");
#pragma HLS stream variable = from_blur_x_offset_3_to_blur_y_pe_2 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_3_to_blur_y_pe_2
  hls::stream<Data<uint16_t>> from_blur_x_offset_3_to_blur_y_pe_1("from_blur_x_offset_3_to_blur_y_pe_1");
#pragma HLS stream variable = from_blur_x_offset_3_to_blur_y_pe_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_x_offset_3_to_blur_y_pe_1
  hls::stream<Data<uint16_t>> from_blur_x_offset_3_to_blur_y_pe_0("from_blur_x_offset_3_to_blur_y_pe_0");
#pragma HLS stream variable = from_blur_x_offset_3_to_blur_y_pe_0 depth = 5
#pragma HLS aggregate bit variable = from_blur_x_offset_3_to_blur_y_pe_0
  hls::stream<Data<uint16_t>> from_blur_x_offset_4_to_blur_y_pe_1("from_blur_x_offset_4_to_blur_y_pe_1");
#pragma HLS stream variable = from_blur_x_offset_4_to_blur_y_pe_1 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_4_to_blur_y_pe_1
  hls::stream<Data<uint16_t>> from_blur_x_offset_4_to_blur_y_pe_0("from_blur_x_offset_4_to_blur_y_pe_0");
#pragma HLS stream variable = from_blur_x_offset_4_to_blur_y_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_4_to_blur_y_pe_0
  hls::stream<Data<uint16_t>> from_blur_y_pe_3_to_blur_y_bank_1("from_blur_y_pe_3_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_3_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_3_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_x_offset_5_to_blur_y_pe_0("from_blur_x_offset_5_to_blur_y_pe_0");
#pragma HLS stream variable = from_blur_x_offset_5_to_blur_y_pe_0 depth = 3
#pragma HLS aggregate bit variable = from_blur_x_offset_5_to_blur_y_pe_0
  hls::stream<Data<uint16_t>> from_blur_y_pe_2_to_blur_y_bank_1("from_blur_y_pe_2_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_2_to_blur_y_bank_1 depth = 5
#pragma HLS aggregate bit variable = from_blur_y_pe_2_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_1_to_blur_y_bank_1("from_blur_y_pe_1_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_1_to_blur_y_bank_1 depth = 3
#pragma HLS aggregate bit variable = from_blur_y_pe_1_to_blur_y_bank_1
  hls::stream<Data<uint16_t>> from_blur_y_pe_0_to_blur_y_bank_1("from_blur_y_pe_0_to_blur_y_bank_1");
#pragma HLS stream variable = from_blur_y_pe_0_to_blur_y_bank_1 depth = 3
#pragma HLS aggregate bit variable = from_blur_y_pe_0_to_blur_y_bank_1

#pragma HLS dataflow
  BurstRead(bank_0_input_buf, bank_0_input, coalesced_data_num);
  Module0Func(
      /*output*/ from_input_bank_0_to_input_offset_0, 
      /*output*/ from_input_bank_0_to_input_offset_1, 
      /*output*/ from_input_bank_0_to_input_offset_2, 
      /*output*/ from_input_bank_0_to_input_offset_3, 
      /* input*/ bank_0_input_buf);
  Module1Func(
      /*output*/ from_input_offset_0_to_input_offset_1024, 
      /*output*/ from_input_offset_0_to_blur_x_pe_3, 
      /* input*/ from_input_bank_0_to_input_offset_0);
  Module1Func(
      /*output*/ from_input_offset_1_to_input_offset_1025, 
      /*output*/ from_input_offset_1_to_blur_x_pe_2, 
      /* input*/ from_input_bank_0_to_input_offset_1);
  Module1Func(
      /*output*/ from_input_offset_2_to_input_offset_1026, 
      /*output*/ from_input_offset_2_to_blur_x_pe_1, 
      /* input*/ from_input_bank_0_to_input_offset_2);
  Module1Func(
      /*output*/ from_input_offset_3_to_input_offset_1027, 
      /*output*/ from_input_offset_3_to_blur_x_pe_0, 
      /* input*/ from_input_bank_0_to_input_offset_3);
  Module2Func(
      /*output*/ from_input_offset_1024_to_input_offset_2048, 
      /*output*/ from_input_offset_1024_to_blur_x_pe_3, 
      /* input*/ from_input_offset_0_to_input_offset_1024);
  Module2Func(
      /*output*/ from_input_offset_1025_to_input_offset_2049, 
      /*output*/ from_input_offset_1025_to_blur_x_pe_2, 
      /* input*/ from_input_offset_1_to_input_offset_1025);
  Module2Func(
      /*output*/ from_input_offset_1026_to_input_offset_2050, 
      /*output*/ from_input_offset_1026_to_blur_x_pe_1, 
      /* input*/ from_input_offset_2_to_input_offset_1026);
  Module2Func(
      /*output*/ from_input_offset_1027_to_input_offset_2051, 
      /*output*/ from_input_offset_1027_to_blur_x_pe_0, 
      /* input*/ from_input_offset_3_to_input_offset_1027);
  Module3Func(
      /*output*/ from_input_offset_2048_to_blur_x_pe_3, 
      /* input*/ from_input_offset_1024_to_input_offset_2048);
  Module4Func(
      /*output*/ from_blur_x_pe_3_to_blur_x_offset_0, 
      /* input*/ from_input_offset_2048_to_blur_x_pe_3, 
      /* input*/ from_input_offset_1024_to_blur_x_pe_3, 
      /* input*/ from_input_offset_0_to_blur_x_pe_3);
  Module1Func(
      /*output*/ from_blur_x_offset_0_to_blur_x_offset_4, 
      /*output*/ from_blur_x_offset_0_to_blur_y_pe_3, 
      /* input*/ from_blur_x_pe_3_to_blur_x_offset_0);
  Module3Func(
      /*output*/ from_input_offset_2049_to_blur_x_pe_2, 
      /* input*/ from_input_offset_1025_to_input_offset_2049);
  Module4Func(
      /*output*/ from_blur_x_pe_2_to_blur_x_offset_1, 
      /* input*/ from_input_offset_2049_to_blur_x_pe_2, 
      /* input*/ from_input_offset_1025_to_blur_x_pe_2, 
      /* input*/ from_input_offset_1_to_blur_x_pe_2);
  Module5Func(
      /*output*/ from_blur_x_offset_1_to_blur_x_offset_5, 
      /*output*/ from_blur_x_offset_1_to_blur_y_pe_3, 
      /*output*/ from_blur_x_offset_1_to_blur_y_pe_2, 
      /* input*/ from_blur_x_pe_2_to_blur_x_offset_1);
  Module3Func(
      /*output*/ from_input_offset_2050_to_blur_x_pe_1, 
      /* input*/ from_input_offset_1026_to_input_offset_2050);
  Module4Func(
      /*output*/ from_blur_x_pe_1_to_blur_x_offset_2, 
      /* input*/ from_input_offset_2050_to_blur_x_pe_1, 
      /* input*/ from_input_offset_1026_to_blur_x_pe_1, 
      /* input*/ from_input_offset_2_to_blur_x_pe_1);
  Module5Func(
      /*output*/ from_blur_x_offset_2_to_blur_y_pe_3, 
      /*output*/ from_blur_x_offset_2_to_blur_y_pe_2, 
      /*output*/ from_blur_x_offset_2_to_blur_y_pe_1, 
      /* input*/ from_blur_x_pe_1_to_blur_x_offset_2);
  Module3Func(
      /*output*/ from_input_offset_2051_to_blur_x_pe_0, 
      /* input*/ from_input_offset_1027_to_input_offset_2051);
  Module4Func(
      /*output*/ from_blur_x_pe_0_to_blur_x_offset_3, 
      /* input*/ from_input_offset_2051_to_blur_x_pe_0, 
      /* input*/ from_input_offset_1027_to_blur_x_pe_0, 
      /* input*/ from_input_offset_3_to_blur_x_pe_0);
  Module5Func(
      /*output*/ from_blur_x_offset_3_to_blur_y_pe_2, 
      /*output*/ from_blur_x_offset_3_to_blur_y_pe_1, 
      /*output*/ from_blur_x_offset_3_to_blur_y_pe_0, 
      /* input*/ from_blur_x_pe_0_to_blur_x_offset_3);
  Module6Func(
      /*output*/ from_blur_x_offset_4_to_blur_y_pe_1, 
      /*output*/ from_blur_x_offset_4_to_blur_y_pe_0, 
      /* input*/ from_blur_x_offset_0_to_blur_x_offset_4);
  Module4Func(
      /*output*/ from_blur_y_pe_3_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_2_to_blur_y_pe_3, 
      /* input*/ from_blur_x_offset_1_to_blur_y_pe_3, 
      /* input*/ from_blur_x_offset_0_to_blur_y_pe_3);
  Module7Func(
      /*output*/ from_blur_x_offset_5_to_blur_y_pe_0, 
      /* input*/ from_blur_x_offset_1_to_blur_x_offset_5);
  Module4Func(
      /*output*/ from_blur_y_pe_2_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_3_to_blur_y_pe_2, 
      /* input*/ from_blur_x_offset_2_to_blur_y_pe_2, 
      /* input*/ from_blur_x_offset_1_to_blur_y_pe_2);
  Module4Func(
      /*output*/ from_blur_y_pe_1_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_4_to_blur_y_pe_1, 
      /* input*/ from_blur_x_offset_3_to_blur_y_pe_1, 
      /* input*/ from_blur_x_offset_2_to_blur_y_pe_1);
  Module4Func(
      /*output*/ from_blur_y_pe_0_to_blur_y_bank_1, 
      /* input*/ from_blur_x_offset_5_to_blur_y_pe_0, 
      /* input*/ from_blur_x_offset_4_to_blur_y_pe_0, 
      /* input*/ from_blur_x_offset_3_to_blur_y_pe_0);
  Module8Func(
      /*output*/ bank_1_blur_y_buf, 
      /* input*/ from_blur_y_pe_3_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_2_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_1_to_blur_y_bank_1, 
      /* input*/ from_blur_y_pe_0_to_blur_y_bank_1);
  BurstWrite(bank_1_blur_y, bank_1_blur_y_buf, coalesced_data_num);
}

}  // extern "C"
