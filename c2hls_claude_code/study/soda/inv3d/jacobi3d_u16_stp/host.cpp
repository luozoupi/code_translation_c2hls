// this file can be generated from the following SODA DSL
/*
kernel: jacobi3d
burst width: 512
iterate: 6
unroll factor: 16
input dram 0 float: t1(32, 32, *)
output dram 1 float: t0(0, 0, 0) = (t1(0, 0, 0) + t1(1, 0, 0) + t1(-1, 0, 0) + t1(0, 1, 0) + t1(0, -1, 0) + t1(0, 0, 1) + t1(0, 0, -1)) * 0.142857142f
border: ignore
cluster: none
*/

// stencil window size: (13, 13, 13)
// stencil distace: 197382
// data layout is documented at
// https://github.com/Blaok/soda/blob/master/docs/data-layout.md

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdbool>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include <frt.h>
#include <ap_int.h>

using std::clog;
using std::endl;
using std::regex;
using std::regex_match;
using std::string;
using std::unique_ptr;
using std::unordered_map;
using std::vector;

// C++ binding
extern "C" {
void jacobi3d_kernel(
  ap_uint<512>* bank_1_t0,
  ap_uint<512>* bank_0_t1,
  uint64_t coalesced_data_num);
}  // extern "C"

namespace soda {
namespace app {
// app-specific constants
constexpr int kStencilDim0 = 13;
constexpr int kStencilDim1 = 13;
constexpr int kStencilDim2 = 13;
constexpr int kStencilDistance = 197382;
constexpr int kWidth_t1 = 32;
constexpr int kWidth_t0 = 32;

// type alias
using Type_t1 = float;
using Type_t0 = float;

int jacobi3d(
  const Type_t1* var_t1_ptr, 
  const int32_t var_t1_extent[3], 
  const int32_t var_t1_stride[3], 
  const int32_t var_t1_min[3], 
  Type_t0* var_t0_ptr, 
  const int32_t var_t0_extent[3], 
  const int32_t var_t0_stride[3], 
  const int32_t var_t0_min[3], 
  const char* bitstream, 
  const int burst_width = 512, 
  const int tile_size_0 = 128, 
  const int tile_size_1 = 128, 
  const int unroll_factor = 16)
{
#ifdef SODA_CPP_BINDING
  // C++ binding only supports a fixed number of banks
  int bank_count_t1 = 1;
  int bank_count_t0 = 1;
#else  // SODA_CPP_BINDING
  // load bitstream
  auto instance = fpga::Instance(bitstream);
  auto args_info = instance.GetArgsInfo();
  // find out how many banks are used for each tensor
  int bank_count_t1 = 0;
  int bank_count_t0 = 0;
  const regex regex_t1(R"(^bank_\d+_t1$)");
  const regex regex_t0(R"(^bank_\d+_t0$)");
  for (const auto& arg : args_info) {
    if (regex_match(arg.name, regex_t1)) ++bank_count_t1;
    if (regex_match(arg.name, regex_t0)) ++bank_count_t0;
  }
#endif  // SODA_CPP_BINDING
  auto round_up = [](int64_t a, int64_t b) -> int64_t { return ((a - 1) / b + 1) * b; };

  // some run-time constants
  const int elem_count_per_cycle_t1 = burst_width / kWidth_t1 * bank_count_t1;
  const int elem_count_per_cycle_t0 = burst_width / kWidth_t0 * bank_count_t0;
  int32_t tile_count_dim_0 = (var_t1_extent[0] - kStencilDim0 + 1 - 1) / (tile_size_0 - kStencilDim0 + 1) + 1;
  int32_t tile_count_dim_1 = (var_t1_extent[1] - kStencilDim1 + 1 - 1) / (tile_size_1 - kStencilDim1 + 1) + 1;
  int64_t tile_count = tile_count_dim_0 * tile_count_dim_1;

  // align each linearized tile to multiples of burst_width
  int64_t elem_count_per_tile = tile_size_0 * tile_size_1 * var_t1_extent[2];
  int64_t cycle_count_per_tile = (elem_count_per_tile - 1) / elem_count_per_cycle_t1 + 1;
  int64_t elem_count_aligned_per_tile_i = cycle_count_per_tile * elem_count_per_cycle_t1;
  int64_t elem_count_aligned_per_tile_o = cycle_count_per_tile * elem_count_per_cycle_t0;

  // calculate size of each buffer
  int64_t buf_size_t1 = (tile_count * elem_count_aligned_per_tile_i + round_up(kStencilDistance, elem_count_per_cycle_t1)) / bank_count_t1 * sizeof(Type_t1);
  int64_t buf_size_t0 = (tile_count * elem_count_aligned_per_tile_o + round_up(kStencilDistance, elem_count_per_cycle_t0)) / bank_count_t0 * sizeof(Type_t0);

  // allocate memory for each buffer
  vector<unique_ptr<Type_t1, decltype(&free)>> buf_t1;
  buf_t1.reserve(bank_count_t1);
  for (int bank = 0; bank < bank_count_t1; ++bank) {
    buf_t1.emplace_back(static_cast<Type_t1*>(aligned_alloc(4096, round_up(buf_size_t1 + (1 << 20), 4096))), &free);
  }
  vector<unique_ptr<Type_t0, decltype(&free)>> buf_t0;
  buf_t0.reserve(bank_count_t0);
  for (int bank = 0; bank < bank_count_t0; ++bank) {
    buf_t0.emplace_back(static_cast<Type_t0*>(aligned_alloc(4096, round_up(buf_size_t0 + (1 << 20), 4096))), &free);
  }

  // tiling
  for(int32_t tile_index_dim_1 = 0; tile_index_dim_1 < tile_count_dim_1; ++tile_index_dim_1)
  {
    int32_t actual_tile_size_dim_1 = (tile_index_dim_1==tile_count_dim_1-1) ? var_t1_extent[1] - (tile_size_1 - kStencilDim1 + 1) * tile_index_dim_1 : tile_size_1;
    for(int32_t tile_index_dim_0 = 0; tile_index_dim_0 < tile_count_dim_0; ++tile_index_dim_0)
    {
      int32_t actual_tile_size_dim_0 = (tile_index_dim_0==tile_count_dim_0-1) ? var_t1_extent[0] - (tile_size_0 - kStencilDim0 + 1) * tile_index_dim_0 : tile_size_0;
#pragma omp parallel for
      for(int32_t k = 0; k < var_t1_extent[2]; ++k)
      {
        for(int32_t j = 0; j < actual_tile_size_dim_1; ++j)
        {
          for(int32_t i = 0; i < actual_tile_size_dim_0; ++i)
          {
            // (x, y, z, w) is coordinates in tiled image
            // (p, q, r, s) is coordinates in original image
            // (i, j, k, l) is coordinates in a tile
            int32_t burst_index = (i + j * tile_size_0 + k * tile_size_0 * tile_size_1) / elem_count_per_cycle_t1;
            int32_t burst_residue = (i + j * tile_size_0 + k * tile_size_0 * tile_size_1) % elem_count_per_cycle_t1;
            int32_t p = tile_index_dim_0 * (tile_size_0 - kStencilDim0) + i;
            int32_t q = tile_index_dim_1 * (tile_size_1 - kStencilDim1) + j;
            int32_t r = k;
            int64_t tiled_offset = (tile_index_dim_0 + tile_count_dim_0 * tile_index_dim_1) * elem_count_aligned_per_tile_i + burst_index * elem_count_per_cycle_t1 + burst_residue;
            int64_t original_offset = p * var_t1_stride[0] + q * var_t1_stride[1] + r * var_t1_stride[2];
            buf_t1[tiled_offset % bank_count_t1].get()[tiled_offset / bank_count_t1] = var_t1_ptr[std::max(int64_t(0), original_offset - 0)];
          }
        }
      }
    }
  }

  clog << "INFO: tile_count[0] = " << tile_count_dim_0 << ", tile_size[0] = " << tile_size_0 << endl;
  clog << "INFO: tile_count[1] = " << tile_count_dim_1 << ", tile_size[1] = " << tile_size_1 << endl;
  clog << "INFO: t1.extent[0] = " << var_t1_extent[0] << ", t1.extent[1] = " << var_t1_extent[1] << ", t1.extent[2] = " << var_t1_extent[2] << "" << endl;
  clog << "INFO: t1.stride[0] = " << var_t1_stride[0] << ", t1.stride[1] = " << var_t1_stride[1] << ", t1.stride[2] = " << var_t1_stride[2] << "" << endl;
  clog << "INFO: t1.min[0] = " << var_t1_min[0] << ", t1.min[1] = " << var_t1_min[1] << ", t1.min[2] = " << var_t1_min[2] << "" << endl;
  clog << "INFO: t0.extent[0] = " << var_t0_extent[0] << ", t0.extent[1] = " << var_t0_extent[1] << ", t0.extent[2] = " << var_t0_extent[2] << "" << endl;
  clog << "INFO: t0.stride[0] = " << var_t0_stride[0] << ", t0.stride[1] = " << var_t0_stride[1] << ", t0.stride[2] = " << var_t0_stride[2] << "" << endl;
  clog << "INFO: t0.min[0] = " << var_t0_min[0] << ", t0.min[1] = " << var_t0_min[1] << ", t0.min[2] = " << var_t0_min[2] << "" << endl;

  int64_t tile_data_count = ((int64_t(var_t1_extent[2]) * tile_size_0 * tile_size_1 - 1) / elem_count_per_cycle_t1 + 1) * elem_count_per_cycle_t1 / unroll_factor;
  int64_t cycle_count = ((int64_t(var_t1_extent[2]) * tile_size_0 * tile_size_1 * tile_count_dim_0 * tile_count_dim_1 + kStencilDistance - 1) / elem_count_per_cycle_t1 + 1);
  clog << "INFO: tile_data_count = " << tile_data_count << ", cycle_count = " << cycle_count << endl;

#ifdef SODA_CPP_BINDING
  ::jacobi3d_kernel(
    (ap_uint<512>*)buf_t0[0].get(), 
    (ap_uint<512>*)buf_t1[0].get(), 
    cycle_count);

#else  // SODA_CPP_BINDING
  int arg_idx = 0;
  auto iter_t1 = buf_t1.begin();
  auto iter_t0 = buf_t0.begin();
  for (const auto& arg : args_info) {
    if (arg.name == "coalesced_data_num") {
      instance.SetArg(arg_idx, cycle_count);
      ++arg_idx;
    } else if (regex_match(arg.name, regex_t1)) {
      auto buf = fpga::WriteOnly(iter_t1->get(), buf_size_t1 / sizeof(Type_t1));
      instance.AllocBuf(arg_idx, buf);
      instance.SetArg(arg_idx, buf);
      ++iter_t1;
      ++arg_idx;
    } else if (regex_match(arg.name, regex_t0)) {
      auto buf = fpga::ReadOnly(iter_t0->get(), buf_size_t0 / sizeof(Type_t0));
      instance.AllocBuf(arg_idx, buf);
      instance.SetArg(arg_idx, buf);
      ++iter_t0;
      ++arg_idx;
    }
  }

  instance.WriteToDevice();
  instance.Exec();
  instance.ReadFromDevice();
  instance.Finish();

  clog << "Load throughput: " << std::setprecision(3) << instance.LoadThroughputGbps() << " GB/s" << endl;
  clog << "Compute latency: " << std::setprecision(3) << instance.ComputeTimeSeconds() << " s" << endl;
  clog << "Compute throughput (t1): " << std::setprecision(3) << var_t1_extent[0] * var_t1_extent[1] * var_t1_extent[2] / instance.ComputeTimeSeconds() * 1e-9 << " pixel/ns" << endl;
  clog << "Compute throughput (t0): " << std::setprecision(3) << var_t0_extent[0] * var_t0_extent[1] * var_t0_extent[2] / instance.ComputeTimeSeconds() * 1e-9 << " pixel/ns" << endl;
  clog << "Store throughput: " << std::setprecision(3) << instance.StoreThroughputGbps() <<" GB/s" << endl;
#endif  // SODA_CPP_BINDING

  for(int32_t tile_index_dim_1 = 0; tile_index_dim_1 < tile_count_dim_1; ++tile_index_dim_1)
  {
    int32_t actual_tile_size_dim_1 = (tile_index_dim_1 == tile_count_dim_1-1) ? var_t1_extent[1] - (tile_size_1 - kStencilDim1 + 1) * tile_index_dim_1 : tile_size_1;
    for(int32_t tile_index_dim_0 = 0; tile_index_dim_0 < tile_count_dim_0; ++tile_index_dim_0)
    {
      int32_t actual_tile_size_dim_0 = (tile_index_dim_0 == tile_count_dim_0-1) ? var_t1_extent[0] - (tile_size_0 - kStencilDim0 + 1) * tile_index_dim_0 : tile_size_0;
#pragma omp parallel for
      for(int32_t k = 6; k < var_t0_extent[2] - 6; ++k)
      {
        for(int32_t j = 6; j < actual_tile_size_dim_1 - 6; ++j)
        {
          for(int32_t i = 6; i < actual_tile_size_dim_0 - 6; ++i)
          {
            // (x, y, z, w) is coordinates in tiled image
            // (p, q, r, s) is coordinates in original image
            // (i, j, k, l) is coordinates in a tile
            int32_t p = tile_index_dim_0 * (tile_size_0-kStencilDim0 + 1) + i;
            int32_t q = tile_index_dim_1 * (tile_size_1-kStencilDim1 + 1) + j;
            int32_t r = k;
            int64_t original_offset = p * var_t0_stride[0] + q * var_t0_stride[1] + r * var_t0_stride[2];
            int32_t burst_index_t0 = (i + j * tile_size_0 + k * tile_size_0 * tile_size_1 + 98304) / elem_count_per_cycle_t0;
            int32_t burst_residue_t0 = (i + j * tile_size_0 + k * tile_size_0 * tile_size_1 + 98304) % elem_count_per_cycle_t0;
            int64_t tiled_offset_t0 = (tile_index_dim_0+tile_count_dim_0 * tile_index_dim_1) * elem_count_aligned_per_tile_o + burst_index_t0 * elem_count_per_cycle_t0 + burst_residue_t0;
            var_t0_ptr[original_offset] = buf_t0[tiled_offset_t0 % bank_count_t0].get()[tiled_offset_t0 / bank_count_t0];
          }
        }
      }
    }
  }
  return 0;
}

} // namespace app
} // namespace soda

#ifdef SODA_TEST_MAIN

#include <ap_int.h>

int main(int argc, char *argv[])
{
  if (argc < 2 || argc > 5) {
    clog << "Usage: \n  " << argv[0] << " <xclbin> [input dimension 0] [input dimension 1] [input dimension 2]" << endl;
    return 2;
  }
  const char* xclbin = argv[1];
  int32_t dims[3] = {128, 128, soda::app::kStencilDim2 + 1};
  if (argc > 2) {
    dims[0] = std::max(dims[0], atoi(argv[2]));
  }
  if (argc > 3) {
    dims[1] = std::max(dims[1], atoi(argv[3]));
  }
  if (argc > 4) {
    dims[2] = std::max(dims[2], atoi(argv[4]));
  }

  vector<float> data_t1(dims[0] * dims[1] * dims[2]);
  int32_t extent_t1[3] = {dims[0], dims[1], dims[2]};
  int32_t stride_t1[3] = {1, dims[0], dims[0] * dims[1]};
  int32_t min_t1[3] = {0, 0, 0};
  vector<float> data_t1_iter1(dims[0] * dims[1] * dims[2]);
  int32_t extent_t1_iter1[3] = {dims[0], dims[1], dims[2]};
  int32_t stride_t1_iter1[3] = {1, dims[0], dims[0] * dims[1]};
  int32_t min_t1_iter1[3] = {0, 0, 0};
  vector<float> data_t1_iter2(dims[0] * dims[1] * dims[2]);
  int32_t extent_t1_iter2[3] = {dims[0], dims[1], dims[2]};
  int32_t stride_t1_iter2[3] = {1, dims[0], dims[0] * dims[1]};
  int32_t min_t1_iter2[3] = {0, 0, 0};
  vector<float> data_t1_iter3(dims[0] * dims[1] * dims[2]);
  int32_t extent_t1_iter3[3] = {dims[0], dims[1], dims[2]};
  int32_t stride_t1_iter3[3] = {1, dims[0], dims[0] * dims[1]};
  int32_t min_t1_iter3[3] = {0, 0, 0};
  vector<float> data_t1_iter4(dims[0] * dims[1] * dims[2]);
  int32_t extent_t1_iter4[3] = {dims[0], dims[1], dims[2]};
  int32_t stride_t1_iter4[3] = {1, dims[0], dims[0] * dims[1]};
  int32_t min_t1_iter4[3] = {0, 0, 0};
  vector<float> data_t1_iter5(dims[0] * dims[1] * dims[2]);
  int32_t extent_t1_iter5[3] = {dims[0], dims[1], dims[2]};
  int32_t stride_t1_iter5[3] = {1, dims[0], dims[0] * dims[1]};
  int32_t min_t1_iter5[3] = {0, 0, 0};
  vector<float> data_t0(dims[0] * dims[1] * dims[2]);
  int32_t extent_t0[3] = {dims[0], dims[1], dims[2]};
  int32_t stride_t0[3] = {1, dims[0], dims[0] * dims[1]};
  int32_t min_t0[3] = {0, 0, 0};

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  // initialization can be parallelized with -fopenmp
#pragma omp parallel for
  for(int32_t r = 0; r < dims[2]; ++r)
  {
    for(int32_t q = 0; q < dims[1]; ++q)
    {
      for(int32_t p = 0; p < dims[0]; ++p)
      {
        data_t1[p * stride_t1[0] + q * stride_t1[1] + r * stride_t1[2]] = distribution(generator);
      }
    }
  }

  soda::app::jacobi3d(
    data_t1.data(), 
    extent_t1, 
    stride_t1, 
    min_t1, 
    data_t0.data(), 
    extent_t0, 
    stride_t0, 
    min_t0, 
    xclbin);

  int error_count = 0;

  // produce t1_iter1, can be parallelized with -fopenmp
#pragma omp parallel for
  for(int32_t r = 1; r < dims[2] - 1; ++r)
  {
    for(int32_t q = 1; q < dims[1] - 1; ++q)
    {
      for(int32_t p = 1; p < dims[0] - 1; ++p)
      {
        // t1_iter1(0, 0, 0) = ((t1(0, 0, 0) + t1(1, 0, 0) + t1(-1, 0, 0) + t1(0, 1, 0) + t1(0, -1, 0) + t1(0, 0, 1) + t1(0, 0, -1)) * 0.142857142f)
        data_t1_iter1[p*stride_t1_iter1[0]+q*stride_t1_iter1[1]+r*stride_t1_iter1[2]] = ((data_t1[(p+0)*stride_t1[0] + (q+0)*stride_t1[1] + (r+0)*stride_t1[2]] + data_t1[(p+1)*stride_t1[0] + (q+0)*stride_t1[1] + (r+0)*stride_t1[2]] + data_t1[(p-1)*stride_t1[0] + (q+0)*stride_t1[1] + (r+0)*stride_t1[2]] + data_t1[(p+0)*stride_t1[0] + (q+1)*stride_t1[1] + (r+0)*stride_t1[2]] + data_t1[(p+0)*stride_t1[0] + (q-1)*stride_t1[1] + (r+0)*stride_t1[2]] + data_t1[(p+0)*stride_t1[0] + (q+0)*stride_t1[1] + (r+1)*stride_t1[2]] + data_t1[(p+0)*stride_t1[0] + (q+0)*stride_t1[1] + (r-1)*stride_t1[2]]) * 0.142857142f);
      }
    }
  }

  // produce t1_iter2, can be parallelized with -fopenmp
#pragma omp parallel for
  for(int32_t r = 2; r < dims[2] - 2; ++r)
  {
    for(int32_t q = 2; q < dims[1] - 2; ++q)
    {
      for(int32_t p = 2; p < dims[0] - 2; ++p)
      {
        // t1_iter2(0, 0, 0) = ((t1_iter1(0, 0, 0) + t1_iter1(1, 0, 0) + t1_iter1(-1, 0, 0) + t1_iter1(0, 1, 0) + t1_iter1(0, -1, 0) + t1_iter1(0, 0, 1) + t1_iter1(0, 0, -1)) * 0.142857142f)
        data_t1_iter2[p*stride_t1_iter2[0]+q*stride_t1_iter2[1]+r*stride_t1_iter2[2]] = ((data_t1_iter1[(p+0)*stride_t1_iter1[0] + (q+0)*stride_t1_iter1[1] + (r+0)*stride_t1_iter1[2]] + data_t1_iter1[(p+1)*stride_t1_iter1[0] + (q+0)*stride_t1_iter1[1] + (r+0)*stride_t1_iter1[2]] + data_t1_iter1[(p-1)*stride_t1_iter1[0] + (q+0)*stride_t1_iter1[1] + (r+0)*stride_t1_iter1[2]] + data_t1_iter1[(p+0)*stride_t1_iter1[0] + (q+1)*stride_t1_iter1[1] + (r+0)*stride_t1_iter1[2]] + data_t1_iter1[(p+0)*stride_t1_iter1[0] + (q-1)*stride_t1_iter1[1] + (r+0)*stride_t1_iter1[2]] + data_t1_iter1[(p+0)*stride_t1_iter1[0] + (q+0)*stride_t1_iter1[1] + (r+1)*stride_t1_iter1[2]] + data_t1_iter1[(p+0)*stride_t1_iter1[0] + (q+0)*stride_t1_iter1[1] + (r-1)*stride_t1_iter1[2]]) * 0.142857142f);
      }
    }
  }

  // produce t1_iter3, can be parallelized with -fopenmp
#pragma omp parallel for
  for(int32_t r = 3; r < dims[2] - 3; ++r)
  {
    for(int32_t q = 3; q < dims[1] - 3; ++q)
    {
      for(int32_t p = 3; p < dims[0] - 3; ++p)
      {
        // t1_iter3(0, 0, 0) = ((t1_iter2(0, 0, 0) + t1_iter2(1, 0, 0) + t1_iter2(-1, 0, 0) + t1_iter2(0, 1, 0) + t1_iter2(0, -1, 0) + t1_iter2(0, 0, 1) + t1_iter2(0, 0, -1)) * 0.142857142f)
        data_t1_iter3[p*stride_t1_iter3[0]+q*stride_t1_iter3[1]+r*stride_t1_iter3[2]] = ((data_t1_iter2[(p+0)*stride_t1_iter2[0] + (q+0)*stride_t1_iter2[1] + (r+0)*stride_t1_iter2[2]] + data_t1_iter2[(p+1)*stride_t1_iter2[0] + (q+0)*stride_t1_iter2[1] + (r+0)*stride_t1_iter2[2]] + data_t1_iter2[(p-1)*stride_t1_iter2[0] + (q+0)*stride_t1_iter2[1] + (r+0)*stride_t1_iter2[2]] + data_t1_iter2[(p+0)*stride_t1_iter2[0] + (q+1)*stride_t1_iter2[1] + (r+0)*stride_t1_iter2[2]] + data_t1_iter2[(p+0)*stride_t1_iter2[0] + (q-1)*stride_t1_iter2[1] + (r+0)*stride_t1_iter2[2]] + data_t1_iter2[(p+0)*stride_t1_iter2[0] + (q+0)*stride_t1_iter2[1] + (r+1)*stride_t1_iter2[2]] + data_t1_iter2[(p+0)*stride_t1_iter2[0] + (q+0)*stride_t1_iter2[1] + (r-1)*stride_t1_iter2[2]]) * 0.142857142f);
      }
    }
  }

  // produce t1_iter4, can be parallelized with -fopenmp
#pragma omp parallel for
  for(int32_t r = 4; r < dims[2] - 4; ++r)
  {
    for(int32_t q = 4; q < dims[1] - 4; ++q)
    {
      for(int32_t p = 4; p < dims[0] - 4; ++p)
      {
        // t1_iter4(0, 0, 0) = ((t1_iter3(0, 0, 0) + t1_iter3(1, 0, 0) + t1_iter3(-1, 0, 0) + t1_iter3(0, 1, 0) + t1_iter3(0, -1, 0) + t1_iter3(0, 0, 1) + t1_iter3(0, 0, -1)) * 0.142857142f)
        data_t1_iter4[p*stride_t1_iter4[0]+q*stride_t1_iter4[1]+r*stride_t1_iter4[2]] = ((data_t1_iter3[(p+0)*stride_t1_iter3[0] + (q+0)*stride_t1_iter3[1] + (r+0)*stride_t1_iter3[2]] + data_t1_iter3[(p+1)*stride_t1_iter3[0] + (q+0)*stride_t1_iter3[1] + (r+0)*stride_t1_iter3[2]] + data_t1_iter3[(p-1)*stride_t1_iter3[0] + (q+0)*stride_t1_iter3[1] + (r+0)*stride_t1_iter3[2]] + data_t1_iter3[(p+0)*stride_t1_iter3[0] + (q+1)*stride_t1_iter3[1] + (r+0)*stride_t1_iter3[2]] + data_t1_iter3[(p+0)*stride_t1_iter3[0] + (q-1)*stride_t1_iter3[1] + (r+0)*stride_t1_iter3[2]] + data_t1_iter3[(p+0)*stride_t1_iter3[0] + (q+0)*stride_t1_iter3[1] + (r+1)*stride_t1_iter3[2]] + data_t1_iter3[(p+0)*stride_t1_iter3[0] + (q+0)*stride_t1_iter3[1] + (r-1)*stride_t1_iter3[2]]) * 0.142857142f);
      }
    }
  }

  // produce t1_iter5, can be parallelized with -fopenmp
#pragma omp parallel for
  for(int32_t r = 5; r < dims[2] - 5; ++r)
  {
    for(int32_t q = 5; q < dims[1] - 5; ++q)
    {
      for(int32_t p = 5; p < dims[0] - 5; ++p)
      {
        // t1_iter5(0, 0, 0) = ((t1_iter4(0, 0, 0) + t1_iter4(1, 0, 0) + t1_iter4(-1, 0, 0) + t1_iter4(0, 1, 0) + t1_iter4(0, -1, 0) + t1_iter4(0, 0, 1) + t1_iter4(0, 0, -1)) * 0.142857142f)
        data_t1_iter5[p*stride_t1_iter5[0]+q*stride_t1_iter5[1]+r*stride_t1_iter5[2]] = ((data_t1_iter4[(p+0)*stride_t1_iter4[0] + (q+0)*stride_t1_iter4[1] + (r+0)*stride_t1_iter4[2]] + data_t1_iter4[(p+1)*stride_t1_iter4[0] + (q+0)*stride_t1_iter4[1] + (r+0)*stride_t1_iter4[2]] + data_t1_iter4[(p-1)*stride_t1_iter4[0] + (q+0)*stride_t1_iter4[1] + (r+0)*stride_t1_iter4[2]] + data_t1_iter4[(p+0)*stride_t1_iter4[0] + (q+1)*stride_t1_iter4[1] + (r+0)*stride_t1_iter4[2]] + data_t1_iter4[(p+0)*stride_t1_iter4[0] + (q-1)*stride_t1_iter4[1] + (r+0)*stride_t1_iter4[2]] + data_t1_iter4[(p+0)*stride_t1_iter4[0] + (q+0)*stride_t1_iter4[1] + (r+1)*stride_t1_iter4[2]] + data_t1_iter4[(p+0)*stride_t1_iter4[0] + (q+0)*stride_t1_iter4[1] + (r-1)*stride_t1_iter4[2]]) * 0.142857142f);
      }
    }
  }

  // produce t0, can be parallelized with -fopenmp
#pragma omp parallel for
  for(int32_t r = 6; r < dims[2] - 6; ++r)
  {
    for(int32_t q = 6; q < dims[1] - 6; ++q)
    {
      for(int32_t p = 6; p < dims[0] - 6; ++p)
      {
        // t0(0, 0, 0) = ((t1_iter5(0, 0, 0) + t1_iter5(1, 0, 0) + t1_iter5(-1, 0, 0) + t1_iter5(0, 1, 0) + t1_iter5(0, -1, 0) + t1_iter5(0, 0, 1) + t1_iter5(0, 0, -1)) * 0.142857142f)
        float result_t0 = ((data_t1_iter5[(p+0)*stride_t1_iter5[0] + (q+0)*stride_t1_iter5[1] + (r+0)*stride_t1_iter5[2]] + data_t1_iter5[(p+1)*stride_t1_iter5[0] + (q+0)*stride_t1_iter5[1] + (r+0)*stride_t1_iter5[2]] + data_t1_iter5[(p-1)*stride_t1_iter5[0] + (q+0)*stride_t1_iter5[1] + (r+0)*stride_t1_iter5[2]] + data_t1_iter5[(p+0)*stride_t1_iter5[0] + (q+1)*stride_t1_iter5[1] + (r+0)*stride_t1_iter5[2]] + data_t1_iter5[(p+0)*stride_t1_iter5[0] + (q-1)*stride_t1_iter5[1] + (r+0)*stride_t1_iter5[2]] + data_t1_iter5[(p+0)*stride_t1_iter5[0] + (q+0)*stride_t1_iter5[1] + (r+1)*stride_t1_iter5[2]] + data_t1_iter5[(p+0)*stride_t1_iter5[0] + (q+0)*stride_t1_iter5[1] + (r-1)*stride_t1_iter5[2]]) * 0.142857142f);
        float val_fpga = data_t0[p * stride_t0[0] + q * stride_t0[1] + r * stride_t0[2]];
        float val_cpu = result_t0;
        double threshold = 0.00001;
        if (nullptr != getenv("THRESHOLD")) {
          threshold = atof(getenv("THRESHOLD"));
        }
        threshold *= threshold;
        if (double(val_fpga-val_cpu) * double(val_fpga-val_cpu) > threshold && double(val_fpga-val_cpu) * double(val_fpga-val_cpu) / (double(val_cpu) * double(val_cpu)) > threshold) {
          fprintf(stderr, "%lf != %lf @(%d, %d, %d)\n", double(val_fpga), double(val_cpu), p, q, r);
          ++error_count;
        }
      }
    }
  }

  clog << "INFO: " << (error_count ? "FAIL" : "PASS") << "!\n";

  return error_count ? 1 : 0;
}
#endif  // SODA_TEST_MAIN
