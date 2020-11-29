// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <THC/THC.h>
#include <iostream>
#include "math.h"

extern THCState *state;

__device__ float diff_abs(float x) {
  float eps = 1e-8;
  return sqrt(x*x+eps);
}

__device__ float d_diff_abs(float x) {
  float eps = 1e-8;
  return x/sqrt(x*x+eps);
}

__device__ float weight_z(float x) {
  float abx = diff_abs(x);
  return max(1.0f-abx, 0.0f);
}

__device__ float d_weight_z(float x) {
  float abx = diff_abs(x);
  if(abx > 1.0f) {
    return 0.0f;
    // return abx;
  } else {
    return d_diff_abs(x);
  }
}

__global__ void BilateralSliceApplyKernel(
    int64_t nthreads,
    const float* grid, const float* guide, const float* input,
    const int bs, const int h, const int w, 
    const int gh, const int gw, const int gd,
    const int input_chans, const int output_chans,
    float* out)
{
  // - Samples centered at 0.5.
  // - Repeating boundary conditions

  int grid_chans = (input_chans+1)*output_chans;
  int coeff_stride = input_chans+1;

  const int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx < nthreads) {
    int x = idx % w;
    int y = (idx / w) % h;
    int out_c = (idx / (w*h)) % output_chans;
    int b = (idx / (output_chans*w*h));

    float gx = (x+0.5f)*gw/(1.0f*w);
    float gy = (y+0.5f)*gh/(1.0f*h);
    float gz = guide[x + w*(y + h*b)]*gd;

    int fx = static_cast<int>(floor(gx-0.5f));
    int fy = static_cast<int>(floor(gy-0.5f));
    int fz = static_cast<int>(floor(gz-0.5f));

    // Grid strides
    int sx = 1;
    int sy = gw;
    int sz = gw*gh;
    int sc = gw*gh*gd;
    int sb = grid_chans*gd*gw*gh;

    float value = 0.0f;
    for (int in_c = 0; in_c < coeff_stride; ++in_c) {
      float coeff_sample = 0.0f;
      for (int xx = fx; xx < fx+2; ++xx) {
        int x_ = max(min(xx, gw-1), 0);
        float wx = max(1.0f-abs(xx+0.5-gx), 0.0f);
        for (int yy = fy; yy < fy+2; ++yy)
        {
          int y_ = max(min(yy, gh-1), 0);
          float wy = max(1.0f-abs(yy+0.5-gy), 0.0f);
          for (int zz = fz; zz < fz+2; ++zz)
          {
            int z_ = max(min(zz, gd-1), 0);
            float wz = weight_z(zz+0.5-gz);
            int grid_idx =
              sc*(coeff_stride*out_c + in_c) + sz*z_ + sx*x_ + sy*y_ + sb*b;
            coeff_sample += grid[grid_idx]*wx*wy*wz;
          }
        }
      } // Grid trilinear interpolation
      if(in_c < input_chans) {
        int input_idx = x + w*(y + input_chans*(in_c + h*b));
        value += coeff_sample*input[input_idx];
      } else { // Offset term
        value += coeff_sample;
      }
    }
    out[idx] = value;
  }
}


__global__ void BilateralSliceApplyGridGradKernel(
    int64_t nthreads,
    const float* grid, const float* guide, const float* input, const float* d_output, 
    const int bs, const int h, const int w, 
    const int gh, const int gw, const int gd,
    const int input_chans, const int output_chans,
    float* out)
{
  int grid_chans = (input_chans+1)*output_chans;
  int coeff_stride = input_chans+1;

  const int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx < nthreads) {
    int gx = idx % gw;
    int gy = (idx / gw) % gh;
    int gz = (idx / (gh*gw)) % gd;
    int c = (idx / (gd*gh*gw)) % grid_chans;
    int b = (idx / (grid_chans*gd*gw*gh));

    float scale_w = w*1.0/gw;
    float scale_h = h*1.0/gh;

    int left_x = static_cast<int>(floor(scale_w*(gx+0.5-1)));
    int right_x = static_cast<int>(ceil(scale_w*(gx+0.5+1)));
    int left_y = static_cast<int>(floor(scale_h*(gy+0.5-1)));
    int right_y = static_cast<int>(ceil(scale_h*(gy+0.5+1)));

    // Strides in the output
    int sx = 1;
    int sy = w;
    int sc = h*w;
    int sb = output_chans*w*h;

    // Strides in the input
    int isx = 1;
    int isy = w;
    int isc = h*w;
    int isb = output_chans*w*h;

    int out_c = c / coeff_stride;
    int in_c = c % coeff_stride;

    float value = 0.0f;
    for (int x = left_x; x < right_x; ++x)
    {
      int x_ = x;

      // mirror boundary
      if (x_ < 0) x_ = -x_-1;
      if (x_ >= w) x_ = 2*w-1-x_;

      float gx2 = (x+0.5f)/scale_w;
      float wx = max(1.0f-abs(gx+0.5-gx2), 0.0f);

      for (int y = left_y; y < right_y; ++y)
      {
        int y_ = y;

        // mirror boundary
        if (y_ < 0) y_ = -y_-1;
        if (y_ >= h) y_ = 2*h-1-y_;

        float gy2 = (y+0.5f)/scale_h;
        float wy = max(1.0f-abs(gy+0.5-gy2), 0.0f);

        int guide_idx = x_ + w*y_ + h*w*b;
        float gz2 = guide[guide_idx]*gd;
        float wz = weight_z(gz+0.5f-gz2);
        if ((gz==0 && gz2<0.5f) || (gz==gd-1 && gz2>gd-0.5f)) {
          wz = 1.0f;
        }

        int back_idx = sc*out_c + sx*x_ + sy*y_ + sb*b;
        if (in_c < input_chans) {
          int input_idx = isc*in_c + isx*x_ + isy*y_ + isb*b;
          value += wz*wx*wy*d_output[back_idx]*input[input_idx];
        } else { // offset term
          value += wz*wx*wy*d_output[back_idx];
        }
      }
    }
    out[idx] = value;
  }
}


__global__ void BilateralSliceApplyGuideGradKernel(
    int64_t nthreads,
    const float* grid, const float* guide, const float* input, const float* d_output, 
    const int bs, const int h, const int w,
    const int gh, const int gw, const int gd,
    const int input_chans, const int output_chans,
    float* out)
{
  int grid_chans = (input_chans+1)*output_chans;
  int coeff_stride = input_chans+1;

  const int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx < nthreads) {
    int x = idx  % w;
    int y = (idx / w) % h;
    int b = (idx / (w*h));

    float gx = (x+0.5f)*gw/(1.0f*w);
    float gy = (y+0.5f)*gh/(1.0f*h);
    float gz = guide[x + w*(y + h*b)]*gd;

    int fx = static_cast<int>(floor(gx-0.5f));
    int fy = static_cast<int>(floor(gy-0.5f));
    int fz = static_cast<int>(floor(gz-0.5f));

    // Grid stride 
    int sx = 1;
    int sy = gw;
    int sz = gw*gh;
    int sc = gw*gh*gd;
    int sb = grid_chans*gd*gw*gh;

    float out_sum = 0.0f;
    for (int out_c = 0; out_c < output_chans; ++out_c) {

      float in_sum = 0.0f;
      for (int in_c = 0; in_c < coeff_stride; ++in_c) {

        float grid_sum = 0.0f;
        for (int xx = fx; xx < fx+2; ++xx) {
          int x_ = max(min(xx, gw-1), 0);
          float wx = max(1.0f-abs(xx+0.5-gx), 0.0f);
          for (int yy = fy; yy < fy+2; ++yy)
          {
            int y_ = max(min(yy, gh-1), 0);
            float wy = max(1.0f-abs(yy+0.5-gy), 0.0f);
            for (int zz = fz; zz < fz+2; ++zz)
            {
              int z_ = max(min(zz, gd-1), 0);
              float dwz = gd*d_weight_z(zz+0.5-gz);

              int grid_idx = sc*(coeff_stride*out_c + in_c) + sz*z_ + sx*x_ + sy*y_ + sb*b;
              grid_sum += grid[grid_idx]*wx*wy*dwz;
            } // z
          } // y
        } // x, grid trilinear interp

        if(in_c < input_chans) {
          in_sum += grid_sum*input[input_chans*(x + w*(y + h*(in_c + input_chans*b)))];
        } else {  // offset term
          in_sum += grid_sum;
        }
      } // in_c

      out_sum += in_sum*d_output[x + w*(y + h*(out_c + output_chans*b))];
    } // out_c

    out[idx] = out_sum;
  }
}


__global__ void BilateralSliceApplyInputGradKernel(
    int64_t nthreads,
    const float* grid, const float* guide, const float* input, const float* d_output, 
    const int bs, const int h, const int w,
    const int gh, const int gw, const int gd,
    const int input_chans, const int output_chans, 
    float* out)
{
  int grid_chans = (input_chans+1)*output_chans;
  int coeff_stride = input_chans+1;

  const int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx < nthreads) {
    int x = idx % w;
    int y = (idx / w) % h;
    int in_c = (idx / (w*h)) % input_chans;
    int b = (idx / (input_chans*w*h));

    float gx = (x+0.5f)*gw/(1.0f*w);
    float gy = (y+0.5f)*gh/(1.0f*h);
    float gz = guide[x + w*(y + h*b)]*gd;

    int fx = static_cast<int>(floor(gx-0.5f));
    int fy = static_cast<int>(floor(gy-0.5f));
    int fz = static_cast<int>(floor(gz-0.5f));

    // Grid stride 
    int sx = 1;
    int sy = gw;
    int sz = gw*gh;
    int sc = gw*gh*gd;
    int sb = grid_chans*gd*gw*gh;

    float value = 0.0f;
    for (int out_c = 0; out_c < output_chans; ++out_c) {
      float chan_val = 0.0f;
      for (int xx = fx; xx < fx+2; ++xx) {
        int x_ = max(min(xx, gw-1), 0);
        float wx = max(1.0f-abs(xx+0.5-gx), 0.0f);
        for (int yy = fy; yy < fy+2; ++yy)
        {
          int y_ = max(min(yy, gh-1), 0);
          float wy = max(1.0f-abs(yy+0.5-gy), 0.0f);
          for (int zz = fz; zz < fz+2; ++zz)
          {

            int z_ = max(min(zz, gd-1), 0);

            float wz = weight_z(zz+0.5-gz);

            int grid_idx = sc*(coeff_stride*out_c + in_c) + sz*z_ + sx*x_ + sy*y_ + sb*b;
            chan_val += grid[grid_idx]*wx*wy*wz;
          } // z
        } // y
      } // x, grid trilinear interp

      value += chan_val*d_output[x + w*(y + h*(out_c + output_chans*b))];
    } // out_c
    out[idx] = value;
  }
}


// // -- KERNEL LAUNCHERS ---------------------------------------------------------
void BilateralSliceApplyKernelLauncher(
    int bs, int gh, int gw, int gd, 
    int input_chans, int output_chans,
    int h, int w,
    const float* const grid, const float* const guide, const float* const input,
    float* const out)
{
  int total_count = bs*h*w*output_chans;
  const int64_t block_sz = 512;
  const int64_t nblocks = (total_count + block_sz - 1) / block_sz;
  if (total_count > 0) {
    BilateralSliceApplyKernel<<<nblocks, block_sz, 0, THCState_getCurrentStream(state)>>>(
        total_count, grid, guide, input,
        bs, h, w, gh, gw, gd, input_chans, output_chans, 
        out);
    THCudaCheck(cudaPeekAtLastError());
  }
}


void BilateralSliceApplyGradKernelLauncher(
    int bs, int gh, int gw, int gd, 
    int input_chans, int output_chans, int h, int w,
    const float* grid, const float* guide, const float* input, 
    const float* d_output,
    float* d_grid, float* d_guide, float* d_input)
{
  int64_t coeff_chans = (input_chans+1)*output_chans;
  const int64_t block_sz = 512;
  int64_t grid_count = bs*gh*gw*gd*coeff_chans;
  if (grid_count > 0) {
    const int64_t nblocks = (grid_count + block_sz - 1) / block_sz;
    BilateralSliceApplyGridGradKernel<<<nblocks, block_sz, 0, THCState_getCurrentStream(state)>>>(
        grid_count, grid, guide, input, d_output,
        bs, h, w, gh, gw, gd,
        input_chans, output_chans,
        d_grid);
  }

  int64_t guide_count = bs*h*w;
  if (guide_count > 0) {
    const int64_t nblocks = (guide_count + block_sz - 1) / block_sz;
    BilateralSliceApplyGuideGradKernel<<<nblocks, block_sz, 0, THCState_getCurrentStream(state)>>>(
        guide_count, grid, guide, input, d_output,
        bs, h, w, gh, gw, gd,
        input_chans, output_chans, 
        d_guide);
  }

  int64_t input_count = bs*h*w*input_chans;
  if (input_count > 0) {
    const int64_t nblocks = (input_count + block_sz - 1) / block_sz;
    BilateralSliceApplyInputGradKernel<<<nblocks, block_sz, 0, THCState_getCurrentStream(state)>>>(
        input_count, grid, guide, input, d_output,
        bs, h, w, gh, gw, gd,
        input_chans, output_chans, 
        d_input);
  }
}
