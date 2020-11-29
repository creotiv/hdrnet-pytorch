#ifndef BILATERAL_SLICE_H_SZ3NVCCJ
#define BILATERAL_SLICE_H_SZ3NVCCJ

// #ifdef __cplusplus
// extern "C" {
// #endif

void BilateralSliceApplyKernelLauncher(
    int bs, int gh, int gw, int gd, 
    int input_chans, int output_chans, int h, int w,
    const float* const grid, const float* const guide, const float* const input,
    float* const out);

void BilateralSliceApplyGradKernelLauncher(
    int bs, int gh, int gw, int gd, 
    int input_chans, int output_chans, int h, int w,
    const float* grid, const float* guide, const float* input, 
    const float* d_output,
    float* d_grid, float* d_guide, float* d_input);
// #ifdef __cplusplus
// }
// #endif

#endif /* end of include guard: BILATERAL_SLICE_H_SZ3NVCCJ */
