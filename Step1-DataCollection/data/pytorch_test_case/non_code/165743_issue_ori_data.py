=================================== FAILURES ===================================
_ TestFP8MatmulCUDA.test_blockwise_mxfp8_nvfp4_mxfp4_numerics_test_case_name_a_ones_b_ones_fast_accum_True_31_1024_64_recipe_mxfp8_cuda _
Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/test_scaled_matmul_cuda.py", line 1495, in test_blockwise_mxfp8_nvfp4_mxfp4_numerics
    C = scaled_mm_wrap(
  File "/var/lib/jenkins/workspace/test/test_scaled_matmul_cuda.py", line 222, in scaled_mm_wrap
    out = scaled_mm(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/nn/functional.py", line 6698, in scaled_mm
    out = torch._scaled_mm_v2(
ValueError: scale_b must be swizzled to SWIZZLE_32_4_4 format
Exception raised from _scaled_mxfp8_mxfp8 at /var/lib/jenkins/workspace/aten/src/ATen/native/cuda/Blas.cpp:2235 (most recent call first):