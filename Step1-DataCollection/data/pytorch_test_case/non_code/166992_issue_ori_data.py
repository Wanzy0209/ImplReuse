|     ~~~~~~~~~~^~~~~~
  [1845/7982] Building CXX object third_party/ideep/mkl-dnn/src/cpu/x64/CMakeFiles/dnnl_cpu_x64.dir/gemm/f32/jit_avx_gemv_t_f32_kern.cpp.o
  FAILED: [code=1] aotriton/src/aotriton_runtime-stamp/aotriton_runtime-configure /pytorch/build/aotriton/src/aotriton_runtime-stamp/aotriton_runtime-configure
...

  -- Checkout nccl release tag: v2.27.5-1
  error: subprocess-exited-with-error
  
  × Building wheel for torch (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> No available output.
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  full command: /opt/python/cp310-cp310/bin/python /opt/python/cp310-cp310/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py build_wheel /tmp/tmp6kd_58v0
  cwd: /pytorch
  Building wheel for torch (pyproject.toml) ... 25l25herror
  ERROR: Failed building wheel for torch
Failed to build torch
error: failed-wheel-build-for-install

× Failed to build installable wheels for some pyproject.toml based projects
╰─> torch
Error: Process completed with exit code 1.