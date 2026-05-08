Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/inductor/test_flex_attention.py", line 3271, in test_flex_attention_stride_ordering
    out, _ = self.run_paged_attention(
  File "/var/lib/jenkins/workspace/test/inductor/test_flex_attention.py", line 754, in run_paged_attention
    compiled_out, compiled_lse = compiled_sdpa(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 886, in compile_wrapper
    return fn(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/nn/attention/flex_attention.py", line 1256, in flex_attention
    def flex_attention(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 1098, in _fn
    return fn(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 1134, in forward
    return compiled_fn(full_args)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 355, in runtime_wrapper
    all_outs = call_func_at_runtime_with_args(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/utils.py", line 130, in call_func_at_runtime_with_args
    out = normalize_as_list(f(args))
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 727, in inner_fn
    outs = compiled_fn(args)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 1959, in __call__
    return self.compiled_fn(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 529, in wrapper
    return compiled_fn(runtime_args)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/output_code.py", line 613, in __call__
    return self.current_callable(inputs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/utils.py", line 3001, in run
    out = model(new_inputs)
  File "/tmp/tmpo7l_i3dv/7z/c7z47qklytq4ytamkbainqu5jax5ehi2iygmz6m5iudxfjiyxmyy.py", line 833, in call
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/runtime/triton_heuristics.py", line 1285, in run
    self.autotune_to_one_config(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/runtime/triton_heuristics.py", line 1061, in autotune_to_one_config
    timings = self.benchmark_all_configs(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/runtime/triton_heuristics.py", line 1035, in benchmark_all_configs
    timings = {
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/runtime/triton_heuristics.py", line 1036, in <dictcomp>
    launcher: self.bench(launcher, *args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/runtime/triton_heuristics.py", line 900, in bench
    return benchmarker.benchmark_gpu(kernel_call, rep=40)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/runtime/benchmarking.py", line 39, in wrapper
    return fn(self, *args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/runtime/benchmarking.py", line 247, in benchmark_gpu
    torch.cuda.synchronize()
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/cuda/__init__.py", line 1084, in synchronize
    return torch._C._cuda_synchronize()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


To execute this test, run the following from the base repo dir:
    python test/inductor/test_flex_attention.py TestFlexAttentionCUDA.test_flex_attention_stride_ordering_mode_paged_attention_permute_order3_shape0_cuda

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0