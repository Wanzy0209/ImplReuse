ERROR: test_linear_reuse_kernels_batch_size_1024_in_features_1024_out_features_2048_cpu_bfloat16 (__main__.TestSelectAlgorithmCPU)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/pytorch/miniforge3/envs/pt_310/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3223, in wrapper
    method(*args, **kwargs)
  File "/home/pytorch/miniforge3/envs/pt_310/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 426, in instantiated_test
    result = test(self, **param_kwargs)
  File "/home/pytorch/miniforge3/envs/pt_310/lib/python3.10/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/mnt/lifeng/pytorch/test/inductor/test_cpu_select_algorithm.py", line 85, in wrapped
    return fn(*args, **kwargs)
  File "/home/pytorch/miniforge3/envs/pt_310/lib/python3.10/unittest/mock.py", line 1379, in patched
    return func(*newargs, **newkeywargs)
  File "/home/pytorch/miniforge3/envs/pt_310/lib/python3.10/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/home/pytorch/miniforge3/envs/pt_310/lib/python3.10/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/home/pytorch/miniforge3/envs/pt_310/lib/python3.10/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/home/pytorch/miniforge3/envs/pt_310/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "/mnt/lifeng/pytorch/test/inductor/test_cpu_select_algorithm.py", line 2939, in test_linear_reuse_kernels
    self.common(mod, (x))
  File "/mnt/lifeng/pytorch/test/inductor/test_torchinductor.py", line 451, in check_model
    eager_result = model(*ref_inputs, **ref_kwargs)
  File "/home/pytorch/miniforge3/envs/pt_310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/pytorch/miniforge3/envs/pt_310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
TypeError: TestSelectAlgorithm.test_linear_reuse_kernels.<locals>.M.forward() takes 2 positional arguments but 1025 were given