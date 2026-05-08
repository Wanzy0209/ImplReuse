Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1800, in wrapper
    raise rv
RuntimeError: Exception in worker process:
Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1667, in _worker_loop
    cls._run_test_given_id(test_id)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1640, in _run_test_given_id
    test_fn(**kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1809, in wrapper
    fn()
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3224, in wrapper
    method(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 553, in instantiated_test
    test(self, **param_kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 1954, in wrap_fn
    return fn(self, *args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 215, in wrapper
    return func(*args, **kwargs)
  File "/var/lib/jenkins/pytorch/test/distributed/test_symmetric_memory.py", line 457, in test_fused_all_gather_scaled_matmul
    ag_output_1, mm_outputs_1 = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_ops.py", line 1255, in __call__
    return self._op(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/_symmetric_memory/__init__.py", line 950, in _fused_all_gather_scaled_matmul
    A, res = _fused_all_gather_matmul_impl(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/_symmetric_memory/__init__.py", line 598, in _fused_all_gather_matmul_impl
    _pipelined_all_gather_and_consume(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/_symmetric_memory/__init__.py", line 312, in _pipelined_all_gather_and_consume
    _pipelined_multi_all_gather_and_consume(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/_symmetric_memory/__init__.py", line 174, in _pipelined_multi_all_gather_and_consume
    group_size = symm_mem.world_size
AttributeError: 'NoneType' object has no attribute 'world_size'

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_ROCM=1 python test/distributed/test_symmetric_memory.py AsyncTPTest.test_fused_all_gather_scaled_matmul_gather_dim_0_scale_mode_row-wise-replicated

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0