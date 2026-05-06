Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/unittest/case.py", line 59, in testPartExecutor
    yield
  File "/opt/conda/envs/py_3.10/lib/python3.10/unittest/case.py", line 591, in run
    self._callTestMethod(testMethod)
  File "/opt/conda/envs/py_3.10/lib/python3.10/unittest/case.py", line 549, in _callTestMethod
    method()
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1775, in wrapper
    raise rv
RuntimeError: Exception in worker process:
Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1642, in _worker_loop
    cls._run_test_given_id(test_id)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1615, in _run_test_given_id
    test_fn(**kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1784, in wrapper
    fn()
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3224, in wrapper
    method(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 553, in instantiated_test
    test(self, **param_kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 1954, in wrap_fn
    return fn(self, *args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 215, in wrapper
    return func(*args, **kwargs)
  File "/var/lib/jenkins/pytorch/test/distributed/test_symmetric_memory.py", line 419, in test_fused_all_gather_scaled_matmul
    ag_output_1, mm_outputs_1 = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_ops.py", line 1254, in __call__
    return self._op(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/_symmetric_memory/__init__.py", line 943, in _fused_all_gather_scaled_matmul
    A, res = _fused_all_gather_matmul_impl(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/_symmetric_memory/__init__.py", line 610, in _fused_all_gather_matmul_impl
    _pipelined_all_gather_and_consume(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/_symmetric_memory/__init__.py", line 305, in _pipelined_all_gather_and_consume
    _pipelined_multi_all_gather_and_consume(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/_symmetric_memory/__init__.py", line 166, in _pipelined_multi_all_gather_and_consume
    symm_mem = get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/_symmetric_memory/__init__.py", line 127, in get_symm_mem_workspace
    return _SymmetricMemory.rendezvous(tensor)
RuntimeError: handle_type_ != Expandable_Segments_Handle_Type::UNSPECIFIED INTERNAL ASSERT FAILED at "/var/lib/jenkins/workspace/torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu":855, please report a bug to PyTorch. 

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_ROCM=1 python test/distributed/test_symmetric_memory.py SymmetricMemoryTest.test_fused_all_gather_scaled_matmul_gather_dim_0_scale_mode_tensor-wise

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0