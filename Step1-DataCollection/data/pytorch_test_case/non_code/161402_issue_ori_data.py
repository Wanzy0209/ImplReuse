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
  File "/var/lib/jenkins/pytorch/test/distributed/test_symmetric_memory.py", line 587, in test_low_contention_reduce_scatter
    res = torch.ops.symm_mem._low_contention_reduce_scatter(t, reduce_op, "0")
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_ops.py", line 1254, in __call__
    return self._op(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/_symmetric_memory/__init__.py", line 1603, in _low_contention_reduce_scatter
    symm_mem = rendezvous(tensor, group_name)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/_symmetric_memory/__init__.py", line 1739, in rendezvous
    return _SymmetricMemory.rendezvous(tensor, group_name)
RuntimeError: handle_type_ != Expandable_Segments_Handle_Type::UNSPECIFIED INTERNAL ASSERT FAILED at "/var/lib/jenkins/workspace/torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu":847, please report a bug to PyTorch. 

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_ROCM=1 python test/distributed/test_symmetric_memory.py SymmetricMemoryTest.test_low_contention_reduce_scatter_reduce_op_avg_symm_mem_input_True

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0