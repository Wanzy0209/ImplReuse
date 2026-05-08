Traceback (most recent call last):
  File "/opt/conda/envs/py_3.12/lib/python3.12/unittest/case.py", line 58, in testPartExecutor
    yield
  File "/opt/conda/envs/py_3.12/lib/python3.12/unittest/case.py", line 634, in run
    self._callTestMethod(testMethod)
  File "/opt/conda/envs/py_3.12/lib/python3.12/unittest/case.py", line 589, in _callTestMethod
    if method() is not None:
       ^^^^^^^^
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_utils.py", line 3224, in wrapper
    method(*args, **kwargs)
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_utils.py", line 3224, in wrapper
    method(*args, **kwargs)
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_device_type.py", line 426, in instantiated_test
    result = test(self, **param_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_device_type.py", line 1215, in dep_fn
    return fn(slf, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/py_3.12/lib/python3.12/contextlib.py", line 81, in inner
    return func(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^
  File "/var/lib/jenkins/pytorch/test/inductor/test_inductor_scheduler.py", line 125, in test_flop_counter_op
    self.assertEqual(
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_utils.py", line 4179, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
AssertionError: Scalars are not equal!

Expected 1384 but got 480.
Absolute difference: 904
Relative difference: 0.653179190751445
op = <function _test_cases.<locals>.composite at 0x76bb53ea4f40> reference flops = 480 != counters 1384

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_ROCM=1 python test/inductor/test_inductor_scheduler.py TestSchedulerCUDA.test_flop_counter_op_options0_cuda_float16

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0