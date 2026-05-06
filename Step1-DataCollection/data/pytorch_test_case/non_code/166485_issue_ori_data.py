Traceback (most recent call last):
  File "/var/lib/jenkins/pytorch/test/test_foreach.py", line 235, in test_parity
    actual = func(
  File "/var/lib/jenkins/pytorch/test/test_foreach.py", line 99, in __call__
    assert mta_called == (expect_fastpath and (not zero_size)), (
AssertionError: mta_called=False, expect_fastpath=True, zero_size=False, self.func.__name__='_foreach_lgamma_', keys=('aten::_foreach_lgamma_', 'hipLaunchKernel', 'hipDeviceSynchronize')

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 1151, in test_wrapper
    return test(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/unittest/mock.py", line 1833, in _inner
    return f(*args, **kw)
  File "/var/lib/jenkins/pytorch/test/test_foreach.py", line 242, in test_parity
    with self.assertRaises(type(e)):
  File "/opt/conda/envs/py_3.10/lib/python3.10/unittest/case.py", line 226, in __exit__
    self._raiseFailure("{} not raised".format(exc_name))
  File "/opt/conda/envs/py_3.10/lib/python3.10/unittest/case.py", line 163, in _raiseFailure
    raise self.test_case.failureException(msg)
AssertionError: AssertionError not raised

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3300, in wrapper
    method(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3300, in wrapper
    method(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 428, in instantiated_test
    result = test(self, **param_kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 1707, in wrapper
    fn(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 1163, in test_wrapper
    raise e_tracked from e
Exception: AssertionError not raised

Caused by sample input at index 0: SampleInput(input=TensorList[Tensor[size=(20, 20), device="cuda:0", dtype=torch.float64], Tensor[size=(19, 19), device="cuda:0", dtype=torch.float64], Tensor[size=(18, 18), device="cuda:0", dtype=torch.float64], Tensor[size=(17, 17), device="cuda:0", dtype=torch.float64], Tensor[size=(16, 16), device="cuda:0", dtype=torch.float64], Tensor[size=(15, 15), device="cuda:0", dtype=torch.float64], Tensor[size=(14, 14), device="cuda:0", dtype=torch.float64], Tensor[size=(13, 13), device="cuda:0", dtype=torch.float64], Tensor[size=(12, 12), device="cuda:0", dtype=torch.float64], Tensor[size=(11, 11), device="cuda:0", dtype=torch.float64], Tensor[size=(10, 10), device="cuda:0", dtype=torch.float64], Tensor[size=(9, 9), device="cuda:0", dtype=torch.float64], Tensor[size=(8, 8), device="cuda:0", dtype=torch.float64], Tensor[size=(7, 7), device="cuda:0", dtype=torch.float64], Tensor[size=(6, 6), device="cuda:0", dtype=torch.float64], Tensor[size=(5, 5), device="cuda:0", dtype=torch.float64], Tensor[size=(4, 4), device="cuda:0", dtype=torch.float64], Tensor[size=(3, 3), device="cuda:0", dtype=torch.float64], Tensor[size=(2, 2), device="cuda:0", dtype=torch.float64], Tensor[size=(1, 1), device="cuda:0", dtype=torch.float64]], args=(), kwargs={}, broadcasts_input=False, name='')

To execute this test, run the following from the base repo dir:
    PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=0 PYTORCH_TEST_WITH_ROCM=1 python test/test_foreach.py TestForeachCUDA.test_parity__foreach_lgamma_fastpath_inplace_cuda_float64

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0