Traceback (most recent call last):
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_device_type.py", line 1151, in test_wrapper
    return test(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/py_3.12/lib/python3.12/unittest/mock.py", line 1861, in _inner
    return f(*args, **kw)
           ^^^^^^^^^^^^^^
  File "/var/lib/jenkins/pytorch/test/test_foreach.py", line 394, in test_pointwise_op_with_tensor_of_scalarlist_overload
    self._pointwise_test(
  File "/var/lib/jenkins/pytorch/test/test_foreach.py", line 518, in _pointwise_test
    actual = op(inputs, self.is_cuda, is_fastpath, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/var/lib/jenkins/pytorch/test/test_foreach.py", line 99, in __call__
    assert mta_called == (expect_fastpath and (not zero_size)), (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: mta_called=False, expect_fastpath=True, zero_size=False, self.func.__name__='_foreach_addcdiv_', keys=('aten::_foreach_addcdiv_', 'aten::result_type', 'hipLaunchKernel', 'hipDeviceSynchronize')

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_utils.py", line 3300, in wrapper
    method(*args, **kwargs)
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_utils.py", line 3300, in wrapper
    method(*args, **kwargs)
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_device_type.py", line 428, in instantiated_test
    result = test(self, **param_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_utils.py", line 1707, in wrapper
    fn(*args, **kwargs)
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_device_type.py", line 1163, in test_wrapper
    raise e_tracked from e
Exception: mta_called=False, expect_fastpath=True, zero_size=False, self.func.__name__='_foreach_addcdiv_', keys=('aten::_foreach_addcdiv_', 'aten::result_type', 'hipLaunchKernel', 'hipDeviceSynchronize')

Caused by sample input at index 17: SampleInput(input=TensorList[Tensor[size=(20, 20), device="cuda:0", dtype=torch.complex128], Tensor[size=(19, 19), device="cuda:0", dtype=torch.complex128], Tensor[size=(18, 18), device="cuda:0", dtype=torch.complex128], Tensor[size=(17, 17), device="cuda:0", dtype=torch.complex128], Tensor[size=(16, 16), device="cuda:0", dtype=torch.complex128], Tensor[size=(15, 15), device="cuda:0", dtype=torch.complex128], Tensor[size=(14, 14), device="cuda:0", dtype=torch.complex128], Tensor[size=(13, 13), device="cuda:0", dtype=torch.complex128], Tensor[size=(12, 12), device="cuda:0", dtype=torch.complex128], Tensor[size=(11, 11), device="cuda:0", dtype=torch.complex128], Tensor[size=(10, 10), device="cuda:0", dtype=torch.complex128], Tensor[size=(9, 9), device="cuda:0", dtype=torch.complex128], Tensor[size=(8, 8), device="cuda:0", dtype=torch.complex128], Tensor[size=(7, 7), device="cuda:0", dtype=torch.complex128], Tensor[size=(6, 6), device="cuda:0", dtype=torch.complex128], Tensor[size=(5, 5), device="cuda:0", dtype=torch.complex128], Tensor[size=(4, 4), device="cuda:0", dtype=torch.complex128], Tensor[size=(3, 3), device="cuda:0", dtype=torch.complex128], Tensor[size=(2, 2), device="cuda:0", dtype=torch.complex128], Tensor[size=(1, 1), device="cuda:0", dtype=torch.complex128]], args=(TensorList[Tensor[size=(20, 20), device="cuda:0", dtype=torch.complex128], Tensor[size=(19, 19), device="cuda:0", dtype=torch.complex128], Tensor[size=(18, 18), device="cuda:0", dtype=torch.complex128], Tensor[size=(17, 17), device="cuda:0", dtype=torch.complex128], Tensor[size=(16, 16), device="cuda:0", dtype=torch.complex128], Tensor[size=(15, 15), device="cuda:0", dtype=torch.complex128], Tensor[size=(14, 14), device="cuda:0", dtype=torch.complex128], Tensor[size=(13, 13), device="cuda:0", dtype=torch.complex128], Tensor[size=(12, 12), device="cuda:0", dtype=torch.complex128], Tensor[size=(11, 11), device="cuda:0", dtype=torch.complex128], Tensor[size=(10, 10), device="cuda:0", dtype=torch.complex128], Tensor[size=(9, 9), device="cuda:0", dtype=torch.complex128], Tensor[size=(8, 8), device="cuda:0", dtype=torch.complex128], Tensor[size=(7, 7), device="cuda:0", dtype=torch.complex128], Tensor[size=(6, 6), device="cuda:0", dtype=torch.complex128], Tensor[size=(5, 5), device="cuda:0", dtype=torch.complex128], Tensor[size=(4, 4), device="cuda:0", dtype=torch.complex128], Tensor[size=(3, 3), device="cuda:0", dtype=torch.complex128], Tensor[size=(2, 2), device="cuda:0", dtype=torch.complex128], Tensor[size=(1, 1), device="cuda:0", dtype=torch.complex128]],TensorList[Tensor[size=(20, 20), device="cuda:0", dtype=torch.complex128], Tensor[size=(19, 19), device="cuda:0", dtype=torch.complex128], Tensor[size=(18, 18), device="cuda:0", dtype=torch.complex128], Tensor[size=(17, 17), device="cuda:0", dtype=torch.complex128], Tensor[size=(16, 16), device="cuda:0", dtype=torch.complex128], Tensor[size=(15, 15), device="cuda:0", dtype=torch.complex128], Tensor[size=(14, 14), device="cuda:0", dtype=torch.complex128], Tensor[size=(13, 13), device="cuda:0", dtype=torch.complex128], Tensor[size=(12, 12), device="cuda:0", dtype=torch.complex128], Tensor[size=(11, 11), device="cuda:0", dtype=torch.complex128], Tensor[size=(10, 10), device="cuda:0", dtype=torch.complex128], Tensor[size=(9, 9), device="cuda:0", dtype=torch.complex128], Tensor[size=(8, 8), device="cuda:0", dtype=torch.complex128], Tensor[size=(7, 7), device="cuda:0", dtype=torch.complex128], Tensor[size=(6, 6), device="cuda:0", dtype=torch.complex128], Tensor[size=(5, 5), device="cuda:0", dtype=torch.complex128], Tensor[size=(4, 4), device="cuda:0", dtype=torch.complex128], Tensor[size=(3, 3), device="cuda:0", dtype=torch.complex128], Tensor[size=(2, 2), device="cuda:0", dtype=torch.complex128], Tensor[size=(1, 1), device="cuda:0", dtype=torch.complex128]]), kwargs={'scalars': '(True,1,2.0,(3+4.5j),3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0)'}, broadcasts_input=False, name='')

To execute this test, run the following from the base repo dir:
    PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=17 PYTORCH_TEST_WITH_ROCM=1 python test/test_foreach.py TestForeachCUDA.test_pointwise_op_with_tensor_of_scalarlist_overload__foreach_addcdiv_is_fastpath_True_cuda_complex128

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0