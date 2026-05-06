Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 1150, in test_wrapper
    return test(*args, **kwargs)
  File "/var/lib/jenkins/pytorch/test/test_ops.py", line 2184, in test_conj_view
    self._test_math_view(
  File "/var/lib/jenkins/pytorch/test/test_ops.py", line 2118, in _test_math_view
    self.assertEqual(expected_forward, forward_with_mathview)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 4233, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
AssertionError: Tensor-likes are not close!

Mismatched elements: 50 / 50 (100.0%)
Greatest absolute difference: 0.02258356846868992 at index (0, 2, 2) (up to 1e-05 allowed)
Greatest relative difference: 0.0051330882124602795 at index (1, 0, 2) (up to 1.3e-06 allowed)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3278, in wrapper
    method(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 427, in instantiated_test
    result = test(self, **param_kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 1230, in dep_fn
    return fn(slf, *args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 1230, in dep_fn
    return fn(slf, *args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 1700, in wrapper
    fn(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 1162, in test_wrapper
    raise e_tracked from e
Exception: Tensor-likes are not close!

Mismatched elements: 50 / 50 (100.0%)
Greatest absolute difference: 0.02258356846868992 at index (0, 2, 2) (up to 1e-05 allowed)
Greatest relative difference: 0.0051330882124602795 at index (1, 0, 2) (up to 1.3e-06 allowed)

Caused by sample input at index 9: SampleInput(input=Tensor[size=(2, 5, 5), device="cuda:0", dtype=torch.complex64], args=TensorList[Tensor[size=(2, 5, 5), device="cuda:0", dtype=torch.complex64, contiguous=False]], kwargs={}, broadcasts_input=False, name='')

To execute this test, run the following from the base repo dir:
    PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=9 PYTORCH_TEST_WITH_ROCM=1 python test/test_ops.py TestMathBitsCUDA.test_conj_view_cholesky_solve_cuda_complex64

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0