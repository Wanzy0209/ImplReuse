Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 1150, in test_wrapper
    return test(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 1230, in dep_fn
    return fn(slf, *args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_cuda.py", line 279, in wrapped
    return f(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 1485, in only_fn
    return fn(self, *args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 2413, in wrapper
    fn(*args, **kwargs)
  File "/var/lib/jenkins/pytorch/test/test_ops.py", line 731, in test_noncontiguous_samples
    self.assertEqual(actual, expected)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 4233, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
AssertionError: Tensor-likes are not close!

Mismatched elements: 48 / 50 (96.0%)
Greatest absolute difference: 0.015094757080078125 at index (1, 0, 3) (up to 1e-05 allowed)
Greatest relative difference: 0.06608710438013077 at index (1, 0, 1) (up to 1.3e-06 allowed)

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

Mismatched elements: 48 / 50 (96.0%)
Greatest absolute difference: 0.015094757080078125 at index (1, 0, 3) (up to 1e-05 allowed)
Greatest relative difference: 0.06608710438013077 at index (1, 0, 1) (up to 1.3e-06 allowed)

Caused by sample input at index 11: SampleInput(input=Tensor[size=(2, 5, 5), device="cuda:0", dtype=torch.float32], args=TensorList[Tensor[size=(2, 5, 5), device="cuda:0", dtype=torch.float32]], kwargs={'upper': 'True'}, broadcasts_input=False, name='')

To execute this test, run the following from the base repo dir:
    PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=11 PYTORCH_TEST_WITH_ROCM=1 python test/test_ops.py TestCommonCUDA.test_noncontiguous_samples_cholesky_solve_cuda_float32

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0