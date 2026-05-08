Traceback (most recent call last):
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_device_type.py", line 1144, in test_wrapper
    return test(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_cuda.py", line 280, in wrapped
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/var/lib/jenkins/pytorch/test/functorch/test_ops.py", line 2143, in test_vmapjvpvjp
    self.assertEqual(loop_out, batched_out)
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_utils.py", line 4168, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
AssertionError: Tensor-likes are not close!

Mismatched elements: 50 / 50 (100.0%)
Greatest absolute difference: 0.41283702850341797 at index (0, 0, 4, 4) (up to 0.0001 allowed)
Greatest relative difference: 0.029489146545529366 at index (0, 0, 4, 4) (up to 0.0001 allowed)

The failure occurred for item [3]

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_utils.py", line 3213, in wrapper
    method(*args, **kwargs)
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_device_type.py", line 426, in instantiated_test
    result = test(self, **param_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_device_type.py", line 1224, in dep_fn
    return fn(slf, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_device_type.py", line 1224, in dep_fn
    return fn(slf, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_utils.py", line 1644, in wrapper
    fn(*args, **kwargs)
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_device_type.py", line 1156, in test_wrapper
    raise e_tracked from e
Exception: Tensor-likes are not close!

Mismatched elements: 50 / 50 (100.0%)
Greatest absolute difference: 0.41283702850341797 at index (0, 0, 4, 4) (up to 0.0001 allowed)
Greatest relative difference: 0.029489146545529366 at index (0, 0, 4, 4) (up to 0.0001 allowed)

The failure occurred for item [3]

Caused by sample input at index 7: SampleInput(input=Tensor[size=(5, 5), device="cuda:0", dtype=torch.float32], args=TensorList[Tensor[size=(5, 5), device="cuda:0", dtype=torch.float32, contiguous=False]], kwargs={'upper': 'False'}, broadcasts_input=False, name='')

To execute this test, run the following from the base repo dir:
    PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=7 PYTORCH_TEST_WITH_ROCM=1 python test/functorch/test_ops.py TestOperatorsCUDA.test_vmapjvpvjp_cholesky_solve_cuda_float32

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0