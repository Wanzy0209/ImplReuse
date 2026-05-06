Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 1151, in test_wrapper
    return test(*args, **kwargs)
  File "/var/lib/jenkins/pytorch/test/test_ops_gradients.py", line 75, in test_fn_gradgrad
    self._check_helper(device, dtype, op, op.get_op(), "bwgrad_bwgrad")
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 5581, in _check_helper
    self.assertTrue(gradgradcheck(fn, gradcheck_args, **kwargs))
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 5166, in gradgradcheck
    return torch.autograd.gradgradcheck(fn, inputs, grad_outputs, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/autograd/gradcheck.py", line 2258, in gradgradcheck
    return gradcheck(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/autograd/gradcheck.py", line 2056, in gradcheck
    return _gradcheck_helper(**args)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/autograd/gradcheck.py", line 2085, in _gradcheck_helper
    _gradcheck_real_imag(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/autograd/gradcheck.py", line 1495, in _gradcheck_real_imag
    gradcheck_fn(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/autograd/gradcheck.py", line 1929, in _fast_gradcheck
    _check_analytical_numerical_equal(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/autograd/gradcheck.py", line 1858, in _check_analytical_numerical_equal
    raise GradcheckError(
torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 1 with respect to input 0,
numerical:4700.987699244844
analytical:-0.2053447426363384

The above quantities relating the numerical and analytical jacobians are computed 
in fast mode. See: https://github.com/pytorch/pytorch/issues/53876 for more background 
about fast mode. Below, we recompute numerical and analytical jacobians in slow mode:

Numerical:
 tensor([[-1.2601,  0.2878,  0.5647,  ...,  0.0000,  0.0000,  0.0000],
        [-1.0498,  0.8976, -0.0466,  ...,  0.0000,  0.0000,  0.0000],
        [-0.9508, -0.1238,  0.4866,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000, -1.3363],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  1.0366],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  1.8311]],
       device='cuda:0', dtype=torch.float64)
Analytical:
tensor([[-1.2601,  0.2878,  0.5647,  ...,  0.0000,  0.0000,  0.0000],
        [-1.0498,  0.8976, -0.0466,  ...,  0.0000,  0.0000,  0.0000],
        [-0.9508, -0.1238,  0.4866,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000, -1.3363],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  1.0366],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  1.8311]],
       device='cuda:0', dtype=torch.float64)

The max per-element difference (slow mode) is: 4.326386277307475e-09.
Fast gradcheck failed but element-wise differences are small. This means that the
test might've passed in slow_mode!

If you are adding a new operator, please file an issue and then use one of the
workarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck:

If the test
- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck
  with `fast_mode=False` as a keyword argument.
- is OpInfo-based (e.g., in test_ops_gradients.py), then modify the OpInfo for the test
  to have `gradcheck_fast_mode=False`
- is a Module test (e.g., in common_nn.py), then modify the corresponding
  module_test entry to have `gradcheck_fast_mode=False`

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3293, in wrapper
    method(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 428, in instantiated_test
    result = test(self, **param_kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 1231, in dep_fn
    return fn(slf, *args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 1231, in dep_fn
    return fn(slf, *args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_device_type.py", line 1163, in test_wrapper
    raise e_tracked from e
Exception: Jacobian mismatch for output 1 with respect to input 0,
numerical:4700.987699244844
analytical:-0.2053447426363384

The above quantities relating the numerical and analytical jacobians are computed 
in fast mode. See: https://github.com/pytorch/pytorch/issues/53876 for more background 
about fast mode. Below, we recompute numerical and analytical jacobians in slow mode:

Numerical:
 tensor([[-1.2601,  0.2878,  0.5647,  ...,  0.0000,  0.0000,  0.0000],
        [-1.0498,  0.8976, -0.0466,  ...,  0.0000,  0.0000,  0.0000],
        [-0.9508, -0.1238,  0.4866,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000, -1.3363],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  1.0366],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  1.8311]],
       device='cuda:0', dtype=torch.float64)
Analytical:
tensor([[-1.2601,  0.2878,  0.5647,  ...,  0.0000,  0.0000,  0.0000],
        [-1.0498,  0.8976, -0.0466,  ...,  0.0000,  0.0000,  0.0000],
        [-0.9508, -0.1238,  0.4866,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000, -1.3363],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  1.0366],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  1.8311]],
       device='cuda:0', dtype=torch.float64)

The max per-element difference (slow mode) is: 4.326386277307475e-09.
Fast gradcheck failed but element-wise differences are small. This means that the
test might've passed in slow_mode!

If you are adding a new operator, please file an issue and then use one of the
workarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck:

If the test
- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck
  with `fast_mode=False` as a keyword argument.
- is OpInfo-based (e.g., in test_ops_gradients.py), then modify the OpInfo for the test
  to have `gradcheck_fast_mode=False`
- is a Module test (e.g., in common_nn.py), then modify the corresponding
  module_test entry to have `gradcheck_fast_mode=False`

Caused by sample input at index 11: SampleInput(input=Tensor[size=(2, 5, 5), device="cuda:0", dtype=torch.float64], args=TensorList[Tensor[size=(2, 5, 5), device="cuda:0", dtype=torch.float64]], kwargs={'upper': 'True'}, broadcasts_input=False, name='')

To execute this test, run the following from the base repo dir:
    PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=11 PYTORCH_TEST_WITH_ROCM=1 PYTORCH_TEST_WITH_INDUCTOR=1 python test/test_ops_gradients.py TestBwdGradientsCUDA.test_fn_gradgrad_cholesky_solve_cuda_float64

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0