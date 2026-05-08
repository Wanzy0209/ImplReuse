Traceback (most recent call last):
  File "/var/lib/jenkins/pytorch/test/inductor/test_torchinductor.py", line 5658, in test_var_mean
    self.common(
  File "/opt/conda/envs/py_3.12/lib/python3.12/contextlib.py", line 81, in inner
    return func(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^
  File "/var/lib/jenkins/pytorch/test/inductor/test_torchinductor.py", line 687, in check_model_gpu
    check_model(
  File "/var/lib/jenkins/pytorch/test/inductor/test_torchinductor.py", line 561, in check_model
    assert_equal_fn(
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/_dynamo/test_case.py", line 111, in assertEqual
    return super().assertEqual(x, y, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_utils.py", line 4248, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
AssertionError: Tensor-likes are not close!

Mismatched elements: 4 / 4 (100.0%)
Greatest absolute difference: 0.5851404070854187 at index (0, 3) (up to 1e-05 allowed)
Greatest relative difference: 0.5756681561470032 at index (0, 1) (up to 1.3e-06 allowed)

The failure occurred for item [2]

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_ROCM=1 python test/inductor/test_torchinductor.py GPUTests.test_var_mean_tile_reduction_True_cuda

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0