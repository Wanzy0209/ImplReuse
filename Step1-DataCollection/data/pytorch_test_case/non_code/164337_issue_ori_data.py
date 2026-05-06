Traceback (most recent call last):
  File "/var/lib/jenkins/pytorch/test/inductor/test_torchinductor.py", line 14257, in new_test
    return value(self)
           ^^^^^^^^^^^
  File "/var/lib/jenkins/pytorch/test/inductor/test_benchmark_fusion.py", line 197, in test_tield_kernel_fusion
    self.common(f, (x,))
  File "/opt/conda/envs/py_3.12/lib/python3.12/contextlib.py", line 81, in inner
    return func(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^
  File "/var/lib/jenkins/pytorch/test/inductor/test_torchinductor.py", line 720, in check_model_gpu
    check_model(
  File "/var/lib/jenkins/pytorch/test/inductor/test_torchinductor.py", line 561, in check_model
    assert_equal_fn(
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/_dynamo/test_case.py", line 111, in assertEqual
    return super().assertEqual(x, y, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_utils.py", line 4168, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
AssertionError: Tensor-likes are not close!

Mismatched elements: 73534 / 1048576 (7.0%)
Greatest absolute difference: 0.00048828125 at index (0, 5) (up to 1e-05 allowed)
Greatest relative difference: 1.0 at index (1, 889) (up to 0.001 allowed)

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_ROCM=1 python test/inductor/test_benchmark_fusion.py BenchmarkFusionCudaTest.test_tield_kernel_fusion_cuda

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0