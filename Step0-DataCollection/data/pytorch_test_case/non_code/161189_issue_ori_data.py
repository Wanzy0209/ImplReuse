Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/dynamo/test_modes.py", line 501, in test_torch_function_mode_restore_on_exc
    self.assertEqual(_len_torch_function_stack(), 0)
  File "/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/testing/_internal/common_utils.py", line 4179, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
AssertionError: Scalars are not equal!

Expected 0 but got 1.
Absolute difference: 1
Relative difference: inf

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_SLOW=1 PYTORCH_TEST_SKIP_FAST=1 python test/dynamo/test_modes.py TorchFunctionModeTests.test_torch_function_mode_restore_on_exc

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0