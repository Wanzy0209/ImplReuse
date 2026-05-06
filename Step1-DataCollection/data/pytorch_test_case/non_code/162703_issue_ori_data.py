Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/inductor/test_mkldnn_pattern_matcher.py", line 368, in test_conv2d_unary
    self._test_conv_unary_base(dim=4)
  File "/var/lib/jenkins/workspace/test/inductor/test_mkldnn_pattern_matcher.py", line 356, in _test_conv_unary_base
    self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
  File "/var/lib/jenkins/workspace/test/inductor/test_mkldnn_pattern_matcher.py", line 232, in _test_common
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_comparison.py", line 1589, in assert_close
    raise error_metas[0].to_error(msg)
AssertionError: Tensor-likes are not close!

Mismatched elements: 2773 / 46656 (5.9%)
Greatest absolute difference: 5.0187110900878906e-05 at index (0, 1, 3, 52) (up to 1e-05 allowed)
Greatest relative difference: 5.210636300034821e-05 at index (0, 1, 3, 11) (up to 1.3e-06 allowed)

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_ASAN=1 PYTORCH_TEST_WITH_UBSAN=1 PYTORCH_TEST_WITH_SLOW=1 PYTORCH_TEST_SKIP_FAST=1 python test/inductor/test_mkldnn_pattern_matcher.py TestDynamicPatternMatcherGenericCPU.test_conv2d_unary_dynamic_shapes_cpu

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0