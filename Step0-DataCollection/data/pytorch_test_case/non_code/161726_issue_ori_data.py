Traceback (most recent call last):
  File "/var/lib/jenkins/pytorch/test/jit/test_models.py", line 588, in test_vae_cuda
    self._test_vae(self, device="cuda", check_export_import=False)
  File "/var/lib/jenkins/pytorch/test/jit/test_models.py", line 576, in _test_vae
    self.checkTrace(
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/jit_utils.py", line 619, in checkTrace
    self.assertEqual(grads, grads_ge, atol=grad_atol, rtol=grad_rtol)
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_utils.py", line 4179, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
AssertionError: Tensor-likes are not close!

Mismatched elements: 844 / 100352 (0.8%)
Greatest absolute difference: 5.346536636352539e-05 at index (31, 0, 15, 8) (up to 1e-05 allowed)
Greatest relative difference: 0.4476863741874695 at index (31, 0, 13, 27) (up to 1.3e-06 allowed)

The failure occurred for item [0]

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_ROCM=1 python test/jit/test_models.py TestModels.test_vae_cuda

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0