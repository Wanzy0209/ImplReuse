Traceback (most recent call last):
  File "/var/lib/jenkins/pytorch/test/test_nn.py", line 7666, in with_tf32_on
    test.test_cuda(self, **kwargs)
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_nn.py", line 3548, in test_cuda
    test_case.assertEqual(cpu_gradInput, gpu_gradInput, atol=self.precision, rtol=0, exact_dtype=False)
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/testing/_internal/common_utils.py", line 4168, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
AssertionError: Tensor-likes are not close!

Mismatched elements: 4 / 36 (11.1%)
Greatest absolute difference: 0.06475556546724981 at index (0, 1, 2) (up to 0.05 allowed)
Greatest relative difference: 0.007305757961947742 at index (0, 1, 2) (up to 0 allowed)

The failure occurred for item [0]

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_ROCM=1 python test/test_nn.py TestNN.test_TransformerDecoderLayer_relu_activation_cuda_tf32

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0