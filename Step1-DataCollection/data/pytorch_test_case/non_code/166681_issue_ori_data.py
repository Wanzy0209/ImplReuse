Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/inductor/test_cudagraph_trees.py", line 1368, in test_multinomial
    self.assertEqual(samples, samples_compiled)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_dynamo/test_case.py", line 112, in assertEqual
    return super().assertEqual(x, y, *args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 4255, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
AssertionError: Tensor-likes are not equal!

Mismatched elements: 4 / 5 (80.0%)
Greatest absolute difference: 3 at index (1,)
Greatest relative difference: 2.0 at index (0,)

To execute this test, run the following from the base repo dir:
    python test/inductor/test_cudagraph_trees.py CudaGraphTreeTests.test_multinomial

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0