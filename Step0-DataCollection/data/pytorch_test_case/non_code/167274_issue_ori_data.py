Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/inductor/test_cudagraph_trees.py", line 4535, in test_graph_partition_cudagraphs_aot_eager_compat_equal
    self._test_cudagraphs_aot_eager_compat_equal(torch.device("cuda:0"))
  File "/opt/conda/envs/py_3.10/lib/python3.10/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/opt/conda/envs/py_3.10/lib/python3.10/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/var/lib/jenkins/workspace/test/inductor/test_cudagraph_trees.py", line 4522, in _test_cudagraphs_aot_eager_compat_equal
    self.assertEqual(outs, outs2)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_dynamo/test_case.py", line 112, in assertEqual
    return super().assertEqual(x, y, *args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 4255, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
AssertionError: Tensor-likes are not close!

Mismatched elements: 16 / 16 (100.0%)
Greatest absolute difference: 0.12529590725898743 at index (3, 2) (up to 1e-05 allowed)
Greatest relative difference: 0.3683590590953827 at index (0, 2) (up to 1.3e-06 allowed)

The failure occurred for item [1]

To execute this test, run the following from the base repo dir:
    python test/inductor/test_cudagraph_trees.py TestSAC.test_graph_partition_cudagraphs_aot_eager_compat_equal

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0