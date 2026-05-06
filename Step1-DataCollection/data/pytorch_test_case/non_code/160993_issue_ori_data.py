FAIL: test_index_copy_mps_float32 (__main__.TestIndexingMPS.test_index_copy_mps_float32)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/Users/malfet/git/pytorch/pytorch/torch/testing/_internal/common_utils.py", line 3224, in wrapper
    method(*args, **kwargs)
    ~~~~~~^^^^^^^^^^^^^^^^^
  File "/Users/malfet/git/pytorch/pytorch/torch/testing/_internal/common_device_type.py", line 426, in instantiated_test
    result = test(self, **param_kwargs)
  File "/Users/malfet/git/pytorch/pytorch/test/test_indexing.py", line 1874, in test_index_copy
    self.assertEqual(dest, dest2)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "/Users/malfet/git/pytorch/pytorch/torch/testing/_internal/common_utils.py", line 4179, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
    ...<4 lines>...
    )
AssertionError: Tensor-likes are not close!

Mismatched elements: 60 / 100 (60.0%)
Greatest absolute difference: 15.180288314819336 at index (0, 1, 2) (up to 1e-05 allowed)
Greatest relative difference: 54.467315673828125 at index (0, 2, 0) (up to 1.3e-06 allowed)