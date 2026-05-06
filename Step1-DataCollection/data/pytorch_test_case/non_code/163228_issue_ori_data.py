Traceback (most recent call last):
  File "/var/lib/jenkins/pytorch/test/inductor/test_cuda_select_algorithm.py", line 125, in test_int8_woq_mm_cuda
    self.assertEqual(counters["inductor"]["woq_matcher_count"], 1)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_dynamo/test_case.py", line 111, in assertEqual
    return super().assertEqual(x, y, *args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 4178, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
AssertionError: Scalars are not equal!

Expected 1 but got 0.
Absolute difference: 1
Relative difference: 1.0

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_ROCM=1 python test/inductor/test_cuda_select_algorithm.py TestSelectAlgorithmCudaCUDA.test_int8_woq_mm_cuda_batch_size_17_mid_dim_8_in_features_128_out_features_65_cuda_bfloat16

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0