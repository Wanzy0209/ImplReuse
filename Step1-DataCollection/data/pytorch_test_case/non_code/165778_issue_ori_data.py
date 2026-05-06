Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3293, in wrapper
    method(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3293, in wrapper
    method(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3292, in wrapper
    with policy():
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 2669, in __exit__
    raise RuntimeError(msg)
RuntimeError: CUDA driver API confirmed a leak in __main__.ActivationCheckpointingViaTagsTestsCUDA.test_compile_selective_checkpoint_must_not_recompute_gemm_no_functionalization_cuda! Caching allocator allocated memory was 0 and is now reported as 2048 on device 0. CUDA driver allocated memory was 257949696 and is now 276824064.

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_CUDA_MEM_LEAK_CHECK=1 python test/dynamo/test_activation_checkpointing.py ActivationCheckpointingViaTagsTestsCUDA.test_compile_selective_checkpoint_must_not_recompute_gemm_no_functionalization_cuda

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0