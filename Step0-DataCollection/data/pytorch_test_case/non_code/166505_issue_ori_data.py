Traceback (most recent call last):
  File "/var/lib/jenkins/pytorch/test/inductor/test_inductor_utils.py", line 29, in test_do_bench_using_profiling
    res = do_bench_using_profiling(self._bench_fn)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/utils.py", line 296, in do_bench_using_profiling
    return may_distort_benchmarking_result(_do_bench_using_profiling)(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/utils.py", line 370, in _do_bench_using_profiling
    raise RuntimeError(
RuntimeError: ('Failed to divide all profiling events into #repeat groups. #CUDA events: %d, #repeats: %s\n\nTo execute this test, run the following from the base repo dir:\n    PYTORCH_TEST_WITH_ROCM=1 python test/inductor/test_inductor_utils.py TestBench.test_do_bench_using_profiling\n\nThis message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0', 1001, 501)