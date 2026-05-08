Traceback (most recent call last):
  File "/var/lib/jenkins/pytorch/test/test_foreach.py", line 1061, in test_big_num_tensors
    actual = fn(
  File "/var/lib/jenkins/pytorch/test/test_foreach.py", line 99, in __call__
    assert mta_called == (expect_fastpath and (not zero_size)), (
AssertionError: mta_called=False, expect_fastpath=True, zero_size=False, self.func.__name__='_foreach_max', keys=('aten::_foreach_max', 'aten::zeros', 'aten::empty', 'aten::zero_', 'aten::fill_', 'hipLaunchKernel', 'void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<double>, std::array<char*, 1ul> >(int, at::native::FillFunctor<double>, std::array<char*, 1ul>) [clone .kd]', 'hipDeviceSynchronize')

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_ROCM=1 python test/test_foreach.py TestForeachCUDA.test_big_num_tensors__foreach_max_use_cuda_graph_False_w_empty_True_cuda_float64

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0