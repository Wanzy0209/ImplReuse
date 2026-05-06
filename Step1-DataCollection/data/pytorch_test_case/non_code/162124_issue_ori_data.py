Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/inductor/test_cudagraph_trees.py", line 3948, in test_cudagraph_or_error
    with self.assertRaises(torch._dynamo.exc.Unsupported):
  File "/opt/conda/envs/py_3.10/lib/python3.10/unittest/case.py", line 226, in __exit__
    self._raiseFailure("{} not raised".format(exc_name))
  File "/opt/conda/envs/py_3.10/lib/python3.10/unittest/case.py", line 163, in _raiseFailure
    raise self.test_case.failureException(msg)
AssertionError: Unsupported not raised

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_CUDA_MEM_LEAK_CHECK=1 PYTORCH_TEST_WITH_SLOW_GRADCHECK=1 python test/inductor/test_cudagraph_trees.py CudaGraphTreeTests.test_cudagraph_or_error

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0