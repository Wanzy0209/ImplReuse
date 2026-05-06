File "/usr/lib/python3.10/unittest/case.py", line 59, in testPartExecutor
    yield
  File "/usr/lib/python3.10/unittest/case.py", line 591, in run
    self._callTestMethod(testMethod)
  File "/usr/lib/python3.10/unittest/case.py", line 549, in _callTestMethod
    method()
  File "/torch/venv3/pytorch/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3191, in wrapper
    method(*args, **kwargs)
  File "/torch/venv3/pytorch/lib/python3.10/site-packages/torch_mlu_overrides/case_overrides.py", line 647, in instantiated_test
    test(self, **param_kwargs)
  File "/torch/src/pytorch/test/test_multiprocessing_spawn.py", line 162, in test_terminate_exit
    self.assertIn(expected_log, logs.records[0].getMessage())
IndexError: list index out of range 

To execute this test, run the following from the base repo dir:
    python test/test_multiprocessing_spawn.py SpawnTest.test_terminate_exit_grace_period0