Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/export/test_export.py", line 4449, in test_detect_leak_nonstrict
    with (
  File "/opt/conda/envs/py_3.10/lib/python3.10/unittest/case.py", line 289, in __exit__
    self._raiseFailure('"{}" does not match "{}"'.format(
  File "/opt/conda/envs/py_3.10/lib/python3.10/unittest/case.py", line 163, in _raiseFailure
    raise self.test_case.failureException(msg)
AssertionError: "Detected 3 fake tensors that are still alive after export" does not match "Detected 2 fake tensors that are still alive after export.
This is likely result of torch.export.export not being able to track side effects that is happening outside of model scope.

Leaked tensors:
  FakeTensor(shape=torch.Size([4, 4]), dtype=torch.float32): <unknown stack trace>
  FakeTensor(shape=torch.Size([4, 4]), dtype=torch.float32): <unknown stack trace>

Alternatively, please file a bug report to PyTorch team for further debugging help."

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_ASAN=1 PYTORCH_TEST_WITH_UBSAN=1 python test/export/test_serdes.py SerDesExportNonStrictTestExport.test_detect_leak_nonstrict_serdes_nonstrict

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0