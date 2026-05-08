Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/export/test_export.py", line 742, in test_annotate_on_assert
    self.assertExpectedInline(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3277, in assertExpectedInline
    return super().assertExpectedInline(actual if isinstance(actual, str) else str(actual), expect, skip + 1)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/expecttest/__init__.py", line 413, in assertExpectedInline
    assert_expected_inline(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/expecttest/__init__.py", line 378, in assert_expected_inline
    assert_eq(expect, actual, msg=help_text)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/expecttest/__init__.py", line 450, in assertMultiLineEqualMaybeCppStack
    self.assertMultiLineEqual(expect, actual, *args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/unittest/case.py", line 1226, in assertMultiLineEqual
    self.fail(self._formatMessage(msg, standardMsg))
  File "/opt/conda/envs/py_3.10/lib/python3.10/unittest/case.py", line 675, in fail
    raise self.failureException(msg)
AssertionError: "('pl[115 chars]d': 'dispatch_trace'})\n('placeholder', 'y', {[1064 chars]e'})" != "('pl[115 chars]d': 'trace'})\n('placeholder', 'y', {'_torchdy[992 chars]e'})"
- ('placeholder', 'x', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'dispatch_trace'})
?                                                                                                                             ---------
+ ('placeholder', 'x', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'trace'})
- ('placeholder', 'y', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'dispatch_trace'})
?                                                                                                                             ---------
+ ('placeholder', 'y', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'trace'})
- ('call_function', 'cat', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'dispatch_trace', 'moo': 0})
?                                                                                                                                 ---------
+ ('call_function', 'cat', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'trace', 'moo': 0})
- ('call_function', 'item', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'dispatch_trace', 'moo': 0})
?                                                                                                                                  ---------
+ ('call_function', 'item', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'trace', 'moo': 0})
- ('call_function', 'ge_1', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'dispatch_trace', 'moo': 0})
?                                                                                                                                  ---------
+ ('call_function', 'ge_1', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'trace', 'moo': 0})
- ('call_function', '_assert_scalar_default', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'dispatch_trace', 'moo': 0})
?                                                                                                                                                    ---------
+ ('call_function', '_assert_scalar_default', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'trace', 'moo': 0})
- ('call_function', 'mul', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'dispatch_trace', 'moo': 0})
?                                                                                                                                 ---------
+ ('call_function', 'mul', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'trace', 'moo': 0})
- ('output', 'output', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'dispatch_trace'})?                                                                                                                             ---------
+ ('output', 'output', {'_torchdynamo_disable': True, '_torchdynamo_disable_recursive': True, '_torchdynamo_disable_method': 'trace'}) : To accept the new output, re-run test with envvar EXPECTTEST_ACCEPT=1 (we recommend staging/committing your changes before doing this)

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_CUDA_MEM_LEAK_CHECK=1 python test/export/test_export_training_ir_to_run_decomp.py TrainingIRToRunDecompExportNonStrictTestExport.test_annotate_on_assert_training_ir_to_decomp_nonstrict

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0