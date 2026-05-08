torch._dynamo.exc.BackendCompilerFailed: backend='aot_eager' raised:
RuntimeError: function GeneratedBackwardFor_mylib_foo_defaultBackward returned a gradient different than None at position 2, but the corresponding forward input was not a Variable

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/dynamo/test_aot_autograd.py", line 1227, in test_data_ptr_access_fails_in_backward
    with self.assertRaisesRegex(RuntimeError, "Cannot access data pointer"):
  File "/opt/conda/envs/py_3.10/lib/python3.10/unittest/case.py", line 239, in __exit__
    self._raiseFailure('"{}" does not match "{}"'.format(
  File "/opt/conda/envs/py_3.10/lib/python3.10/unittest/case.py", line 163, in _raiseFailure
    raise self.test_case.failureException(msg)
AssertionError: "Cannot access data pointer" does not match "backend='aot_eager' raised:
RuntimeError: function GeneratedBackwardFor_mylib_foo_defaultBackward returned a gradient different than None at position 2, but the corresponding forward input was not a Variable

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
"

To execute this test, run the following from the base repo dir:
    python test/dynamo/test_dynamic_shapes.py DynamicShapesAotAutogradFallbackTests.test_data_ptr_access_fails_in_backward_dynamic_shapes

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0