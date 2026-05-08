Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/dynamo/test_export.py", line 4056, in test_retracibility_nested_list_out
    self.assertTrue(torch.allclose(gm(inp_test)[0][0], gm2(inp_test)[0][0]))
  File "/opt/conda/envs/py_3.10/lib/python3.10/unittest/case.py", line 687, in assertTrue
    raise self.failureException(msg)
AssertionError: False is not true

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_ASAN=1 PYTORCH_TEST_WITH_UBSAN=1 PYTORCH_TEST_WITH_SLOW=1 PYTORCH_TEST_SKIP_FAST=1 python test/dynamo/test_dynamic_shapes.py DynamicShapesExportTests.test_retracibility_nested_list_out_dynamic_shapes

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0