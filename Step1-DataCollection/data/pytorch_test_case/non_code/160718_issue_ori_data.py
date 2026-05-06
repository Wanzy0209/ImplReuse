Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/inductor/test_cudagraph_trees.py", line 2894, in test_graph_partition_cpu_scalar_multiple
    self.assertEqual(self.get_manager().new_graph_id().id, 1)
AttributeError: 'NoneType' object has no attribute 'new_graph_id'

To execute this test, run the following from the base repo dir:
    python test/inductor/test_cudagraph_trees.py CudaGraphTreeTests.test_graph_partition_cpu_scalar_multiple

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0