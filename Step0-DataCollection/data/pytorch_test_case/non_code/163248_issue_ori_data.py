Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/inductor/test_pcache.py", line 131, in test_str_bytes_get_insert_thread_safe
    self.assertIsEqual(get_result, value)
    ^^^^^^^^^^^^^^^^^^
AttributeError: 'CacheTest' object has no attribute 'assertIsEqual'

To execute this test, run the following from the base repo dir:
    python test/inductor/test_pcache.py CacheTest.test_str_bytes_get_insert_thread_safe_Cache0

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0. Did you mean: 'assertEqual'?