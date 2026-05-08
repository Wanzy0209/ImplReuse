Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1791, in wrapper
    raise rv
RuntimeError: Exception in worker process:
Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1658, in _worker_loop
    cls._run_test_given_id(test_id)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1631, in _run_test_given_id
    test_fn(**kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1800, in wrapper
    fn()
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3223, in wrapper
    method(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 552, in instantiated_test
    test(self, **param_kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 1952, in wrap_fn
    raise unittest.SkipTest(reason)
unittest.case.SkipTest: skipIfRocm: test only runs on ('gfx942',)