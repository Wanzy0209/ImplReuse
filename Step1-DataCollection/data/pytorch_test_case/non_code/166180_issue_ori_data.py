File "<ta-01K8BA92H6RT7D4R3V6CBA2Q9T>:/usr/local/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
  File "<ta-01K8BA92H6RT7D4R3V6CBA2Q9T>:/usr/local/lib/python3.12/site-packages/diffusers/pipelines/wan/pipeline_wan_i2v.py", line 756, in __call__
  File "<ta-01K8BA92H6RT7D4R3V6CBA2Q9T>:/usr/local/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
  File "<ta-01K8BA92H6RT7D4R3V6CBA2Q9T>:/usr/local/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
  File "<ta-01K8BA92H6RT7D4R3V6CBA2Q9T>:/usr/local/lib/python3.12/site-packages/diffusers/models/transformers/transformer_wan.py", line 663, in forward
  File "<ta-01K8BA92H6RT7D4R3V6CBA2Q9T>:/usr/local/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
  File "<ta-01K8BA92H6RT7D4R3V6CBA2Q9T>:/usr/local/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
  File "<ta-01K8BA92H6RT7D4R3V6CBA2Q9T>:/usr/local/lib/python3.12/site-packages/spaces/zero/torch/aoti.py", line 77, in __call__
  File "<ta-01K8BA92H6RT7D4R3V6CBA2Q9T>:/usr/local/lib/python3.12/contextlib.py", line 137, in __enter__
    return next(self.gen)
^^^^^^^^^^^^^^^
  File "<ta-01K8BA92H6RT7D4R3V6CBA2Q9T>:/usr/local/lib/python3.12/site-packages/spaces/zero/torch/aoti.py", line 47, in _register_aoti_cleanup
  File "<ta-01K8BA92H6RT7D4R3V6CBA2Q9T>:/usr/local/lib/python3.12/pathlib.py", line 1056, in iterdir
    for name in os.listdir(self):
  ^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/proc/2/map_files'