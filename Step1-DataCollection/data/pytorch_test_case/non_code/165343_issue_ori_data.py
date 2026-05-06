2025-10-13T17:10:54.3136670Z === Found 1 errors ===�[39;49;00m
2025-10-13T17:10:54.3136780Z �[31m--- Error: 1 / 1 ---�[39;49;00m
2025-10-13T17:10:54.3136830Z     * REASON: RuntimeError
2025-10-13T17:10:54.3136880Z     DOCTEST DEBUG INFO
2025-10-13T17:10:54.3137280Z       XDoc "/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/torch/distributed/tensor/_dtensor_spec.py::ShardOrderEntry:0", line 1 <- wrt doctest
2025-10-13T17:10:54.3137640Z       File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/torch/distributed/tensor/_dtensor_spec.py", line 30, <- wrt source file
2025-10-13T17:10:54.3137700Z     DOCTEST PART BREAKDOWN
2025-10-13T17:10:54.3137750Z     Passed Parts:
2025-10-13T17:10:54.3137860Z         1 >>> # Tensor dim 1 sharded across mesh dim 2, then mesh dim 0
2025-10-13T17:10:54.3137950Z         2 >>> ShardOrderEntry(tensor_dim=1, mesh_dims=(2, 0))
2025-10-13T17:10:54.3138030Z         4 >>> # Tensor dim 0 sharded only on mesh dim 1
2025-10-13T17:10:54.3138120Z         5 >>> ShardOrderEntry(tensor_dim=0, mesh_dims=(1,))
2025-10-13T17:10:54.3138170Z     DOCTEST TRACEBACK
2025-10-13T17:10:54.3138240Z     Traceback (most recent call last):
2025-10-13T17:10:54.3138280Z     
2025-10-13T17:10:54.3138620Z       File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/xdoctest/utils/util_import.py", line 212, in _custom_import_modpath
2025-10-13T17:10:54.3138700Z         module = import_module_from_name(modname)
2025-10-13T17:10:54.3138760Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-10-13T17:10:54.3138800Z     
2025-10-13T17:10:54.3139150Z       File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/xdoctest/utils/util_import.py", line 423, in import_module_from_name
2025-10-13T17:10:54.3139220Z         module = importlib.import_module(modname)
2025-10-13T17:10:54.3139280Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-10-13T17:10:54.3139320Z     
2025-10-13T17:10:54.3139670Z       File "/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/importlib/__init__.py", line 90, in import_module
2025-10-13T17:10:54.3139780Z         return _bootstrap._gcd_import(name[level:], package, level)
2025-10-13T17:10:54.3139860Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-10-13T17:10:54.3139900Z     
2025-10-13T17:10:54.3140020Z       File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
2025-10-13T17:10:54.3140060Z     
2025-10-13T17:10:54.3140180Z       File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
2025-10-13T17:10:54.3140230Z     
2025-10-13T17:10:54.3140360Z       File "<frozen importlib._bootstrap>", line 1310, in _find_and_load_unlocked
2025-10-13T17:10:54.3140410Z     
2025-10-13T17:10:54.3140540Z       File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
2025-10-13T17:10:54.3140580Z     
2025-10-13T17:10:54.3140700Z       File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
2025-10-13T17:10:54.3140790Z     
2025-10-13T17:10:54.3140910Z       File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
2025-10-13T17:10:54.3140950Z     
2025-10-13T17:10:54.3141090Z       File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
2025-10-13T17:10:54.3141130Z     
2025-10-13T17:10:54.3141290Z       File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
2025-10-13T17:10:54.3141330Z     
2025-10-13T17:10:54.3141460Z       File "<frozen importlib._bootstrap_external>", line 999, in exec_module
2025-10-13T17:10:54.3141500Z     
2025-10-13T17:10:54.3141640Z       File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
2025-10-13T17:10:54.3141680Z     
2025-10-13T17:10:54.3142010Z       File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/torch/distributed/tensor/__init__.py", line 4, in <module>
2025-10-13T17:10:54.3142170Z         import torch.distributed.tensor._ops  # force import all built-in dtensor ops
2025-10-13T17:10:54.3142260Z         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-10-13T17:10:54.3142300Z     
2025-10-13T17:10:54.3142630Z       File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/torch/distributed/tensor/_ops/__init__.py", line 2, in <module>
2025-10-13T17:10:54.3142710Z         from ._conv_ops import *  # noqa: F403
2025-10-13T17:10:54.3142760Z         ^^^^^^^^^^^^^^^^^^^^^^^^
2025-10-13T17:10:54.3142800Z     
2025-10-13T17:10:54.3143130Z       File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/torch/distributed/tensor/_ops/_conv_ops.py", line 5, in <module>
2025-10-13T17:10:54.3143280Z         from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
2025-10-13T17:10:54.3143320Z     
2025-10-13T17:10:54.3143650Z       File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/torch/distributed/tensor/_dtensor_spec.py", line 8, in <module>
2025-10-13T17:10:54.3143770Z         from torch.distributed.tensor.placement_types import (
2025-10-13T17:10:54.3143810Z     
2025-10-13T17:10:54.3144150Z       File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/torch/distributed/tensor/placement_types.py", line 8, in <module>
2025-10-13T17:10:54.3144270Z         import torch.distributed._functional_collectives as funcol
2025-10-13T17:10:54.3144310Z     
2025-10-13T17:10:54.3144650Z       File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py", line 9, in <module>
2025-10-13T17:10:54.3144740Z         import torch.distributed.distributed_c10d as c10d
2025-10-13T17:10:54.3144780Z     
2025-10-13T17:10:54.3145110Z       File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 24, in <module>
2025-10-13T17:10:54.3145190Z         from torch._C._distributed_c10d import (
2025-10-13T17:10:54.3145230Z     
2025-10-13T17:10:54.3145420Z     ModuleNotFoundError: No module named 'torch._C._distributed_c10d'; 'torch._C' is not a package
2025-10-13T17:10:54.3145460Z     
2025-10-13T17:10:54.3145500Z     
2025-10-13T17:10:54.3145630Z     During handling of the above exception, another exception occurred:
2025-10-13T17:10:54.3145670Z     
2025-10-13T17:10:54.3145710Z     
2025-10-13T17:10:54.3145770Z     Traceback (most recent call last):
2025-10-13T17:10:54.3145810Z     
2025-10-13T17:10:54.3146100Z       File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/xdoctest/doctest_example.py", line 793, in run
2025-10-13T17:10:54.3146160Z         self._import_module()
2025-10-13T17:10:54.3146200Z     
2025-10-13T17:10:54.3146510Z       File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/xdoctest/doctest_example.py", line 599, in _import_module
2025-10-13T17:10:54.3146660Z         self.module = utils.import_module_from_path(self.modpath, index=-1)
2025-10-13T17:10:54.3146770Z                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-10-13T17:10:54.3146810Z     
2025-10-13T17:10:54.3147160Z       File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/xdoctest/utils/util_import.py", line 380, in import_module_from_path
2025-10-13T17:10:54.3147290Z         module = _custom_import_modpath(modpath, index=index)
2025-10-13T17:10:54.3147360Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-10-13T17:10:54.3147400Z     
2025-10-13T17:10:54.3147750Z       File "/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/xdoctest/utils/util_import.py", line 220, in _custom_import_modpath
2025-10-13T17:10:54.3147820Z         raise RuntimeError('\n'.join(msg_parts))
2025-10-13T17:10:54.3147860Z     
2025-10-13T17:10:54.3148650Z     RuntimeError: ERROR: Failed to import modname=torch.distributed.tensor._dtensor_spec with modpath=/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages/torch/distributed/tensor/_dtensor_spec.py and sys.path modified with '/Users/ec2-user/runner/_work/_temp/venv-3.12-1760375316/lib/python3.12/site-packages' at index=-1
2025-10-13T17:10:54.3148910Z     Caused by: ModuleNotFoundError("No module named 'torch._C._distributed_c10d'; 'torch._C' is not a package")