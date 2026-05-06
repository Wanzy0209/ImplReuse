+ conda run -p /__w/_temp/conda_environment_16907649430 python setup.py clean
Traceback (most recent call last):
  File "/__w/vision/vision/pytorch/vision/setup.py", line 12, in <module>
    import torch
  File "/__w/_temp/conda_environment_16907649430/lib/python3.12/site-packages/torch/__init__.py", line 415, in <module>
    from torch._C import *  # noqa: F403
    ^^^^^^^^^^^^^^^^^^^^^^
ImportError: /lib64/libm.so.6: version `GLIBC_2.29' not found (required by /usr/local/cuda/lib64/libnvshmem_host.so.3)

ERROR conda.cli.main_run:execute(125): `conda run python setup.py clean` failed. (See above for error)