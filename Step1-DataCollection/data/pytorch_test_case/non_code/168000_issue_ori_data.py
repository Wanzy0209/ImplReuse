jenkins@2314b9b1cd5d:/pytorch$ pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
Looking in indexes: https://download.pytorch.org/whl/nightly/cpu
jenkins@2314b9b1cd5d:/pytorch$ python test/inductor/test_torchinductor.py CpuTests.test_to_dtype_cpu
Traceback (most recent call last):
  File "/pytorch/test/inductor/test_torchinductor.py", line 30, in <module>
    import torch
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/__init__.py", line 433, in <module>
    from torch._C import *  # noqa: F403
ImportError: libopenblas.so.0: cannot open shared object file: No such file or directory