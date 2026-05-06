root@e8faae565f19:/# python a.py
/usr/local/lib/python3.12/site-packages/torch/_subclasses/functional_tensor.py:279: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
  cpu = _conversion_method_template(device=torch.device("cpu"))
Traceback (most recent call last):
  File "//a.py", line 17, in <module>
    add_kernel = _compile_kernel(kernel_source, "add_tensors")
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/torch/cuda/__init__.py", line 1780, in _compile_kernel
    ptx = _nvrtc_compile(
          ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/torch/cuda/_utils.py", line 68, in _nvrtc_compile
    libnvrtc = _get_nvrtc_library()
               ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/site-packages/torch/cuda/_utils.py", line 38, in _get_nvrtc_library
    return ctypes.CDLL("libnvrtc.so")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/ctypes/__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: libnvrtc.so: cannot open shared object file: No such file or directory