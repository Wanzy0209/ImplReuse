$ python test/run_test.py -i export/test_export_opinfo --exclude-distributed-tests -- -k test_fake_export___radd___cpu_float32

Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/torch/testing/_internal/common_device_type.py", line 1135, in test_wrapper
    return test(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/pytorch/pytorch/test/export/test_export_opinfo.py", line 133, in test_fake_export
    _test_export_helper(self, dtype, op)
  File "/opt/pytorch/pytorch/test/export/test_export_opinfo.py", line 116, in _test_export_helper
    ep = torch.export.export(m, args)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/export/__init__.py", line 311, in export
    raise e
  File "/usr/local/lib/python3.12/dist-packages/torch/export/__init__.py", line 277, in export
    return _export(
           ^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/export/_trace.py", line 1177, in wrapper
    raise e
  File "/usr/local/lib/python3.12/dist-packages/torch/export/_trace.py", line 1143, in wrapper
    ep = fn(*args, **kwargs)
         ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/export/exported_program.py", line 124, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/export/_trace.py", line 2269, in _export
    ep = _export_for_training(
         ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/export/_trace.py", line 1177, in wrapper
    raise e
  File "/usr/local/lib/python3.12/dist-packages/torch/export/_trace.py", line 1143, in wrapper
    ep = fn(*args, **kwargs)
         ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/export/exported_program.py", line 124, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/export/_trace.py", line 2085, in _export_for_training
    export_artifact = export_func(
                      ^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/export/_trace.py", line 1971, in _non_strict_export
    ) = make_fake_inputs(
        ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_export/non_strict_utils.py", line 402, in make_fake_inputs
    fake_args, fake_kwargs = tree_map_with_path(
                             ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/_pytree.py", line 2056, in tree_map_with_path
    return treespec.unflatten(func(*xs) for xs in zip(*all_keypath_leaves))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/_pytree.py", line 1193, in unflatten
    leaves = list(leaves)
             ^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/_pytree.py", line 2056, in <genexpr>
    return treespec.unflatten(func(*xs) for xs in zip(*all_keypath_leaves))
                              ^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_export/non_strict_utils.py", line 403, in <lambda>
    lambda kp, val: fakify(
                    ^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_export/non_strict_utils.py", line 232, in fakify
    fake = mode.from_tensor(t, source=source, symbolic_context=symbolic_context)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_subclasses/fake_tensor.py", line 3004, in from_tensor
    return self.fake_tensor_converter.from_real_tensor(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_subclasses/fake_tensor.py", line 404, in from_real_tensor
    out = self.meta_converter(
          ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_subclasses/meta_utils.py", line 1922, in __call__
    r = self.meta_tensor(
        ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_subclasses/meta_utils.py", line 1698, in meta_tensor
    r = callback(
        ^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_subclasses/fake_tensor.py", line 395, in mk_fake_tensor
    return FakeTensor(
           ^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_subclasses/fake_tensor.py", line 744, in __new__
    init_gpu_context(device)
  File "/usr/local/lib/python3.12/dist-packages/torch/_subclasses/fake_tensor.py", line 519, in init_gpu_context
    torch.empty(1, device=device)
torch.AcceleratorError: CUDA error: invalid device ordinal
GPU device may be out of range, do you have enough GPUs?
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.