Traceback (most recent call last):
  File "torch_reproducer.py", line 41, in <module>
    exported_program = torch.export.export(
  File "torch/export/__init__.py", line 319, in export
    raise e
  File "torch/export/__init__.py", line 286, in export
    return _export(
  File "torch/export/_trace.py", line 1164, in wrapper
    raise e
  File "torch/export/_trace.py", line 1130, in wrapper
    ep = fn(*args, **kwargs)
  File "torch/export/exported_program.py", line 123, in wrapper
    return fn(*args, **kwargs)
  File "torch/export/_trace.py", line 2176, in _export
    ep = _export_for_training(
  File "torch/export/_trace.py", line 1164, in wrapper
    raise e
  File "torch/export/_trace.py", line 1130, in wrapper
    ep = fn(*args, **kwargs)
  File "torch/export/exported_program.py", line 123, in wrapper
    return fn(*args, **kwargs)
  File "torch/export/_trace.py", line 2037, in _export_for_training
    export_artifact = export_func(
  File "torch/export/_trace.py", line 1501, in _strict_export
    aten_export_artifact = _to_aten_func(
  File "torch/export/_trace.py", line 1802, in _export_to_aten_ir_make_fx
    return _produce_aten_artifact(
  File "torch/export/_trace.py", line 547, in _produce_aten_artifact
    _replace_unbacked_bindings(gm)
  File "torch/export/_trace.py", line 503, in _replace_unbacked_bindings
    unbacked_bindings := _free_unbacked_symbols_with_path(
  File "torch/fx/experimental/symbolic_shapes.py", line 1154, in _free_unbacked_symbols_with_path
    assert isinstance(a, FakeTensor)
AssertionError