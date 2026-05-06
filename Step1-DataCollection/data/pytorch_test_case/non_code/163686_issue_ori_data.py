ERROR    __main__:test_public_bindings.py:287 import_module failed
Traceback (most recent call last):
  File "/pytorch/test/test_public_bindings.py", line 284, in test_modules_can_be_imported
    importlib.import_module(modname)
  File "/miniconda/envs/torch_build/lib/python3.9/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1030, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1007, in _find_and_load
  File "<frozen importlib._bootstrap>", line 986, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 680, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 850, in exec_module
  File "<frozen importlib._bootstrap>", line 228, in _call_with_frames_removed
  File "/pytorch/torch/_export/db/gen_example.py", line 4, in <module>
    import torch._export.db.examples as examples
  File "/pytorch/torch/_export/db/examples/__init__.py", line 30, in <module>
    _collect_examples()
  File "/pytorch/torch/_export/db/examples/__init__.py", line 28, in _collect_examples
    export_case(**{v: getattr(case, v) for v in variables})(case.model)
  File "/pytorch/torch/_export/db/case.py", line 147, in wrapper
    raise RuntimeError("export_case should only be used once per example file.")
RuntimeError: export_case should only be used once per example file.

Traceback (most recent call last):
  File "/pytorch/test/test_public_bindings.py", line 418, in test_modules_can_be_imported
    self.assertEqual("", "\n".join(errors))
  File "/pytorch/torch/testing/_internal/common_utils.py", line 4178, in assertEqual
    raise error_metas.pop()[0].to_error(  # type: ignore[index]
AssertionError: String comparison failed: '' != "torch._export.db.examples failed to impo[626 chars]ion)"


ERROR    __main__:test_public_bindings.py:287 import_module failed
Traceback (most recent call last):
  File "/pytorch/test/test_public_bindings.py", line 284, in test_modules_can_be_imported
    importlib.import_module(modname)
  File "/miniconda/envs/torch_build/lib/python3.9/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1030, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1007, in _find_and_load
  File "<frozen importlib._bootstrap>", line 986, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 680, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 850, in exec_module
  File "<frozen importlib._bootstrap>", line 228, in _call_with_frames_removed
  File "/pytorch/torch/testing/_internal/hop_db.py", line 7, in <module>
    from functorch.experimental.control_flow import map
  File "/pytorch/functorch/experimental/__init__.py", line 2, in <module>
    from functorch import functionalize
ImportError: cannot import name 'functionalize' from 'functorch' (unknown location)