Traceback (most recent call last):
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/testing/_internal/common_device_type.py", line 1135, in test_wrapper
    return test(*args, **kwargs)
  File "/var/lib/jenkins/workspace/test/export/test_export_opinfo.py", line 129, in test_fake_export
    _test_export_helper(self, dtype, op)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/var/lib/jenkins/workspace/test/export/test_export_opinfo.py", line 112, in _test_export_helper
    ep = torch.export.export(m, args)
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/export/__init__.py", line 307, in export
    raise e
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/export/__init__.py", line 274, in export
    return _export(
        mod,
    ...<5 lines>...
        pre_dispatch=True,
    )
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/export/_trace.py", line 1158, in wrapper
    raise e
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/export/_trace.py", line 1124, in wrapper
    ep = fn(*args, **kwargs)
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/export/exported_program.py", line 124, in wrapper
    return fn(*args, **kwargs)
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/export/_trace.py", line 2192, in _export
    ep = _export_for_training(
        mod,
    ...<4 lines>...
        preserve_module_call_signature=preserve_module_call_signature,
    )
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/export/_trace.py", line 1158, in wrapper
    raise e
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/export/_trace.py", line 1124, in wrapper
    ep = fn(*args, **kwargs)
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/export/exported_program.py", line 124, in wrapper
    return fn(*args, **kwargs)
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/export/_trace.py", line 2055, in _export_for_training
    export_artifact = export_func(
        mod=mod,
    ...<6 lines>...
        _to_aten_func=_export_to_aten_ir_make_fx,
    )
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/export/_trace.py", line 1997, in _non_strict_export
    aten_export_artifact = _to_aten_func(  # type: ignore[operator]
        patched_mod,
    ...<5 lines>...
        transform=_tuplify_outputs,
    )
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/export/_trace.py", line 1820, in _export_to_aten_ir_make_fx
    return _produce_aten_artifact(
        gm=gm,
    ...<6 lines>...
        fake_params_buffers=fake_params_buffers,
    )
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/export/_trace.py", line 581, in _produce_aten_artifact
    gm, export_graph_signature = replace_set_grad_with_hop_pass(
                                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        gm, export_graph_signature
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/_export/passes/replace_set_grad_with_hop_pass.py", line 117, in replace_set_grad_with_hop_pass
    return _replace_with_hop_pass_helper(
        gm,
        graph_signature,
        _sequential_split_and_maybe_inline_subgraphs,
    )
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/_export/passes/replace_with_hop_pass_util.py", line 169, in _replace_with_hop_pass_helper
    new_gm, new_signature = sequential_split_and_maybe_inline_subgraphs(
                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        gm, graph_signature
        ^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/_export/passes/replace_set_grad_with_hop_pass.py", line 105, in _sequential_split_and_maybe_inline_subgraphs
    return _sequential_split_and_maybe_inline_subgraphs_helper(
        new_gm, graph_signature, _maybe_inline_or_replace_with_hop
    )
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/_export/passes/replace_with_hop_pass_util.py", line 126, in _sequential_split_and_maybe_inline_subgraphs_helper
    new_gm_out_node = next(reversed(new_gm.graph.find_nodes(op="output")))
StopIteration

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/testing/_internal/common_utils.py", line 3224, in wrapper
    method(*args, **kwargs)
    ~~~~~~^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/testing/_internal/common_device_type.py", line 426, in instantiated_test
    result = test(self, **param_kwargs)
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/testing/_internal/common_utils.py", line 1645, in wrapper
    fn(*args, **kwargs)
    ~~^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/py_3.13/lib/python3.13/site-packages/torch/testing/_internal/common_device_type.py", line 1147, in test_wrapper
    raise e_tracked from e
Exception: Caused by sample input at index 0: SampleInput(input=Tensor[size=(5,), device="cpu", dtype=torch.float32], args=(), kwargs={'dim': 'None', 'spacing': 'None', 'edge_order': '1'}, broadcasts_input=False, name='')

To execute this test, run the following from the base repo dir:
    PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=0 python test/export/test_export_opinfo.py TestExportOpInfoCPU.test_fake_export_gradient_cpu_float32

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0