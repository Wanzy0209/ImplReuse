Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/dynamo/test_modes.py", line 462, in test_torch_function_mode_graph_break
    actual = fn_opt(*inp)
  File "/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/_dynamo/eval_frame.py", line 817, in compile_wrapper
    raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
  File "/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/_inductor/compile_fx.py", line 987, in _compile_fx_inner
    raise InductorError(e, currentframe()).with_traceback(
  File "/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/_inductor/compile_fx.py", line 971, in _compile_fx_inner
    mb_compiled_graph = fx_codegen_and_compile(
  File "/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/_inductor/compile_fx.py", line 1673, in fx_codegen_and_compile
    return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
  File "/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/_inductor/compile_fx.py", line 1525, in codegen_and_compile
    compiled_module = graph.compile_to_module()
  File "/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/_inductor/graph.py", line 2318, in compile_to_module
    return self._compile_to_module()
  File "/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/_inductor/graph.py", line 2324, in _compile_to_module
    self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
  File "/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/_inductor/graph.py", line 2270, in codegen
    result = self.wrapper_code.generate(self.is_inference)
  File "/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/_inductor/codegen/wrapper.py", line 1529, in generate
    return self._generate(is_inference)
  File "/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/_inductor/codegen/wrapper.py", line 1564, in _generate
    self.run_wrapper_ir_passes(is_inference)
  File "/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/_inductor/codegen/wrapper.py", line 1717, in run_wrapper_ir_passes
    self.estimate_peak = EfficientPeakEstimate()
  File "/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/_inductor/codegen/wrapper.py", line 607, in __init__
    self.segmented_tree = SegmentedTree(
  File "/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/_inductor/codegen/segmented_tree.py", line 32, in __init__
    raise ValueError("Cannot create a segment tree with empty values list")
torch._inductor.exc.InductorError: ValueError: Cannot create a segment tree with empty values list

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_SLOW=1 PYTORCH_TEST_SKIP_FAST=1 python test/dynamo/test_modes.py TorchFunctionModeTests.test_torch_function_mode_graph_break

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0