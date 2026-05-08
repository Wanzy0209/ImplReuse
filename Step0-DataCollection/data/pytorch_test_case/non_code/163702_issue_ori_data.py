root@781293c9ba63:/opt/pytorch/pytorch# python test/inductor/test_torchinductor_strided_blocks.py TritonTensorDescriptorTestCUDA.test_broadcast_prefer_nd_tiling_False_x_size1_y_size1_cuda
/usr/local/lib/python3.12/dist-packages/hypothesis/entry_points.py:23: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources

  warnings.warn(
W0923 15:23:30.180000 2725 torch/_inductor/utils.py:1566] Not enough SMs to use max_autotune_gemm mode
C0923 15:23:30.950000 2725 torch/_inductor/scheduler.py:1394] [0/0] Error in codegen for ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[8, 8], stride=[8, 1]), data=Pointwise(device=device(type='cuda', index=0), dtype=torch.float32, inner_fn=<function make_pointwise.<locals>.inner.<locals>.inner_fn at 0xf1cefbabd760>, ranges=[8, 8]))
Eframes [('total', 1)]
stats [('calls_captured', 3)]
inductor [('fxgraph_cache_miss', 1)]
aot_autograd [('total', 1), ('autograd_cache_miss', 1), ('not_ok', 1)]
graph_break []

======================================================================
ERROR: test_broadcast_prefer_nd_tiling_False_x_size1_y_size1_cuda (__main__.TritonTensorDescriptorTestCUDA.test_broadcast_prefer_nd_tiling_False_x_size1_y_size1_cuda)
Test that we can generate strided block pointers when inputs have different
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/torch/testing/_internal/common_utils.py", line 3223, in wrapper
    method(*args, **kwargs)
  File "/opt/pytorch/pytorch/test/inductor/test_torchinductor.py", line 14116, in new_test
    return value(self)
           ^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/testing/_internal/common_utils.py", line 552, in instantiated_test
    test(self, **param_kwargs)
  File "/opt/pytorch/pytorch/test/inductor/test_torchinductor_strided_blocks.py", line 322, in test_broadcast
    self._run_and_compare(
  File "/opt/pytorch/pytorch/test/inductor/test_torchinductor_strided_blocks.py", line 149, in _run_and_compare
    result, code = run_and_get_code(compiled, *args)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/utils.py", line 2179, in run_and_get_code
    result = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/eval_frame.py", line 899, in compile_wrapper
    raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 990, in _compile_fx_inner
    raise InductorError(e, currentframe()).with_traceback(
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 974, in _compile_fx_inner
    mb_compiled_graph = fx_codegen_and_compile(
                        ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 1697, in fx_codegen_and_compile
    return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/compile_fx.py", line 1507, in codegen_and_compile
    compiled_module = graph.compile_to_module()
                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/graph.py", line 2319, in compile_to_module
    return self._compile_to_module()
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/graph.py", line 2325, in _compile_to_module
    self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
                                                             ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/graph.py", line 2264, in codegen
    self.scheduler.codegen()
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/scheduler.py", line 5228, in codegen
    self._codegen_partitions()
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/scheduler.py", line 5369, in _codegen_partitions
    self._codegen(partition)
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/scheduler.py", line 5467, in _codegen
    self.get_backend(device).codegen_node(node)
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/codegen/cuda_combined_scheduling.py", line 127, in codegen_node
    return self._triton_scheduling.codegen_node(node)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/codegen/simd.py", line 1407, in codegen_node
    return self.codegen_node_schedule(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/codegen/simd.py", line 1465, in codegen_node_schedule
    self.codegen_node_schedule_with_kernel(node_schedule, kernel)
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/codegen/simd.py", line 1571, in codegen_node_schedule_with_kernel
    node.codegen(index_vars)
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/scheduler.py", line 1392, in codegen
    self._body(*index_vars)
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/loop_body.py", line 460, in __call__
    result = self.root_block()
             ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/loop_body.py", line 529, in __call__
    return InterpreterShim(graph, submodules).run(V.get_ops_handler())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/loop_body.py", line 60, in run
    return super().run(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/fx/interpreter.py", line 174, in run
    self.env[node] = self.run_node(node)
                     ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/loop_body.py", line 56, in run_node
    return super().run_node(n)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/fx/interpreter.py", line 256, in run_node
    return getattr(self, n.op)(n.target, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/fx/interpreter.py", line 360, in call_method
    return getattr(self_obj, target)(*args_tail, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/sizevars.py", line 1040, in store
    return self._inner.store(name, self._simplify(index), value, mode=mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/codegen/common.py", line 2703, in store
    self.kernel.store(name, index, value, mode=mode)
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/codegen/triton.py", line 2818, in store
    line = self.codegen_block_ptr_store_line(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/codegen/triton.py", line 2547, in codegen_block_ptr_store_line
    raise AssertionError(
torch._inductor.exc.InductorError: AssertionError: TMA store requires no broadcasting when a shape is provided

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


To execute this test, run the following from the base repo dir:
    python test/inductor/test_torchinductor_strided_blocks.py TritonTensorDescriptorTestCUDA.test_broadcast_prefer_nd_tiling_False_x_size1_y_size1_cuda

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0

----------------------------------------------------------------------
Ran 1 test in 0.490s

FAILED (errors=1)