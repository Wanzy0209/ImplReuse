Count: 1
Traceback (most recent call last):
  File "/home/luka/git/vllm/._workspace/multi-output-add.py", line 88, in <module>
    print(my_func_static(*inputs))
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 749, in compile_wrapper
    raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1871, in _call_user_compiler
    raise BackendCompilerFailed(
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 1846, in _call_user_compiler
    compiled_fn = compiler_fn(gm, example_inputs)
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_dynamo/repro/after_dynamo.py", line 150, in __call__
    compiled_gm = compiler_fn(gm, example_inputs)
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/__init__.py", line 2380, in __call__
    return compile_fx(model_, inputs_, config_patches=self.config)
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 2002, in compile_fx
    return compile_fx(
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 2418, in compile_fx
    return aot_autograd(
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_dynamo/backends/common.py", line 109, in __call__
    cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 1199, in aot_module_simplified
    compiled_fn = AOTAutogradCache.load(
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/autograd_cache.py", line 1140, in load
    compiled_fn = dispatch_and_compile()
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 1184, in dispatch_and_compile
    compiled_fn, _ = create_aot_dispatcher_function(
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 576, in create_aot_dispatcher_function
    return _create_aot_dispatcher_function(
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 836, in _create_aot_dispatcher_function
    compiled_fn, fw_metadata = compiler_fn(
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 245, in aot_dispatch_base
    compiled_fw = compiler(fw_module, updated_flat_args)
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 483, in __call__
    return self.compiler_fn(gm, example_inputs)
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 2250, in fw_compiler_base
    return inner_compile(
  File "/usr/lib/python3.10/contextlib.py", line 79, in inner
    return func(*args, **kwds)
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 745, in compile_fx_inner
    return wrap_compiler_debug(_compile_fx_inner, compiler_name="inductor")(
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_dynamo/repro/after_aot.py", line 124, in debug_wrapper
    inner_compiled_fn = compiler_fn(gm, example_inputs)
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 896, in _compile_fx_inner
    mb_compiled_graph = fx_codegen_and_compile(
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 1578, in fx_codegen_and_compile
    return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 1236, in codegen_and_compile
    _recursive_post_grad_passes(gm, is_inference=is_inference)
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 504, in _recursive_post_grad_passes
    post_grad_passes(gm, is_inference)
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/_inductor/fx_passes/post_grad.py", line 181, in post_grad_passes
    GraphTransformObserver(gm, "post_grad_custom_post_pass").apply_graph_pass(
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/fx/passes/graph_transform_observer.py", line 85, in apply_graph_pass
    return pass_fn(self.gm.graph)
  File "/home/luka/git/vllm/._workspace/multi-output-add.py", line 71, in custom_pass
    graph.eliminate_dead_code()
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/fx/graph.py", line 1793, in eliminate_dead_code
    self.lint()
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/fx/graph.py", line 1701, in lint
    check_arg(arg, node)
  File "/home/luka/git/vllm/.venv/lib/python3.10/site-packages/torch/fx/graph.py", line 1686, in check_arg
    raise RuntimeError(
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Argument 'permute' of Node 'auto_functionalized_2' was used before it has been defined! Please check that Nodes in the graph are topologically ordered
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%arg0_1,), kwargs = {})
    %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%relu,), kwargs = {})
    %auto_functionalized_2 : [num_users=2] = call_function[target=torch.ops.higher_order.auto_functionalized](args = (vllm.fused_add_rms_norm_quant.default,), kwargs = {result: %permute, input: %relu, weight: %arg1_1, residual: %sqrt, epsilon: 1e-06, scale: %full_default})
    %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%auto_functionalized_2, 1), kwargs = {})
    %getitem_6 : [num_users=1] = call_function[target=operator.getitem](args = (%auto_functionalized_2, 2), kwargs = {})
    %empty : [num_users=1] = call_function[target=torch.ops.aten.empty.memory_format](args = ([5, 4],), kwargs = {dtype: torch.float8_e4m3fn, layout: torch.strided, device: cuda:0, pin_memory: False})
    %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%empty, [0, 1]), kwargs = {})
    %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
    return (getitem_5, getitem_6)

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"