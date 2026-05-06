❯ pytest test/inductor/test_flex_attention.py -k test_kv_batch_broadcast_batch_dims2_head_dims0_score_mod1_cuda_float16
================================================================= test session starts ==================================================================
platform linux -- Python 3.12.11, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/drisspg/meta/pytorch
configfile: pytest.ini
plugins: hypothesis-6.136.6
collected 1406 items / 1405 deselected / 1 selected                                                                                                    
Running 1 items in this shard

test/inductor/test_flex_attention.py F                                                                                                           [100%]

======================================================================= FAILURES =======================================================================
_____________________________ TestFlexAttentionCUDA.test_kv_batch_broadcast_batch_dims2_head_dims0_score_mod1_cuda_float16 _____________________________
Traceback (most recent call last):
  File "/home/drisspg/.conda/envs/dev/lib/python3.12/unittest/case.py", line 58, in testPartExecutor
    yield
  File "/home/drisspg/.conda/envs/dev/lib/python3.12/unittest/case.py", line 634, in run
    self._callTestMethod(testMethod)
  File "/home/drisspg/.conda/envs/dev/lib/python3.12/unittest/case.py", line 589, in _callTestMethod
    if method() is not None:
       ^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/testing/_internal/common_utils.py", line 3224, in wrapper
    method(*args, **kwargs)
  File "/home/drisspg/meta/pytorch/torch/testing/_internal/common_utils.py", line 3224, in wrapper
    method(*args, **kwargs)
  File "/home/drisspg/meta/pytorch/torch/testing/_internal/common_device_type.py", line 426, in instantiated_test
    result = test(self, **param_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/testing/_internal/common_device_type.py", line 1349, in dep_fn
    return fn(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/test/inductor/test_flex_attention.py", line 1333, in test_kv_batch_broadcast
    self.run_test(
  File "/home/drisspg/meta/pytorch/test/inductor/test_flex_attention.py", line 566, in run_test
    compiled_out.backward(backward_grad)
  File "/home/drisspg/meta/pytorch/torch/_tensor.py", line 625, in backward
    torch.autograd.backward(
  File "/home/drisspg/meta/pytorch/torch/autograd/__init__.py", line 354, in backward
    _engine_run_backward(
  File "/home/drisspg/meta/pytorch/torch/autograd/graph.py", line 829, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/autograd/function.py", line 315, in apply
    return user_fn(self, *args)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 2303, in backward
    return impl_fn()
           ^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 2289, in impl_fn
    out = CompiledFunction._backward_impl(ctx, all_args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 2394, in _backward_impl
    CompiledFunction.compiled_bw = aot_config.bw_compiler(
                                   ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_functorch/_aot_autograd/schemas.py", line 1256, in __call__
    return self.compiler_fn(gm, example_inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_dynamo/backends/common.py", line 76, in _wrapped_bw_compiler
    disable(
  File "/home/drisspg/meta/pytorch/torch/_dynamo/eval_frame.py", line 1005, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_utils_internal.py", line 92, in wrapper_function
    return function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_inductor/compile_fx.py", line 2428, in bw_compiler
    return inner_compile(
           ^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_inductor/compile_fx.py", line 773, in compile_fx_inner
    return wrap_compiler_debug(_compile_fx_inner, compiler_name="inductor")(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_dynamo/repro/after_aot.py", line 124, in debug_wrapper
    inner_compiled_fn = compiler_fn(gm, example_inputs)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_inductor/compile_fx.py", line 952, in _compile_fx_inner
    mb_compiled_graph = fx_codegen_and_compile(
                        ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_inductor/compile_fx.py", line 1652, in fx_codegen_and_compile
    return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_inductor/compile_fx.py", line 1506, in codegen_and_compile
    compiled_module = graph.compile_to_module()
                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_inductor/graph.py", line 2318, in compile_to_module
    return self._compile_to_module()
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_inductor/graph.py", line 2328, in _compile_to_module
    mod = self._compile_to_module_lines(wrapper_code)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_inductor/graph.py", line 2396, in _compile_to_module_lines
    mod = PyCodeCache.load_by_key_path(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_inductor/codecache.py", line 3466, in load_by_key_path
    mod = _reload_python_module(key, path, set_sys_modules=in_toplevel)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_inductor/runtime/compile_tasks.py", line 33, in _reload_python_module
    exec(code, mod.__dict__, mod.__dict__)
  File "/tmp/tmprlw3l41j/tp/ctp6qddnfrq5q7wbgtikfvqpjkr53m7uz6orsfrjt4pukngwh7tu.py", line 1026, in <module>
  File "/home/drisspg/meta/pytorch/torch/_inductor/async_compile.py", line 581, in wait
    self._wait_futures(scope)
  File "/home/drisspg/meta/pytorch/torch/_inductor/async_compile.py", line 601, in _wait_futures
    kernel = result.result()
             ^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_inductor/codecache.py", line 4201, in result
    return self.result_fn()
           ^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_inductor/async_compile.py", line 453, in get_result
    kernel, elapsed_us = task.result()
                         ^^^^^^^^^^^^^
  File "/home/drisspg/.conda/envs/dev/lib/python3.12/concurrent/futures/_base.py", line 449, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/.conda/envs/dev/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception
torch._inductor.compile_worker.subproc_pool.SubprocException: An exception occurred in a subprocess:

Traceback (most recent call last):
  File "/home/drisspg/meta/pytorch/torch/_inductor/compile_worker/subproc_pool.py", line 377, in do_job
    result = job()
             ^^^^^
  File "/home/drisspg/meta/pytorch/torch/_inductor/runtime/compile_tasks.py", line 66, in _worker_compile_triton
    kernel.precompile(warm_cache_only=True)
  File "/home/drisspg/meta/pytorch/torch/_inductor/runtime/triton_heuristics.py", line 427, in precompile
    self._precompile_worker()
  File "/home/drisspg/meta/pytorch/torch/_inductor/runtime/triton_heuristics.py", line 458, in _precompile_worker
    compile_results.append(self._precompile_config(c))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/meta/pytorch/torch/_inductor/runtime/triton_heuristics.py", line 747, in _precompile_config
    binary = triton.compile(*compile_args, **compile_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/.conda/envs/dev/lib/python3.12/site-packages/triton/compiler/compiler.py", line 359, in compile
    next_module = compile_ir(module, metadata)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/.conda/envs/dev/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py", line 456, in <lambda>
    stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, capability)
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/drisspg/.conda/envs/dev/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py", line 298, in make_ttgir
    pm.run(mod)
RuntimeError: PassManager::run failed


To execute this test, run the following from the base repo dir:
    python test/inductor/test_flex_attention.py TestFlexAttentionCUDA.test_kv_batch_broadcast_batch_dims2_head_dims0_score_mod1_cuda_float16

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0
----------------------------------------------------------------- Captured stdout call -----------------------------------------------------------------
frames [('total', 3), ('ok', 3)]
inline_call []
stats [('calls_captured', 8), ('unique_graphs', 3)]
inductor [('async_compile_cache_miss', 9), ('triton_bundler_save_kernel', 8), ('async_compile_cache_hit', 3), ('fxgraph_cache_miss', 1), ('triton_bundler_save_static_autotuner', 1), ('fxgraph_cache_bypass', 1)]
aot_autograd [('total', 1), ('autograd_cache_miss', 1), ('ok', 1)]
graph_break []
----------------------------------------------------------------- Captured stderr call -----------------------------------------------------------------
/home/drisspg/meta/pytorch/torch/nn/attention/flex_attention.py:1582: UserWarning: flex_attention called without torch.compile() - this will use an unfused implementation that materializes the full scores matrix instead of generating a fused kernel.

SOLUTION: Use torch.compile(flex_attention)(...)

If you want to debug your score_mod/mask_mod, you can set:
torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

This will allow you to use print statements or breakpoints. Note: This doesn't work with the backwards pass and may produce incorrect results.
  _warn_once(
=============================================================== short test summary info ================================================================
FAILED [5.4911s] test/inductor/test_flex_attention.py::TestFlexAttentionCUDA::test_kv_batch_broadcast_batch_dims2_head_dims0_score_mod1_cuda_float16 - torch._inductor.compile_worker.subproc_pool.SubprocException: An exception occurred in a subprocess:
========================================================== 1 failed, 1405 deselected in 8.54s ==========================================================
~/meta/pytorch @c8205cb3 ❯