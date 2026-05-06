Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/inductor/test_mkldnn_pattern_matcher.py", line 3041, in test_qlinear_add_int8_mixed_bf16
    self._qlinear_add_test_helper(
  File "/var/lib/jenkins/workspace/test/inductor/test_mkldnn_pattern_matcher.py", line 2972, in _qlinear_add_test_helper
    self._test_common(
  File "/var/lib/jenkins/workspace/test/inductor/test_mkldnn_pattern_matcher.py", line 220, in _test_common
    _ = torch.compile(convert_model)(*inputs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 418, in __call__
    return super().__call__(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1777, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1788, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 899, in compile_wrapper
    raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 2310, in _call_user_compiler
    raise BackendCompilerFailed(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_dynamo/output_graph.py", line 2285, in _call_user_compiler
    compiled_fn = compiler_fn(gm, example_inputs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_dynamo/repro/after_dynamo.py", line 156, in __call__
    compiled_gm = compiler_fn(gm, example_inputs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/__init__.py", line 2399, in __call__
    return compile_fx(model_, inputs_, config_patches=self.config)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 2682, in compile_fx
    return aot_autograd(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_dynamo/backends/common.py", line 117, in __call__
    cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 1108, in aot_module_simplified
    compiled_fn, _ = aot_stage2_compile(aot_state, aot_graph_capture)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/graph_compile.py", line 244, in aot_stage2_compile
    return aot_stage2_inference(aot_state, aot_graph_capture)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_functorch/_aot_autograd/graph_compile.py", line 317, in aot_stage2_inference
    compiled_fw = compiler(fw_module, updated_flat_args)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 2033, in fw_compiler_freezing
    optimized_function = inner_compile(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 785, in compile_fx_inner
    return wrap_compiler_debug(_compile_fx_inner, compiler_name="inductor")(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_dynamo/repro/after_aot.py", line 144, in debug_wrapper
    inner_compiled_fn = compiler_fn(gm, example_inputs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 966, in _compile_fx_inner
    mb_compiled_graph = fx_codegen_and_compile(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 1696, in fx_codegen_and_compile
    return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/compile_fx.py", line 1506, in codegen_and_compile
    compiled_module = graph.compile_to_module()
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/graph.py", line 2359, in compile_to_module
    return self._compile_to_module()
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/graph.py", line 2369, in _compile_to_module
    mod = self._compile_to_module_lines(wrapper_code)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/graph.py", line 2440, in _compile_to_module_lines
    mod = PyCodeCache.load_by_key_path(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/codecache.py", line 3578, in load_by_key_path
    mod = _reload_python_module(key, path, set_sys_modules=in_toplevel)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/runtime/compile_tasks.py", line 33, in _reload_python_module
    exec(code, mod.__dict__, mod.__dict__)
  File "/tmp/tmp0exlbkwh/r3/cr3cgjuvxobcmzkrmayfljnjujrkxw67vn5abhq2rccptpionixp.py", line 453, in <module>
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/async_compile.py", line 636, in wait
    self._wait_futures(scope)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/async_compile.py", line 656, in _wait_futures
    kernel = result.result()
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/codecache.py", line 4319, in result
    return self.result_fn()
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/codecache.py", line 3055, in future
    result = get_result()
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/codecache.py", line 2851, in load_fn
    future.result()
  File "/opt/conda/envs/py_3.10/lib/python3.10/concurrent/futures/_base.py", line 451, in result
    return self.__get_result()
  File "/opt/conda/envs/py_3.10/lib/python3.10/concurrent/futures/_base.py", line 403, in __get_result
    raise self._exception
  File "/opt/conda/envs/py_3.10/lib/python3.10/concurrent/futures/thread.py", line 58, in run
    result = self.fn(*self.args, **self.kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/codecache.py", line 2881, in _worker_compile_cpp
    builder.build()
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/cpp_builder.py", line 2096, in build
    run_compile_cmd(build_cmd, cwd=_build_tmp_dir)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/cpp_builder.py", line 624, in run_compile_cmd
    _run_compile_cmd(cmd_line, cwd)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_inductor/cpp_builder.py", line 619, in _run_compile_cmd
    raise exc.CppCompileError(cmd, output) from e
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
CppCompileError: C++ compile error

Command:
g++ /tmp/tmp0exlbkwh/j4/cj4jafalktnzrg2px7mvgmbljkgs7u2qepjvj5c2iw7so7trzxrf.main.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX512 -O3 -DNDEBUG -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-math-errno -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=native -shared -fPIC -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp -include /tmp/torchinductor_jenkins/precompiled_headers/cg3agm3x5pguztwxgwmk642bvxyc46rzav2sile2htyzuq5t2wow.h -I/opt/conda/envs/py_3.10/include/python3.10 -I/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/include -I/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -o /tmp/tmp0exlbkwh/j4/cj4jafalktnzrg2px7mvgmbljkgs7u2qepjvj5c2iw7so7trzxrf.main.so -ltorch -ltorch_cpu -ltorch_python -lgomp -lc10 -L/opt/conda/envs/py_3.10/lib -L/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/lib

Output:

g++: internal compiler error: Segmentation fault signal terminated program cc1plus
Please submit a full bug report,
with preprocessed source if appropriate.
See <file:///usr/share/doc/gcc-11/README.Bugs> for instructions.


Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"


To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_ASAN=1 PYTORCH_TEST_WITH_UBSAN=1 PYTORCH_TEST_WITH_SLOW=1 PYTORCH_TEST_SKIP_FAST=1 python test/inductor/test_mkldnn_pattern_matcher.py TestPatternMatcher.test_qlinear_add_int8_mixed_bf16_use_relu_True_is_qat_True_is_dynamic_False

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0