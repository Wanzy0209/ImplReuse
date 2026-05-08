File "e:\xyz\python\Lib\site-packages\torch\_dynamo\eval_frame.py", line 929, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "e:\xyz\python\Lib\site-packages\torch\_functorch\aot_autograd.py", line 1241, in forward
    return compiled_fn(full_args)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "e:\xyz\python\Lib\site-packages\torch\_functorch\_aot_autograd\runtime_wrappers.py", line 384, in runtime_wrapper
    all_outs = call_func_at_runtime_with_args(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "e:\xyz\python\Lib\site-packages\torch\_functorch\_aot_autograd\utils.py", line 126, in call_func_at_runtime_with_args
    out = normalize_as_list(f(args))
                            ^^^^^^^
  File "e:\xyz\python\Lib\site-packages\torch\_functorch\_aot_autograd\runtime_wrappers.py", line 556, in wrapper
    return compiled_fn(runtime_args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "e:\xyz\python\Lib\site-packages\torch\_inductor\output_code.py", line 584, in __call__
    return self.current_callable(inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "e:\xyz\python\Lib\site-packages\torch\_inductor\compile_fx.py", line 1655, in run
    return compiled_fn(new_inputs)  # type: ignore[arg-type]
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "e:\xyz\python\Lib\site-packages\torch\_inductor\cudagraph_trees.py", line 403, in deferred_cudagraphify
    fn, out = cudagraphify(model, inputs, new_static_input_idxs, *args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "e:\xyz\python\Lib\site-packages\torch\_inductor\cudagraph_trees.py", line 462, in cudagraphify
    return manager.add_function(
           ^^^^^^^^^^^^^^^^^^^^^
  File "e:\xyz\python\Lib\site-packages\torch\_inductor\cudagraph_trees.py", line 2316, in add_function
    return fn, fn(inputs)
               ^^^^^^^^^^
  File "e:\xyz\python\Lib\site-packages\torch\_inductor\cudagraph_trees.py", line 2012, in run
    out = self._run(new_inputs, function_id)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "e:\xyz\python\Lib\site-packages\torch\_inductor\cudagraph_trees.py", line 2116, in _run
    return self.run_eager(new_inputs, function_id)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "e:\xyz\python\Lib\site-packages\torch\_inductor\cudagraph_trees.py", line 2277, in run_eager
    return node.run(new_inputs)
           ^^^^^^^^^^^^^^^^^^^^
  File "e:\xyz\python\Lib\site-packages\torch\_inductor\cudagraph_trees.py", line 685, in run
    out = self.wrapped_function.model(new_inputs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\XYZ\AppData\Local\Temp\1\torchinductor_xyz\it\cite53vorrcgqh2agijfgurh2b757avwpj4y23bsg4brxd2357z5.py", line 105, in call
    triton_poi_fused_addmm_silu_0.run(buf1, arg1_1, 51200, stream=stream0)
  File "e:\xyz\python\Lib\site-packages\torch\_inductor\runtime\triton_heuristics.py", line 1180, in run
    return launcher(
           ^^^^^^^^^
  File "<string>", line 5, in launcher
  File "e:\xyz\python\Lib\site-packages\torch\_inductor\runtime\static_cuda_launcher.py", line 227, in run
    _StaticCudaLauncher._launch_kernel(
OverflowError: Python int too large to convert to C long