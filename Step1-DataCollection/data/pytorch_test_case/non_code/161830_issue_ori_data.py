File "..../torch/_dynamo/symbolic_convert.py", line 890, in wrapper
    return handle_graph_break(self, inst, speculation.reason)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "..../torch/_dynamo/symbolic_convert.py", line 1004, in handle_graph_break
    self.create_call_resume_at(
  File "..../torch/_dynamo/symbolic_convert.py", line 3681, in create_call_resume_at
    new_code: types.CodeType = ContinueExecutionCache.lookup(
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "..../torch/_dynamo/resume_execution.py", line 296, in lookup
    cls.cache[code][key] = cls.generate(code, lineno, *key)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "..../torch/_dynamo/resume_execution.py", line 321, in generate
    return cls.generate_based_on_original_code_object(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "..../torch/_dynamo/resume_execution.py", line 608, in generate_based_on_original_code_object
    setup_fn_target_offsets = tuple(
                              ^^^^^^
  File "..../torch/_dynamo/resume_execution.py", line 609, in <genexpr>
    meta.block_target_offset_remap[n] for n in setup_fn_target_offsets
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
torch._dynamo.exc.InternalTorchDynamoError: KeyError: 896