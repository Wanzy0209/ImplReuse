✅ Compiled execution successful
Compiled result type: <class 'complex'>
✅ Results match between original and compiled 1 (fullgraph=False)

=== Executing Compiled Program  fullgraph=False dynamic=True
W0815 01:16:12.941263 1226452 torch/fx/experimental/sym_node.py:1383] [12/1] failed to eval mul(True, s46*(s43 + 1))
❌ Compiled execution 2 failed: Dynamo failed to run FX node with fake tensors: call_function <built-in function mul>(*(True, s46*(s43 + 1)), **{}): got TypeError('BooleanAtom not allowed in this context.')

from user code:
   File "/tmp/tmpexsuhp1v_generated.py", line 511, in torch_dynamo_resume_in_fuzzed_program_at_487
    tmp_124 = tmp_125 * tmp_126

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"