RuntimeError: self.stride(-1) must be 1 to view ComplexDouble as Double (different element sizes), but got 10

While executing %tmp_1 : [num_users=1] = call_function[target=torch.ops.aten.add](args = (%l_arg_0_, %tmp_3), kwargs = {})
Original traceback:
  File "/tmp/tmpmpsfa_4r_generated.py", line 471, in fuzzed_program
    tmp_1 = torch.ops.aten.add(tmp_2, tmp_3)

Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs)

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more de