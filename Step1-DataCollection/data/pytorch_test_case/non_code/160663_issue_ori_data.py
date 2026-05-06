=== Executing Compiled Program  fullgraph=False
E0814 11:38:00.371520 659802 torch/_subclasses/fake_tensor.py:2755] [0/0] failed while attempting to run meta for aten.view.dtype
E0814 11:38:00.371520 659802 torch/_subclasses/fake_tensor.py:2755] [0/0] Traceback (most recent call last):
E0814 11:38:00.371520 659802 torch/_subclasses/fake_tensor.py:2755] [0/0]   File "/home/lsakka/pytorch/torch/_subclasses/fake_tensor.py", line 2751, in _dispatch_impl
E0814 11:38:00.371520 659802 torch/_subclasses/fake_tensor.py:2755] [0/0]     r = func(*args, **kwargs)
E0814 11:38:00.371520 659802 torch/_subclasses/fake_tensor.py:2755] [0/0]   File "/home/lsakka/pytorch/torch/_ops.py", line 840, in __call__
E0814 11:38:00.371520 659802 torch/_subclasses/fake_tensor.py:2755] [0/0]     return self._op(*args, **kwargs)
E0814 11:38:00.371520 659802 torch/_subclasses/fake_tensor.py:2755] [0/0] RuntimeError: self.dim() cannot be 0 to view ComplexFloat as Float (different element sizes)
❌ Compiled execution failed: backend='inductor' raised:
RuntimeError: self.dim() cannot be 0 to view ComplexFloat as Float (different element sizes)

While executing %tmp_37 : [num_users=1] = call_function[target=torch.ops.aten.add](args = (%tmp_38, %tmp_41), kwargs = {})
Original traceback:
  File "/home/lsakka/pytorch/example.py", line 404, in fuzzed_program
    tmp_37 = torch.ops.aten.add(tmp_38, tmp_41)

Use tlparse to see full graph. (https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs)

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"