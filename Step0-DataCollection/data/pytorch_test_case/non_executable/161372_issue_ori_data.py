/Users/sebastian/.venv/lib/python3.10/site-packages/torch/_refs/__init__.py:5767: UserWarning: record_context_cpp is not support on non-linux non-x86_64 platforms (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/profiler/unwind/unwind.cpp:12.)
  r = torch.where(mask, value, a)  # type: ignore[arg-type]
W0823 18:56:31.554000 40177 torch/_dynamo/convert_frame.py:1016] [0/8] torch._dynamo hit config.recompile_limit (8)
W0823 18:56:31.554000 40177 torch/_dynamo/convert_frame.py:1016] [0/8]    function: 'forward' (/Users/sebastian/Desktop/qwen3.py:57)
W0823 18:56:31.554000 40177 torch/_dynamo/convert_frame.py:1016] [0/8]    last reason: 0/7: tensor 'cache.cache[0][0]' size mismatch at index 2. expected 77, actual 78
W0823 18:56:31.554000 40177 torch/_dynamo/convert_frame.py:1016] [0/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
W0823 18:56:31.554000 40177 torch/_dynamo/convert_frame.py:1016] [0/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.