[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/ilia/repos/vllm/rendezvous_script.py", line 44, in <module>
[rank0]:     run_rendezvous()
[rank0]:   File "/home/ilia/repos/vllm/rendezvous_script.py", line 32, in run_rendezvous
[rank0]:     handle = torch_symm_mem.rendezvous(buffer, group_name)
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/ilia/vllm-venv/lib/python3.12/site-packages/torch/distributed/_symmetric_memory/__init__.py", line 1740, in rendezvous
[rank0]:     return _SymmetricMemory.rendezvous(tensor, group_name)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: RuntimeError: CUDA driver error: system not yet initialized
[rank1]: Traceback (most recent call last):
[rank1]:   File "/home/ilia/repos/vllm/rendezvous_script.py", line 44, in <module>
[rank1]:     run_rendezvous()
[rank1]:   File "/home/ilia/repos/vllm/rendezvous_script.py", line 32, in run_rendezvous
[rank1]:     handle = torch_symm_mem.rendezvous(buffer, group_name)
[rank1]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ilia/vllm-venv/lib/python3.12/site-packages/torch/distributed/_symmetric_memory/__init__.py", line 1740, in rendezvous
[rank1]:     return _SymmetricMemory.rendezvous(tensor, group_name)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]: RuntimeError: CUDA driver error: system not yet initialized
W1111 04:39:05.501000 533012 torch/distributed/elastic/multiprocessing/api.py:908] Sending process 533105 closing signal SIGTERM
E1111 04:39:05.715000 533012 torch/distributed/elastic/multiprocessing/api.py:882] failed (exitcode: 1) local_rank: 1 (pid: 533106) of binary: /home/ilia/vllm-venv/bin/python3
Traceback (most recent call last):
  File "/home/ilia/vllm-venv/bin/torchrun", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/ilia/vllm-venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 357, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/home/ilia/vllm-venv/lib/python3.12/site-packages/torch/distributed/run.py", line 936, in main
    run(args)
  File "/home/ilia/vllm-venv/lib/python3.12/site-packages/torch/distributed/run.py", line 927, in run
    elastic_launch(
  File "/home/ilia/vllm-venv/lib/python3.12/site-packages/torch/distributed/launcher/api.py", line 156, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ilia/vllm-venv/lib/python3.12/site-packages/torch/distributed/launcher/api.py", line 293, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: