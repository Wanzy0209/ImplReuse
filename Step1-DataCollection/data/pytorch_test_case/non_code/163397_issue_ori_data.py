W0920 10:36:10.718000 1320983 site-packages/torch/multiprocessing/spawn.py:169] Terminating process 1321005 via signal SIGTERM
W0920 10:36:10.720000 1320983 site-packages/torch/multiprocessing/spawn.py:169] Terminating process 1321006 via signal SIGTERM
W0920 10:36:10.720000 1320983 site-packages/torch/multiprocessing/spawn.py:169] Terminating process 1321007 via signal SIGTERM
Traceback (most recent call last):
  File "-/run.py", line 102, in <module>
    main()
  File "-/run.py", line 95, in main
    mp.spawn(proc_main, nprocs=torch.cuda.device_count())
  File "-/python3.12/site-packages/torch/multiprocessing/spawn.py", line 340, in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "-/python3.12/site-packages/torch/multiprocessing/spawn.py", line 296, in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
  File "-/python3.12/site-packages/torch/multiprocessing/spawn.py", line 215, in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
torch.multiprocessing.spawn.ProcessRaisedException: 

-- Process 3 terminated with the following error:
Traceback (most recent call last):
  File "-/python3.12/site-packages/torch/multiprocessing/spawn.py", line 90, in _wrap
    fn(i, *args)
  File "-/run.py", line 88, in proc_main
    destroy_process_group(local_rank)
  File "-/run.py", line 76, in destroy_process_group
    distributed.destroy_process_group()
  File "-/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 2146, in destroy_process_group
    _shutdown_backend(pg_to_shutdown)
  File "-/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 1815, in _shutdown_backend
    backend._shutdown()
torch.distributed.DistBackendError: NCCL error in: /pytorch/torch/csrc/distributed/c10d/NCCLUtils.cpp:133, unhandled cuda error (run with NCCL_DEBUG=INFO for details), NCCL version 2.21.5
ncclUnhandledCudaError: Call to CUDA function failed.
Last error:
Cuda failure 'out of memory'