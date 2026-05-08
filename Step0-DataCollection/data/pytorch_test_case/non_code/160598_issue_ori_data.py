Traceback (most recent call last):
  File "/var/lib/jenkins/pytorch/test/test_cuda.py", line 6295, in test_autocast_custom_enabled
    loss.backward()
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/_tensor.py", line 625, in backward
    torch.autograd.backward(
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/autograd/__init__.py", line 354, in backward
    _engine_run_backward(
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/autograd/graph.py", line 841, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/autograd/function.py", line 315, in apply
    return user_fn(self, *args)
           ^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/amp/autocast_mode.py", line 573, in decorate_bwd
    return bwd(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^
  File "/var/lib/jenkins/pytorch/test/test_cuda.py", line 6280, in backward
    a_grad, b_grad = grad.mm(b.t()), a.t().mm(grad)
                     ^^^^^^^^^^^^^^
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 76.00 MiB. GPU 0 has a total capacity of 255.69 GiB of which 252.82 GiB is free. 1.20 GiB allowed; Of the allocated memory 1.14 GiB is allocated by PyTorch, with 17.00 MiB allocated in private pools (e.g., HIP Graphs), and 18.63 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

To execute this test, run the following from the base repo dir:
    PYTORCH_TEST_WITH_ROCM=1 python test/test_cuda.py TestCudaAutocast.test_autocast_custom_enabled

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0