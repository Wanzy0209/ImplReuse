Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1221, in wrapper
    self._join_threads(self.threads, fn)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1362, in _join_threads
    cls._check_return_codes(failed_ranks, timeout, fn)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1398, in _check_return_codes
    raise RuntimeError(error_msg)
RuntimeError: Thread 0 exited with exception:
Traceback (most recent call last):
  File "/var/lib/jenkins/workspace/test/distributed/_composable/fsdp/test_fully_shard_extensions.py", line 185, in _patch_two_tensor_fsdp_all_gather
    yield
  File "/var/lib/jenkins/workspace/test/distributed/_composable/fsdp/test_fully_shard_extensions.py", line 271, in test_all_gather_extensions_end_to_end
    self.run_subtests(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_fsdp.py", line 1137, in run_subtests
    return run_subtests(self, *args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1099, in run_subtests
    test_fn(*test_args, **test_kwargs, **subtest_kwargs)
  File "/var/lib/jenkins/workspace/test/distributed/_composable/fsdp/test_fully_shard_extensions.py", line 297, in _test_all_gather_extensions_end_to_end
    nn.init.trunc_normal_(param)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/nn/init.py", line 295, in trunc_normal_
    return _no_grad_trunc_normal_(tensor, mean, std, a, b, generator=generator)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/nn/init.py", line 114, in _no_grad_trunc_normal_
    tensor.uniform_(2 * l - 1, 2 * u - 1, generator=generator)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_compile.py", line 53, in inner
    return disable_fn(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 1005, in _fn
    return fn(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/tensor/_api.py", line 358, in __torch_dispatch__
    return DTensor._op_dispatcher.dispatch(
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/tensor/_dispatch.py", line 219, in dispatch
    with rng_context:
  File "/opt/conda/envs/py_3.10/lib/python3.10/contextlib.py", line 135, in __enter__
    return next(self.gen)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/tensor/_random.py", line 241, in _distribute_region
    assert g_name not in self.rng_states
AssertionError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1324, in run_test_with_threaded_pg
    getattr(self, test_name)()
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 1223, in wrapper
    fn()
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_utils.py", line 3224, in wrapper
    method(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/common_distributed.py", line 215, in wrapper
    return func(*args, **kwargs)
  File "/var/lib/jenkins/workspace/test/distributed/_composable/fsdp/test_fully_shard_extensions.py", line 270, in test_all_gather_extensions_end_to_end
    with self._patch_two_tensor_fsdp_all_gather(pre_all_gather_version=1):
  File "/opt/conda/envs/py_3.10/lib/python3.10/contextlib.py", line 153, in __exit__
    self.gen.throw(typ, value, traceback)
  File "/var/lib/jenkins/workspace/test/distributed/_composable/fsdp/test_fully_shard_extensions.py", line 187, in _patch_two_tensor_fsdp_all_gather
    dist.barrier()
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
    return func(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py", line 4818, in barrier
    work = group.barrier(opts=opts)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/distributed/multi_threaded_pg.py", line 377, in barrier
    return self.allreduce(tensor_list=[torch.ones(1)])
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/distributed/multi_threaded_pg.py", line 366, in allreduce
    res = coll.join(self._rank, tensor_list)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/distributed/multi_threaded_pg.py", line 295, in join
    self._collective.work(self._data)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/distributed/multi_threaded_pg.py", line 146, in work
    tensors = [
  File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/testing/_internal/distributed/multi_threaded_pg.py", line 147, in <listcomp>
    data[src_rank][i].to(rank_0_device) for src_rank in range(0, len(data))
AttributeError: 'list' object has no attribute 'to'

To execute this test, run the following from the base repo dir:
    python test/distributed/_composable/fsdp/test_fully_shard_extensions.py TestFullyShardAllGatherExtensionsMultiThread.test_all_gather_extensions_end_to_end

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0