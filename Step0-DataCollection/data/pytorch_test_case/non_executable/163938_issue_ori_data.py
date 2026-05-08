python3 -c "import torch.distributed.algorithms.ddp_comm_hooks"

/pytorch/torch/distributed/algorithms/ddp_comm_hooks/__init__.py:54: FutureWarning: functools.partial will be a method descriptor in future Python versions; wrap it in enum.member() if you want to preserve the old behavior
  ALLREDUCE = partial(_ddp_comm_hook_wrapper, comm_hook=default.allreduce_hook)
/pytorch/torch/distributed/algorithms/ddp_comm_hooks/__init__.py:55: FutureWarning: functools.partial will be a method descriptor in future Python versions; wrap it in enum.member() if you want to preserve the old behavior
  FP16_COMPRESS = partial(
/pytorch/torch/distributed/algorithms/ddp_comm_hooks/__init__.py:58: FutureWarning: functools.partial will be a method descriptor in future Python versions; wrap it in enum.member() if you want to preserve the old behavior
  BF16_COMPRESS = partial(
/pytorch/torch/distributed/algorithms/ddp_comm_hooks/__init__.py:61: FutureWarning: functools.partial will be a method descriptor in future Python versions; wrap it in enum.member() if you want to preserve the old behavior
  QUANTIZE_PER_TENSOR = partial(
/pytorch/torch/distributed/algorithms/ddp_comm_hooks/__init__.py:64: FutureWarning: functools.partial will be a method descriptor in future Python versions; wrap it in enum.member() if you want to preserve the old behavior
  QUANTIZE_PER_CHANNEL = partial(
/pytorch/torch/distributed/algorithms/ddp_comm_hooks/__init__.py:67: FutureWarning: functools.partial will be a method descriptor in future Python versions; wrap it in enum.member() if you want to preserve the old behavior
  POWER_SGD = partial(
/pytorch/torch/distributed/algorithms/ddp_comm_hooks/__init__.py:74: FutureWarning: functools.partial will be a method descriptor in future Python versions; wrap it in enum.member() if you want to preserve the old behavior
  POWER_SGD_RANK2 = partial(
/pytorch/torch/distributed/algorithms/ddp_comm_hooks/__init__.py:80: FutureWarning: functools.partial will be a method descriptor in future Python versions; wrap it in enum.member() if you want to preserve the old behavior
  BATCHED_POWER_SGD = partial(
/pytorch/torch/distributed/algorithms/ddp_comm_hooks/__init__.py:85: FutureWarning: functools.partial will be a method descriptor in future Python versions; wrap it in enum.member() if you want to preserve the old behavior
  BATCHED_POWER_SGD_RANK2 = partial(
/pytorch/torch/distributed/algorithms/ddp_comm_hooks/__init__.py:90: FutureWarning: functools.partial will be a method descriptor in future Python versions; wrap it in enum.member() if you want to preserve the old behavior
  NOOP = partial(