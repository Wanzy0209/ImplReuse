CUDA_VISIBLE_DEVICES=0 \
 NCCL_DEBUG=INFO \
 NCCL_DEBUG_SUBSYS=INIT,GRAPH \
 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
 NCCL_SOCKET_IFNAME=eno3 \ # ib0 when using infiniband
 GLOO_SOCKET_IFNAME=eno3 \ # ib0 when using infiniband
 NCCL_IB_DISABLE=1 \ # 0 when using infiniband
 TORCH_CPP_LOG_LEVEL=INFO \
 TORCH_DISTRIBUTED_DEBUG=DETAIL \
 python -m torch.distributed.run \
 --nnodes 2 \
 --node_rank 0 \
 --nproc_per_node 1 \
 --rdzv-endpoint=127.0.0.1:29510 \
 --rdzv-id ttmultinode \
 -m torchtitan.train \
 --job.config_file torchtitan/models/llama3/train_configs/debug_model.toml