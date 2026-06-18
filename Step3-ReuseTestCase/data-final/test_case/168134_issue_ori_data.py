@with_comms
def test_strided_shard_bug(self):
    assert self.world_size == 2
    device_mesh = init_device_mesh(self.device_type, (2, ))
    global_tensor = torch.Tensor([0, 1, 2, 3, 4])
    dtensor = distribute_tensor(global_tensor, device_mesh, (Replicate(), )).redistribute(device_mesh, (_StridedShard(0, split_factor=2),))
    import time
    time.sleep(torch.distributed.get_rank())
    print(f"{torch.distributed.get_rank()=} {dtensor._local_tensor=}")