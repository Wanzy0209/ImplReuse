@with_comms
    def test_strided_shard_compute_local_shape_and_global_offset_2D(self):
        device_mesh = init_device_mesh(self.device_type, (2, 2))
        batch_size, seq_len, dim1 = 2, 3, 3
        nelem = batch_size * seq_len * dim1
        global_tensor = torch.arange(nelem).view(batch_size * seq_len * dim1)
        global_shape = global_tensor.size()

        placements = (
            _StridedShard(dim=0, split_factor=batch_size),
            _StridedShard(dim=0, split_factor=batch_size * math.ceil(seq_len * 1.0 / device_mesh.size(0)))
        )

        dtensor = distribute_tensor(global_tensor, device_mesh, (Replicate(), Replicate())).redistribute(device_mesh, placements)
        local_size, global_offset = compute_local_shape_and_global_offset(
            global_shape, device_mesh, placements
        )

        import time
        time.sleep(torch.distributed.get_rank())
        print(f"{torch.distributed.get_rank()=} {local_size=} expected_size={dtensor._local_tensor.shape}")