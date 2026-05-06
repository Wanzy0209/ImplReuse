# Sender rank
        local_tensor = torch.randn(batch_size, total_columns)
        dst_t1 = local_tensor[:, split_offsets[0]:split_offsets[1]]
        dst_t2 = local_tensor[:, split_offsets[2]:split_offsets[3]]
        process_group = dist.group.WORLD
        send_op1 = dist.P2POp(
            dist.isend, dst_t1, 1, process_group, tag1
        )
        send_op2 = dist.P2POp(
            dist.isend, dst_t2, 1, process_group, tag2
        )
        # Batch send operations
        dist.batch_isend_irecv([send_op1, send_op2])
        print(f"Rank {rank} sent tensors with shapes {dst_t1.shape} and {dst_t2.shape}")
    elif rank == 1:
        # Receiver rank
        local_tensor_dst = torch.zeros(batch_size, total_columns)
        tensor_col_offset1 = split_offsets[0]
        end_col_offset1 = split_offsets[1]
        tensor_col_offset2 = split_offsets[2]
        end_col_offset2 = split_offsets[3]
        receiving_tensor_view1 = local_tensor_dst[:, tensor_col_offset1:end_col_offset1]
        receiving_tensor_view2 = local_tensor_dst[:, tensor_col_offset2:end_col_offset2]
        process_group = dist.group.WORLD
        receive_op1 = dist.P2POp(
            dist.irecv, receiving_tensor_view1, 0, process_group, tag1
        )
        receive_op2 = dist.P2POp(
            dist.irecv, receiving_tensor_view2, 0, process_group, tag2
        )
        # Batch receive operations
        dist.batch_isend_irecv([receive_op1, receive_op2])```

### Versions

version: main branch

cc @ezyang @gchanan @zou3519 @kadeng @msaroufim @H-Huang @awgu @wanchaol @fegin @fduwjj @wz337 @wconstab @d4l3k @pragupta @dcci