torch._inductor.exc.InductorError: CompilationError: at 28:4:
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.reshape(tma_descriptor0.load([xoffset, r0_offset // 128, (r0_offset % 128)]), [XBLOCK, R0_BLOCK])
        tmp1 = tl.reshape(tma_descriptor1.load([xoffset, r0_offset // 128, (r0_offset % 128)]), [XBLOCK, R0_BLOCK])
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.make_tensor_descriptor(out_ptr0, shape=[2], strides=[1], block_shape=[XBLOCK]).store([xoffset], tl.reshape(tl.broadcast_to(tmp4, [XBLOCK, 1]), [XBLOCK]).to(tl.float32))
    ^
Descriptor block shape must have at least 16 bytes in the last dimension, but got 1 * 4 = 4 bytes

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"