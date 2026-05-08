weights = torch.randint(0, 2, (K, N), device="mps").float() * 2 - 1

weights_fp16 = weights.half()

print(f"weights_fp16 strides : {weights_fp16.stride(0), weights_fp16.stride(1)}, shape={weights_fp16.shape}")
assert(weights_fp16.is_contiguous())