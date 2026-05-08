import gc
import time
import traceback
import argparse
import torch
from functools import partial
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


flex_attention_comp = torch.compile(flex_attention)
#flex_attention_comp = flex_attention


def get_block_mask(q_len, kv_len, device, sink_tokens, window_size):
    offset = kv_len - q_len
    def sink_local_mask(b, h, q_idx, kv_idx, num_sink_tokens, window_size, offset):
        q_abs = q_idx + offset
        causal_condition = q_abs >= kv_idx
        sink_condition = kv_idx < num_sink_tokens
        local_condition = q_abs - kv_idx < window_size
        return causal_condition & (sink_condition | local_condition)
    mask_fn = partial(sink_local_mask, num_sink_tokens=sink_tokens, 
                        window_size=window_size, offset=offset)
    block_mask = create_block_mask(
        mask_mod=mask_fn,
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
    )
    return block_mask


def test_flex_attention_compile(
    batch_size=32,
    sequence_length=2048,
    num_heads=32,
    head_dim=128,
    window_size=64,
    num_sink_tokens=4,
    device="cuda",
    dtype=torch.float16,
    use_compile=True,
    verbose=True,
):
    """
    Test flex_attention with torch.compile on long sequences and large batch sizes.

    Args:
        batch_size: Batch size to test
        sequence_length: Sequence length to test
        num_heads: Number of attention heads
        head_dim: Head dimension
        window_size: Window size for local attention
        num_sink_tokens: Number of sink tokens
        device: Device to run on
        dtype: Data type
        use_compile: Whether to use torch.compile
        verbose: Whether to print verbose output
    """
    print(f"Testing flex_attention with:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Window size: {window_size}")
    print(f"  Num sink tokens: {num_sink_tokens}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    print(f"  Use compile: {use_compile}")
    print()
        
    q = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    if verbose:
        print(f"Created tensors with shapes:")
        print(f"  q: {q.shape}")
        print(f"  k: {k.shape}")
        print(f"  v: {v.shape}")
        print()

    try:
        if verbose:
            print("Creating block mask...")

        block_mask = get_block_mask(
            q_len=sequence_length,
            kv_len=sequence_length,
            device=device,
            sink_tokens=num_sink_tokens,
            window_size=window_size,
        )
        if verbose:
            print(f"Block mask created successfully with shape: {block_mask.shape}")
            print()

    except Exception as e:
        print(f"Error creating block mask: {e}")
        traceback.print_exc()
        return False

    try:
        print("Testing flex_attention...")
        if use_compile:
            attention_fn = flex_attention_comp
        else:
            attention_fn = flex_attention

        print("Warming up...")
        with torch.no_grad():
            _ = attention_fn(q, k, v, block_mask=block_mask)

        torch.cuda.synchronize()
        print("Running forward pass...")
        start_time = time.time()
        output = attention_fn(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()
        forward_time = time.time() - start_time
        print(f"Total time: {forward_time:.4f}s")
        return True

    except Exception as e:
        print(f"Error during flex_attention test: {e}")
        traceback.print_exc()
        return False


def test_memory_usage(
    batch_size=32, sequence_length=2048, num_heads=32, head_dim=128, device="cuda", window_size=64, num_sink_tokens=4
):
    """Test memory usage with different configurations."""
    print("Testing memory usage...")

    torch.cuda.empty_cache()
    gc.collect()
    initial_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB

    try:
        q = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=device)
        tensor_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
        block_mask = get_block_mask(
            q_len=sequence_length,
            kv_len=sequence_length,
            device=device,
            sink_tokens=num_sink_tokens,
            window_size=window_size,
        )
        mask_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
        output = flex_attention_comp(q, k, v, block_mask=block_mask)
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3  # GB

        print(f"Memory usage:")
        print(f"  Initial: {initial_memory:.2f} GB")
        print(
            f"  After tensors: {tensor_memory:.2f} GB (+{tensor_memory - initial_memory:.2f} GB)"
        )
        print(
            f"  After mask: {mask_memory:.2f} GB (+{mask_memory - tensor_memory:.2f} GB)"
        )
        print(f"  Peak: {peak_memory:.2f} GB (+{peak_memory - initial_memory:.2f} GB)")

    except Exception as e:
        print(f"Error during memory test: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Test flex_attention with torch.compile"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size to test")
    parser.add_argument(
        "--seq-len", type=int, default=32768, help="Sequence length to test"
    )
    parser.add_argument(
        "--num-heads", type=int, default=32, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument(
        "--window-size", type=int, default=256, help="Window size for local attention"
    )
    parser.add_argument(
        "--num-sink-tokens", type=int, default=4, help="Number of sink tokens"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument(
        "--no-compile", action="store_true", help="Disable torch.compile"
    )
    parser.add_argument(
        "--memory-test", action="store_true", help="Run memory usage test"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    print("=" * 80)
    print("FLEX ATTENTION TORCH.COMPILE TEST")
    print("=" * 80)

    # Test basic functionality
    success = test_flex_attention_compile(
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        window_size=args.window_size,
        num_sink_tokens=args.num_sink_tokens,
        device=args.device,
        dtype=dtype,
        use_compile=not args.no_compile,
        verbose=args.verbose,
    )

    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")

    # Memory test if requested
    if args.memory_test:
        print("\n" + "=" * 80)
        print("MEMORY USAGE TEST")
        print("=" * 80)
        test_memory_usage(
            batch_size=args.batch_size,
            sequence_length=args.seq_len,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            device=args.device,
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()