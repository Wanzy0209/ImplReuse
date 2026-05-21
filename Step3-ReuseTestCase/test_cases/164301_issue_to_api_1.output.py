import torch
import time
from collections import namedtuple

# Reusing the structure of the Similar API: tf.compat.v1.train.SessionRunValues
# This helps structure the results of the compiled run (results, options, metadata)
SessionRunValues = namedtuple("SessionRunValues", ["results", "options", "run_metadata"])

def test_torch_compile_mxfp8_dim0_regression():
    """
    Test case for torch.compile regression with mxfp8 quantization along rows.
    This test mimics the logic of the bug report (dim0 scaling) and uses the
    SessionRunValues pattern to capture execution details.
    """
    
    # Check for CUDA availability as the bug is specific to B200 GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available. Running on CPU. Performance metrics will not reflect the bug.")

    # Reproduction parameters from the issue
    M, K, BLOCK_SIZE = 16384, 16384, 32
    
    # Define the quantization logic mimicking 'dim0_mxfp8_floor'
    # The bug involves scaling granularity along rows (dim 0)
    def dim0_mxfp8_floor_fn(x):
        # Simulate row-wise scaling (1 x block_size granularity logic approximation)
        # We calculate a scale per column (dim 0 reduction) to mimic the "along rows" description
        # Note: Actual mxfp8 requires specific hardware support or torchao kernels,
        # here we approximate the compute pattern to trigger the compiler path.
        scale = x.abs().max(dim=0, keepdim=True).values
        # Avoid division by zero
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        
        # Cast to float8 (e4m3fnuz is common for MX formats)
        # Using standard float8_e4m3fn if fnuz is not available on all archs, 
        # but aiming for the MX format target.
        try:
            dtype = torch.float8_e4m3fnuz
        except AttributeError:
            dtype = torch.float8_e4m3fn
            
        quantized = (x / scale).to(dtype)
        return quantized, scale

    # Compile the function
    # The regression is in inductor codegen, so we use torch.compile
    compiled_fn = torch.compile(dim0_mxfp8_floor_fn, mode="max-autotune")

    # Create input tensor
    # Using float16 as input is common for quantization workflows
    input_tensor = torch.randn(M, K, dtype=torch.float16, device=device)

    # Warmup runs to allow compilation and initialization
    for _ in range(5):
        compiled_fn(input_tensor)

    # Synchronize to ensure accurate timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark run
    start_time = time.time()
    results, scales = compiled_fn(input_tensor)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.time()

    # Calculate metrics
    time_us = (end_time - start_time) * 1e6
    
    # Estimate memory bandwidth (GB/s)
    # Read: M*K*2 bytes (fp16), Write: M*K*1 byte (fp8) + M*K*2 bytes (scale approx)
    # This is a rough approximation for the test assertion context
    total_bytes = (M * K * 2) + (M * K * 1) + (K * 2) 
    mem_bw_gbps = (total_bytes / 1e9) / ((end_time - start_time))

    # Package results into the SessionRunValues structure
    # results: The output tensors
    # options: Configuration used for the run
    # run_metadata: Performance metrics and environment info
    run_values = SessionRunValues(
        results=(results, scales),
        options={
            "mode": "dim0_mxfp8_floor",
            "block_size": BLOCK_SIZE,
            "dtype": str(input_tensor.dtype)
        },
        run_metadata={
            "time_us": time_us,
            "mem_bw_gbps": mem_bw_gbps,
            "device": str(device),
            "torch_version": torch.__version__
        }
    )

    # Assertions
    # 1. Check output shape
    assert run_values.results[0].shape == (M, K), "Output shape mismatch"
    
    # 2. Check output dtype
    assert run_values.results[0].dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz), \
        f"Output dtype mismatch: {run_values.results[0].dtype}"

    # 3. Performance Regression Check
    # The bug report shows a drop from ~5600 gbps to ~1485 gbps.
    # We set a threshold to catch the regression state (e.g., should be > 2000 gbps on B200).
    # Note: This threshold is hardware dependent. On CPU/GPUs without Tensor Cores, 
    # this will be lower. We only assert if we are on a compatible CUDA device.
    if device.type == "cuda" and torch.cuda.get_device_properties(device).major >= 9:
        # Assuming B200 (Hopper+ architecture) or similar high performance
        # If the bug is present, bandwidth will be ~1500. If fixed, ~5600.
        # We use a loose threshold to avoid flakiness but catch the severe regression.
        assert run_values.run_metadata["mem_bw_gbps"] > 2000.0, \
            f"Performance regression detected: {run_values.run_metadata['mem_bw_gbps']:.2f} gbps is too low."

    print(f"Test Passed.")
    print(f"Mode: {run_values.options['mode']}")
    print(f"Time (us): {run_values.run_metadata['time_us']:.2f}")
    print(f"Memory Bandwidth (gbps): {run_values.run_metadata['mem_bw_gbps']:.2f}")

if __name__ == "__main__":
    test_torch_compile_mxfp8_dim0_regression()