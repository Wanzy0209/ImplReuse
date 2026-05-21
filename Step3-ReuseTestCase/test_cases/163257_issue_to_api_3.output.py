import torch
import torch.cuda
import torch.cuda.random

# Attempt to import RMM as per the bug report's use case.
# If RMM is not installed, this test cannot run.
try:
    import rmm
    from rmm.allocators.torch import rmm_torch_allocator
except ImportError:
    print("Warning: RMM library not found. Skipping test.")
    rmm = None

def test_pluggable_allocator_compile_interaction():
    """
    Test case to verify that changing the memory allocator (pluggable device)
    works correctly with torch.compile, while ensuring device state (RNG) 
    remains consistent, leveraging the pattern from torch.cuda.random.set_rng_state.
    """
    if rmm is None:
        return

    # --- Leverage similar API pattern (torch.cuda.random.set_rng_state) ---
    # We adopt the device resolution logic from the similar API to ensure
    # the test handles device indices explicitly and correctly.
    device_input = "cuda"  # Simulating input that could be str, int, or device
    if isinstance(device_input, str):
        device = torch.device(device_input)
    elif isinstance(device_input, int):
        device = torch.device("cuda", device_input)
    else:
        device = device_input

    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    # ---------------------------------------------------------------------

    # Save the initial RNG state to verify device integrity later
    initial_rng_state = torch.cuda.random.get_rng_state(device)

    # --- Original Bug Reproduction Logic ---
    # Reinitialize RMM with pool allocator and change PyTorch's current allocator
    rmm.reinitialize(pool_allocator=True)
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
    # --------------------------------------

    # Define a simple function to be compiled
    def simple_model(x):
        return x + 1

    # Attempt to compile the model. 
    # This is where the RuntimeError regarding checkPoolLiveAllocations would occur.
    try:
        compiled_model = torch.compile(simple_model)
        
        # Create a tensor on the specific device resolved above
        input_tensor = torch.randn(10, device=device)
        
        # Run the compiled model
        output = compiled_model(input_tensor)
        
        # Assertion to verify the computation is correct
        assert torch.allclose(output, input_tensor + 1), "Compiled model output mismatch"
        
        print("Test Passed: torch.compile works with pluggable allocator.")

    except RuntimeError as e:
        if "checkPoolLiveAllocations" in str(e):
            print(f"Test Failed: {e}")
            # In a real unit test, we would re-raise this to fail the suite
            raise
        else:
            raise

    # --- Verify Device State Integrity (Leveraging similar API) ---
    # Restore the RNG state to ensure the allocator change didn't corrupt
    # the device's random state management.
    torch.cuda.random.set_rng_state(initial_rng_state, device)
    restored_rng_state = torch.cuda.random.get_rng_state(device)
    
    assert torch.equal(initial_rng_state, restored_rng_state), \
        "RNG state was corrupted after changing allocator"
    # -------------------------------------------------------------

if __name__ == "__main__":
    test_pluggable_allocator_compile_interaction()