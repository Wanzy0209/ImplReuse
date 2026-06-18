import torch
import torch.nn

def slide_to_the_left2(state, new_events, arange, dev_null):
    batch_size, _, _ = new_events.shape

    concatenated = torch.cat([state, new_events], dim=1)

    # these three lines are a very complicated identity transformation:
    batch_idx = torch.arange(batch_size, dtype=torch.int32, device=state.device)[:, None]
    arange = arange[None, :]
    concatenated = concatenated[batch_idx, arange]

    state[:, :, :] = concatenated[:, -2048:, :]
    dev_null[:, :, :] = concatenated[:, :, :]

# Compile the function to trigger the potential miscompilation
slide_to_the_left3 = torch.compile(slide_to_the_left2)

def test_torch_all_verification_of_compile_state():
    """
    Test case to verify torch.all correctly detects miscompilation 
    in torch.compile when updating state tensors.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Test skipped: CUDA is required for this specific miscompilation scenario.")
        return

    # Initialize inputs
    state = torch.zeros([4, 2048, 1024], device=device)
    new_events = torch.arange(start=1, end=3, device=device)[None, :, None].expand(4, 2, 1024).contiguous()
    arange = torch.arange(2050, dtype=torch.int32, device=device)
    dev_null = torch.zeros([4, 2050, 1024], device=device)

    # The bug is probabilistic/racy, so we loop to try and trigger it
    for attempt in range(1, 100):
        state.zero_()
        
        slide_to_the_left3(state, new_events, arange, dev_null)

        # Use torch.all to verify the state of the tensor.
        # We expect all elements in state[:, :-2, :] to be zero.
        # If torch.compile miscompiles, non-zero values will appear here,
        # and torch.all should return False, causing the assertion to fail.
        is_state_clean = (state[:, :-2, :] == 0).all()
        
        assert is_state_clean, (
            f"torch.all detected non-zero values in state tensor on attempt {attempt}. "
            "This indicates a miscompilation bug in torch.compile."
        )

if __name__ == "__main__":
    test_torch_all_verification_of_compile_state()