import torch
import torch.distributions

def test_compile_torch_eq_preserves_distribution_state():
    """
    Test that torch.compile, when processing functions involving torch.eq,
    does not inadvertently modify global distribution validation settings.
    
    This relates to Issue #167064 where redundant global code in context parallel
    (triggering torch.compile) was calling set_default_validate_args(False).
    """
    # 1. Establish a known global state for distributions
    # We set it to True to detect if it gets flipped to False (the bug behavior)
    torch.distributions.Distribution.set_default_validate_args(True)
    initial_state = torch.distributions.Distribution._default_validate_args
    assert initial_state is True, "Failed to set initial state"

    # 2. Define a function utilizing the similar API: torch.eq
    # The similarity score suggests a relationship between the issue's context
    # and the implementation/usage of torch.eq (specifically in inductor codegen).
    def simple_eq_fn(x, y):
        return torch.eq(x, y)

    # 3. Trigger torch.compile
    # The bug report indicates that the mere act of compilation (or creating
    # compiled code) triggers the side effect in torch._dynamo/eval_frame.
    compiled_fn = torch.compile(simple_eq_fn)

    # 4. Execute the compiled function to ensure the inductor backend runs
    input_x = torch.randn(5)
    input_y = torch.randn(5)
    _ = compiled_fn(input_x, input_y)

    # 5. Verify that the global state has NOT been altered
    final_state = torch.distributions.Distribution._default_validate_args
    
    assert final_state is initial_state, (
        f"Global distribution validation args changed unexpectedly. "
        f"Expected {initial_state}, got {final_state}. "
        "This indicates torch.compile had a side effect on global state."
    )

if __name__ == "__main__":
    test_compile_torch_eq_preserves_distribution_state()
    print("Test passed.")