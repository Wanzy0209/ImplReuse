import torch
import torch.distributions

# Handle the import error for older PyTorch versions where impl_abstract does not exist
try:
    from torch.library import impl_abstract
except ImportError:
    impl_abstract = None

def test_impl_abstract_distribution_side_effect():
    """
    Test that torch.library.impl_abstract does not inadvertently change
    global distribution validation settings, unlike the torch.compile
    behavior described in the bug report.
    """
    # Skip test if the required API is not available
    if impl_abstract is None:
        print("Skipping test: torch.library.impl_abstract is not available in this PyTorch version (requires PyTorch 2.0+).")
        return

    # Get the initial state of distribution validation args
    initial_state = torch.distributions.Distribution.get_default_validate_args()

    # Define a custom library and operator to register a fake impl for
    lib = torch.Library("test_custom_lib", "DEF")
    lib.define("custom_op(Tensor x) -> Tensor")

    # Define the abstract implementation (fake kernel)
    def custom_op_meta(x):
        return torch.empty_like(x)

    # Call the similar API: torch.library.impl_abstract
    # In the original bug, torch.compile was called here which triggered
    # torch.distributions.Distribution.set_default_validate_args(False)
    impl_abstract("test_custom_lib::custom_op", custom_op_meta)

    # Check the state after calling the API
    final_state = torch.distributions.Distribution.get_default_validate_args()

    # Assert that the global state has not changed
    assert initial_state == final_state, (
        f"torch.library.impl_abstract changed global distribution validation args "
        f"from {initial_state} to {final_state}"
    )

    print("Test passed: torch.library.impl_abstract does not affect global distribution validation args.")

if __name__ == "__main__":
    test_impl_abstract_distribution_side_effect()