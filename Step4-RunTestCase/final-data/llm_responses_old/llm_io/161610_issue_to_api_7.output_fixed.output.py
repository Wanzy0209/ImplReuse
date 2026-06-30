import torch
import sys

def test_cufft_plan_cache_dynamic_attributes():
    """
    Test case to verify if dynamic attributes persist on torch.backends.cuda.cufft_plan_cache
    when passed through torch.compile, similar to the NamedTuple issue.
    """
    if not torch.cuda.is_available():
        print("Skipping test: CUDA is not available.")
        return

    # The API object to test
    cache_manager = torch.backends.cuda.cufft_plan_cache
    attr_name = "dynamic_test_attribute"

    # Ensure clean state before testing
    if hasattr(cache_manager, attr_name):
        try:
            delattr(cache_manager, attr_name)
        except AttributeError:
            # Handle cases where the attribute might not be deletable
            pass

    try:
        def fn(obj):
            # Reproduce the logic from the bug report: setting a dynamic attribute
            setattr(obj, attr_name, 42)
            return obj

        # 1. Test Eager Mode (Baseline)
        print("Testing eager mode...")
        eager_result = fn(cache_manager)
        assert hasattr(eager_result, attr_name), \
            f"Eager mode: Failed to set attribute '{attr_name}' on {type(cache_manager).__name__}"
        assert getattr(eager_result, attr_name) == 42
        print(f"Eager mode result: {getattr(eager_result, attr_name)}")

        # Clean up for the next test
        try:
            delattr(cache_manager, attr_name)
        except AttributeError:
            # Handle cases where the attribute might not be deletable
            pass

        # 2. Test Compiled Mode (Eager Backend)
        # This mirrors the bug report's failing scenario
        print("\nTesting compiled mode (backend='eager')...")
        compiled_fn = torch.compile(fn, backend="eager")
        compiled_result = compiled_fn(cache_manager)

        # Check if the attribute persists (this is where the bug manifests for NamedTuples)
        assert hasattr(compiled_result, attr_name), \
            f"Compiled mode: Attribute '{attr_name}' was lost on {type(cache_manager).__name__}"
        
        assert getattr(compiled_result, attr_name) == 42, \
            f"Compiled mode: Attribute '{attr_name}' has incorrect value"
        
        print(f"Compiled mode result: {getattr(compiled_result, attr_name)}")
        print("\nTest passed: Dynamic attributes persist correctly.")

    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
    finally:
        # Ensure cleanup even if test fails
        try:
            if hasattr(cache_manager, attr_name):
                delattr(cache_manager, attr_name)
        except AttributeError:
            pass

if __name__ == "__main__":
    test_cufft_plan_cache_dynamic_attributes()