import torch
import warnings

def test_maxwell_architecture_exclusion():
    """
    Test case to verify that Maxwell (sm_52) architecture is excluded from
    the supported architectures list in PyTorch 2.8.0+, consistent with the
    deprecation warnings.

    This test mirrors the logic of 'tf.debugging.disable_traceback_filtering'
    by verifying the visibility (or lack thereof) of specific internal
    capabilities/architectures. Just as the TF API controls the visibility
    of internal stack frames, this test checks the visibility of legacy
    GPU architectures in the compiled capability list.
    """
    # Retrieve the list of supported CUDA architectures
    # This corresponds to the 'Original API Under Test'
    arch_list = torch.cuda.get_arch_list()

    # The bug report indicates that sm_52 (Maxwell) is missing and the minimum
    # capability is 6.1 (sm_61).
    # We assert that sm_52 is NOT in the list, reproducing the bug scenario.
    assert "sm_52" not in arch_list, \
        "sm_52 (Maxwell) should not be present in the architecture list for PyTorch 2.8.0+"

    # We also assert that the new minimum (sm_61) IS present.
    assert "sm_61" in arch_list, \
        "sm_61 (Pascal) should be the minimum supported architecture"

    # Verify that other older architectures are also filtered out,
    # ensuring the 'filtering' logic is consistent.
    deprecated_archs = ["sm_35", "sm_37", "sm_50"]
    for arch in deprecated_archs:
        assert arch not in arch_list, f"Deprecated architecture {arch} should be filtered out"

    print("Test passed: Architecture list correctly excludes Maxwell and older generations.")

if __name__ == "__main__":
    test_maxwell_architecture_exclusion()