import torch
import torch.distributed as dist
import tempfile
import os
import sys

def test_new_group_dynamic_inputs():
    """
    Test case for torch.distributed.new_group adapted from the context of 
    Issue #163146 (Data dependent error on slices in torch.export).
    
    The original bug involved dynamic slicing based on a tensor value (max_item_num).
    This test verifies that torch.distributed.new_group handles dynamic inputs
    (ranks derived from a tensor) correctly without raising data-dependent errors
    similar to those encountered in torch.export.
    """
    
    # Setup: Initialize a minimal distributed environment (single process)
    # This is required to call torch.distributed.new_group
    if not dist.is_available():
        print("torch.distributed is not available. Skipping test.")
        return

    # Use a temporary file for the initialization method to ensure portability
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        init_method = f"file://{tmp_file.name}"
        
        try:
            # Initialize process group with world_size=1 (single rank)
            dist.init_process_group(
                backend='gloo', 
                init_method=init_method, 
                world_size=1, 
                rank=0
            )

            # Adaptation of the bug scenario:
            # The original bug had 'max_item_num' as a Tensor causing data-dependent issues.
            # We simulate a similar scenario where the input to the API depends on a Tensor value.
            
            # Simulate the dynamic tensor input
            # In the bug: item_embedding[:, :max_item_num, :]
            # Here: ranks determined by a tensor value
            dynamic_rank_limit = torch.tensor(1) # We use 1 because world_size is 1
            
            # Construct ranks list dynamically based on the tensor
            # This mimics the data dependency in the original bug report
            ranks = [i for i in range(dynamic_rank_limit.item())]
            
            # Call the similar API: torch.distributed.new_group
            # We verify that it accepts the dynamically constructed arguments
            print(f"Attempting to create new group with ranks: {ranks}")
            group = dist.new_group(ranks=ranks)
            
            # Assertions
            assert group is not None, "Failed to create new group"
            print("Test passed: torch.distributed.new_group handled dynamic inputs successfully.")

        except Exception as e:
            print(f"Test failed with error: {e}")
            raise
        finally:
            # Cleanup
            if dist.is_initialized():
                dist.destroy_process_group()

if __name__ == "__main__":
    test_new_group_dynamic_inputs()