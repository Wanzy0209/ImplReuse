import os
import torch

def test_dynamo_resume_with_torchelastic_check():
    """
    Test case for Issue 162313 adapted to use torch.distributed.is_torchelastic_launched.
    
    The original bug occurs when a global boolean flag changes between calls to a 
    compiled function, causing a KeyError in resume_execution.py due to incorrect 
    block_target_offset_remap handling.
    
    This test replaces the hardcoded 'flag' with torch.distributed.is_torchelastic_launched(),
    which checks an environment variable. We manipulate this variable between calls to 
    simulate the changing condition that triggers the bug.
    """
    
    # Ensure the environment variable is not set initially
    if "TORCHELASTIC_RUN_ID" in os.environ:
        del os.environ["TORCHELASTIC_RUN_ID"]

    @torch.compile(backend="eager")
    def fn(x):
        x = x + 1
        torch._dynamo.graph_break()
        x = x + 2
        
        # Use the similar API to determine control flow
        # This replaces the 'if flag:' logic from the original bug report
        if torch.distributed.is_torchelastic_launched():
            with torch.no_grad():
                torch._dynamo.graph_break()
        else:
            with torch.no_grad():
                torch._dynamo.graph_break()
                
        return x + 4

    # First call: is_torchelastic_launched() returns False
    # This compiles the function for the 'else' branch
    input_tensor = torch.ones(3)
    result1 = fn(input_tensor)
    
    # Change the environment state to trigger the other branch
    # This simulates 'flag = False' -> 'flag = True' from the original bug
    os.environ["TORCHELASTIC_RUN_ID"] = "test_job_id"
    
    # Second call: is_torchelastic_launched() returns True
    # This triggers recompilation/resume logic where the KeyError originally occurred
    result2 = fn(input_tensor)
    
    # Clean up
    del os.environ["TORCHELASTIC_RUN_ID"]
    
    # Basic assertions to ensure execution completed
    assert torch.allclose(result1, torch.tensor([4.0, 4.0, 4.0]))
    assert torch.allclose(result2, torch.tensor([4.0, 4.0, 4.0]))
    
    print("Test passed: No KeyError encountered during resume function creation.")

if __name__ == "__main__":
    test_dynamo_resume_with_torchelastic_check()