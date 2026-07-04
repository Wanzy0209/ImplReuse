import os
import tensorflow as tf
from tensorflow.experimental import dtensor

def test_dtensor_jobs():
    """
    Adapted test case for tf.experimental.dtensor.jobs based on the 
    structure of the PyTorch sparse.mm bug report.
    
    The original bug involved setting up specific inputs (sparse matrices),
    performing an operation, and verifying the result (which caused a crash).
    
    Here, we set up specific inputs (environment variables for DTENSOR_JOBS),
    perform the operation (calling jobs()), and verify the behavior 
    (checking for correct parsing and validation logic).
    """
    
    # Store original environment state to restore later
    original_env = os.environ.get("DTENSOR_JOBS")

    try:
        # --- Test Case 1: Standard Operation (Analogous to the working PyTorch example) ---
        # Setup inputs
        job_list_str = "worker0,worker1,worker2"
        os.environ["DTENSOR_JOBS"] = job_list_str
        
        # Perform operation
        result = dtensor.jobs()
        
        # Verify result (Analogous to to_dense() inspection)
        expected = ["worker0", "worker1", "worker2"]
        assert result == expected, f"Expected {expected}, but got {result}"
        print("Test Case 1 Passed: Standard job list parsed correctly.")

        # --- Test Case 2: Edge Case / Validation Logic (Analogous to the crashing PyTorch example) ---
        # The PyTorch bug occurred with specific sparse inputs. 
        # The dtensor.jobs API has specific validation logic for BNS style names.
        # We test the unsorted BNS path which raises a ValueError.
        
        bns_jobs_unsorted = "/bns/worker2,/bns/worker1"
        os.environ["DTENSOR_JOBS"] = bns_jobs_unsorted
        
        try:
            # Perform operation
            result = dtensor.jobs()
            # If we reach here, the validation logic failed (unexpected behavior)
            print("Test Case 2 Failed: Expected ValueError for unsorted BNS names, but got none.")
            assert False, "Expected ValueError for unsorted BNS names"
        except ValueError as e:
            # Verify the error message matches the implementation logic
            assert "Unexpected DTENSOR_JOBS content" in str(e)
            print("Test Case 2 Passed: Correctly raised ValueError for unsorted BNS names.")

        # --- Test Case 3: Empty Input ---
        os.environ["DTENSOR_JOBS"] = ""
        result = dtensor.jobs()
        assert result == [], f"Expected empty list, but got {result}"
        print("Test Case 3 Passed: Empty string handled correctly.")

    finally:
        # Cleanup: Restore original environment state
        if original_env is None:
            os.environ.pop("DTENSOR_JOBS", None)
        else:
            os.environ["DTENSOR_JOBS"] = original_env

if __name__ == "__main__":
    test_dtensor_jobs()