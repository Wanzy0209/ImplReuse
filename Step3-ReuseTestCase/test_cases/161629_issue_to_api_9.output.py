import os
import unittest
from unittest.mock import patch

class TestFullJobNameDistributedLogic(unittest.TestCase):
    """
    Test case for tf.experimental.dtensor.full_job_name based on the 
    logic similarity to the PyTorch TORCH_CUDA_ARCH_LIST warning issue (Issue 161629).
    
    The PyTorch issue involves environment variable checks and distributed awareness 
    (printing warnings on all ranks vs rank 0). This test verifies that the similar 
    TF API correctly handles environment variables and distributed context (client IDs),
    ensuring it behaves correctly when the environment variable is unset or set, 
    mirroring the desired logic in the bug report.
    """

    def test_env_var_handling_and_distributed_awareness(self):
        """
        Tests the logic of checking environment variables in a distributed context.
        
        1. Default Behavior: When the env var is not set, the API should default 
           to a sensible value (client 0), similar to how PyTorch defaults to 'all archs'.
        2. Explicit Behavior: When the env var is set, the API should respect it.
        3. Distributed Awareness: The API should validate consistency between the 
           env var and the distributed context (e.g., local run vs multi-client).
        """
        # We mock the dependencies of full_job_name to isolate the logic
        # similar to how one would isolate the warning logic in cpp_extension.
        with patch('tensorflow.experimental.dtensor.num_clients') as mock_num_clients, \
             patch('tensorflow.experimental.dtensor.job_name') as mock_job_name:
            
            mock_job_name.return_value = "worker"
            
            # Case 1: Environment variable is not set (Default behavior)
            # PyTorch Bug Context: TORCH_CUDA_ARCH_LIST is not set.
            # TF API Context: _DT_CLIENT_ID is not set, should default to 0.
            with patch.dict(os.environ, {}, clear=True):
                mock_num_clients.return_value = 1
                
                from tensorflow.experimental import dtensor
                result = dtensor.full_job_name()
                
                # Verify it defaults to task:0
                self.assertIn("task:0", result, 
                              "Should default to task 0 when env var is not set")

            # Case 2: Environment variable is set explicitly
            # PyTorch Bug Context: User sets TORCH_CUDA_ARCH_LIST to specific archs.
            # TF API Context: User sets _DT_CLIENT_ID to a specific ID.
            with patch.dict(os.environ, {'DT_CLIENT_ID': '2'}):
                mock_num_clients.return_value = 3 # Simulate a multi-client environment
                
                result = dtensor.full_job_name()
                
                # Verify it uses the value from the env var
                self.assertIn("task:2", result, 
                              "Should use the env var value for task ID")

            # Case 3: Distributed Awareness / Consistency Check
            # PyTorch Bug Context: Warning should only appear on rank 0 (distributed awareness).
            # TF API Context: The code checks if num_clients() == 1 and task_id != 0.
            # This ensures that in a local run, we don't have inconsistent task IDs.
            with patch.dict(os.environ, {'DT_CLIENT_ID': '1'}):
                mock_num_clients.return_value = 1 # Simulate a local run (1 client)
                
                # This should raise a ValueError due to inconsistency (distributed awareness)
                with self.assertRaises(ValueError) as context:
                    dtensor.full_job_name()
                
                self.assertIn("Unexpected task ID", str(context.exception),
                              "Should raise error for inconsistent task ID in local run")

if __name__ == '__main__':
    unittest.main()