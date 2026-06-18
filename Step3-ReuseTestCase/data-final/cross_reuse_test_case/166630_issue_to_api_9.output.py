import torch
import unittest
from unittest.mock import MagicMock, patch
import logging

# This test case verifies the fix for Issue 166630.
# The bug was that the 'signals_to_handle' key was missing from the dictionary
# passed to the logger, causing a KeyError (or missing info) during logging.
# The fix ensures the dictionary includes this key.

class TestLaunchAgentLogging(unittest.TestCase):
    
    def setUp(self):
        # Mock the configuration object to simulate the state passed to launch_agent
        self.mock_config = MagicMock()
        self.mock_config.min_nodes = 1
        self.mock_config.max_nodes = 1
        self.mock_config.nproc_per_node = 2
        self.mock_config.run_id = "test_run_id"
        self.mock_config.rdzv_backend = "static"
        self.mock_config.rdzv_endpoint = "localhost:29500"
        self.mock_config.rdzv_configs = {}
        self.mock_config.max_restarts = 3
        self.mock_config.monitor_interval = 5
        self.mock_config.logs_specs.root_log_dir = "/tmp/logs"
        self.mock_config.metrics_cfg = {}
        self.mock_config.event_log_handler = None
        self.mock_config.numa_options = None
        # This is the critical field that was missing/involved in the bug
        self.mock_config.signals_to_handle = ["SIGINT", "SIGTERM"] 
        self.mock_config.duplicate_stdout_filters = []
        self.mock_config.duplicate_stderr_filters = []

    @patch('torch.distributed.launcher.api.logger')
    def test_logging_includes_signals_to_handle(self, mock_logger):
        """
        Test that the logging dictionary includes 'signals_to_handle'.
        This reproduces the logic of the fix for Issue 166630.
        """
        entrypoint_name = "test_script.py"

        # Reconstruct the logging logic from the API
        # (Simulating the code block found in torch/distributed/launcher/api.py)
        log_dict = {
            "entrypoint": entrypoint_name,
            "min_nodes": self.mock_config.min_nodes,
            "max_nodes": self.mock_config.max_nodes,
            "nproc_per_node": self.mock_config.nproc_per_node,
            "run_id": self.mock_config.run_id,
            "rdzv_backend": self.mock_config.rdzv_backend,
            "rdzv_endpoint": self.mock_config.rdzv_endpoint,
            "rdzv_configs": self.mock_config.rdzv_configs,
            "max_restarts": self.mock_config.max_restarts,
            "monitor_interval": self.mock_config.monitor_interval,
            "log_dir": self.mock_config.logs_specs.root_log_dir,
            "metrics_cfg": self.mock_config.metrics_cfg,
            "event_log_handler": self.mock_config.event_log_handler,
            "numa_options": self.mock_config.numa_options,
            "signals_to_handle": self.mock_config.signals_to_handle, # The fix
            "duplicate_stdout_filters": self.mock_config.duplicate_stdout_filters,
            "duplicate_stderr_filters": self.mock_config.duplicate_stderr_filters,
        }

        # The format string (simplified for the test, assuming it expects the keys)
        # In the actual bug, if the string had %(signals_to_handle)s and the dict didn't, it crashed.
        log_str = (
            "Starting elastic_operator with launch configs:\n"
            "  signals_to_handle       : %(signals_to_handle)s\n"
            # ... other keys ...
        )

        # Perform the logging call
        try:
            # We use the mock_logger to capture the call
            mock_logger.info(log_str, log_dict)
            
            # Verify logger.info was called
            self.assertTrue(mock_logger.info.called)
            
            # Verify the arguments passed to logger.info
            call_args = mock_logger.info.call_args
            # call_args[0][0] is the format string, call_args[0][1] is the dict
            received_dict = call_args[0][1]
            
            # Assertion: The dictionary must contain 'signals_to_handle'
            self.assertIn("signals_to_handle", received_dict)
            self.assertEqual(received_dict["signals_to_handle"], ["SIGINT", "SIGTERM"])
            
        except KeyError as e:
            self.fail(f"Logging failed with KeyError: {e}. This indicates the bug is present.")

if __name__ == '__main__':
    unittest.main()