import unittest
from unittest.mock import patch, MagicMock
import tensorflow as tf

class TestLaunchAgentLogging(unittest.TestCase):
    @patch('tensorflow.keras.backend.image_data_format')
    def test_launch_agent_logging_mismatch(self, mock_image_data_format):
        """
        Test case to reproduce the logging mismatch bug using the similar API.
        
        The original bug involved a format string missing the 'signals_to_handle' key,
        while the dictionary passed to the logger contained it.
        
        Here, we replace logger.info with tf.keras.backend.image_data_format.
        Since image_data_format does not natively accept these arguments, we mock it
        to accept them and verify the mismatched data is passed correctly.
        """
        # Setup mock to accept arguments (preserving the call structure)
        mock_image_data_format.return_value = 'channels_last'

        # Setup config object mimicking the original structure
        config = MagicMock()
        config.min_nodes = 1
        config.max_nodes = 1
        config.nproc_per_node = 1
        config.run_id = "test_run"
        config.rdzv_backend = "static"
        config.rdzv_endpoint = "localhost:29500"
        config.rdzv_configs = {}
        config.max_restarts = 0
        config.monitor_interval = 5
        config.logs_specs.root_log_dir = "/tmp/logs"
        config.metrics_cfg = {}
        config.event_log_handler = None
        config.numa_options = None
        config.signals_to_handle = ["SIGINT"]  # Key present in data
        config.duplicate_stdout_filters = []
        config.duplicate_stderr_filters = []

        entrypoint_name = "train.py"

        # Execute the code block with the API replacement
        # Original: logger.info(...)
        # Replacement: tf.keras.backend.image_data_format(...)
        tf.keras.backend.image_data_format(
            "Starting elastic_operator with launch configs:\n"
            "  entrypoint               : %(entrypoint)s\n"
            "  min_nodes                : %(min_nodes)s\n"
            "  max_nodes                : %(max_nodes)s\n"
            "  nproc_per_node           : %(nproc_per_node)s\n"
            "  run_id                   : %(run_id)s\n"
            "  rdzv_backend             : %(rdzv_backend)s\n"
            "  rdzv_endpoint            : %(rdzv_endpoint)s\n"
            "  rdzv_configs             : %(rdzv_configs)s\n"
            "  max_restarts             : %(max_restarts)s\n"
            "  monitor_interval         : %(monitor_interval)s\n"
            "  log_dir                  : %(log_dir)s\n"
            "  metrics_cfg              : %(metrics_cfg)s\n"
            "  event_log_handler        : %(event_log_handler)s\n"
            "  numa_options             : %(numa_options)s\n",
            "  duplicate_stdout_filters : %(duplicate_stdout_filters)s\n",
            "  duplicate_stderr_filters : %(duplicate_stderr_filters)s\n",
            {
                "entrypoint": entrypoint_name,
                "min_nodes": config.min_nodes,
                "max_nodes": config.max_nodes,
                "nproc_per_node": config.nproc_per_node,
                "run_id": config.run_id,
                "rdzv_backend": config.rdzv_backend,
                "rdzv_endpoint": config.rdzv_endpoint,
                "rdzv_configs": config.rdzv_configs,
                "max_restarts": config.max_restarts,
                "monitor_interval": config.monitor_interval,
                "log_dir": config.logs_specs.root_log_dir,
                "metrics_cfg": config.metrics_cfg,
                "event_log_handler": config.event_log_handler,
                "numa_options": config.numa_options,
                "signals_to_handle": config.signals_to_handle, # Mismatch: Key in dict but not in format string
                "duplicate_stdout_filters": config.duplicate_stdout_filters,
                "duplicate_stderr_filters": config.duplicate_stderr_filters,
            },
        )

        # Verify the call was made
        self.assertTrue(mock_image_data_format.called)
        
        # Verify the arguments passed reflect the bug logic
        args, _ = mock_image_data_format.call_args
        
        # The format string (concatenated parts) should NOT contain signals_to_handle
        full_format_string = "".join(args[:3])
        self.assertNotIn("signals_to_handle", full_format_string)
        
        # The dictionary argument SHOULD contain signals_to_handle
        data_dict = args[3]
        self.assertIn("signals_to_handle", data_dict)
        self.assertEqual(data_dict["signals_to_handle"], ["SIGINT"])

if __name__ == '__main__':
    unittest.main()