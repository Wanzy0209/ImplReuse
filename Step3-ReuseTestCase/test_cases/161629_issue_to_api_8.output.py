import unittest
import logging
from unittest import mock
import tensorflow as tf

# The issue reports a bug where a warning is printed on all ranks in a distributed setup,
# at a log level that is too verbose (WARNING), and lacks specific details about what is being set.
# This test case targets the similar API tf.tpu.experimental.initialize_tpu_system
# to verify that it handles logging levels correctly and provides informative messages.

class TestTPUInitializationLogging(unittest.TestCase):

    def setUp(self):
        # Reset logging configuration before each test
        self.logger = logging.getLogger('tensorflow')
        self.original_level = self.logger.level
        self.logger.setLevel(logging.INFO)

    def tearDown(self):
        self.logger.setLevel(self.original_level)

    @mock.patch('tensorflow.dtensor.python.tpu_util.context')
    @mock.patch('tensorflow.dtensor.python.tpu_util.config')
    @mock.patch('tensorflow.dtensor.python.tpu_util.tpu_system_init_helper')
    def test_initialize_logs_informative_message(self, mock_init_helper, mock_config, mock_context):
        """
        Corresponds to Bug Requirement 3: 
        "the message could be improved to telling the user what it sets the env var to"
        
        Verifies that initialize_tpu_system logs the specific configuration 
        (e.g., TFRT runtime status) rather than a generic warning.
        """
        # Mock the TPU context and configuration
        mock_context.context.return_value.use_tfrt = True
        mock_config.client_id.return_value = 0
        mock_config.num_clients.return_value = 1
        mock_config.num_global_devices.return_value = 8
        
        # Mock the topology object to simulate a successful init
        mock_topology = mock.MagicMock()
        mock_topology.mesh_shape = [2, 2, 1]
        mock_topology.device_coordinates = [[0,0,0], [0,0,1]]
        mock_device = mock.MagicMock()
        mock_init_helper.return_value = (mock_topology, mock_device)

        # Capture logs to verify content
        with self.assertLogs('tensorflow', level='INFO') as cm:
            tf.tpu.experimental.initialize_tpu_system()

        # Verify the log message contains the specific setting value
        # PyTorch bug wanted: "setting TORCH_CUDA_ARCH_LIST=9.0..."
        # TF API logs: "Using TFRT host runtime is set to True"
        self.assertTrue(any("Using TFRT host runtime is set to True" in message for message in cm.output))
        self.assertTrue(any("TPU Topology" in message for message in cm.output))

    @mock.patch('tensorflow.dtensor.python.tpu_util.context')
    @mock.patch('tensorflow.dtensor.python.tpu_util.config')
    @mock.patch('tensorflow.dtensor.python.tpu_util.tpu_system_init_helper')
    def test_initialize_respects_log_level(self, mock_init_helper, mock_config, mock_context):
        """
        Corresponds to Bug Requirement 2: 
        "unless the logging level is set to DEBUG/INFO it shouldn't be printed at all"
        
        Verifies that initialize_tpu_system does not print INFO logs 
        when the logging level is set to WARNING.
        """
        # Setup mocks
        mock_context.context.return_value.use_tfrt = True
        mock_config.client_id.return_value = 0
        mock_config.num_clients.return_value = 1
        mock_config.num_global_devices.return_value = 8
        mock_init_helper.return_value = (mock.MagicMock(), mock.MagicMock())

        # Set logger to WARNING (simulating a user who doesn't want INFO logs)
        self.logger.setLevel(logging.WARNING)

        # Use a custom handler to capture records and verify absence of INFO logs
        class ListHandler(logging.Handler):
            def __init__(self):
                super().__init__()
                self.records = []
            def emit(self, record):
                self.records.append(record)

        handler = ListHandler()
        self.logger.addHandler(handler)
        
        try:
            tf.tpu.experimental.initialize_tpu_system()
            
            # Check that no INFO level logs were emitted
            info_logs = [r for r in handler.records if r.levelno == logging.INFO]
            self.assertEqual(len(info_logs), 0, 
                             "INFO logs should not be emitted when logger is set to WARNING")
        finally:
            self.logger.removeHandler(handler)

if __name__ == '__main__':
    unittest.main()