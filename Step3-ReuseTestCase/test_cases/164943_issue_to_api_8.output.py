import tensorflow as tf
import unittest

class TestDistributeStrategyAvailability(unittest.TestCase):
    def test_get_strategy_initialization(self):
        """
        Test that tf.distribute.get_strategy() correctly initializes and returns 
        a strategy object on the current environment. This mirrors the PyTorch 
        MPS backend availability check, ensuring that the library correctly 
        identifies the environment capabilities without raising spurious 
        version-related errors.
        """
        # Mimics the 'foo.to('mps')' call which triggers backend availability checks.
        # In TensorFlow, accessing the strategy triggers the initialization of the 
        # distribution context.
        strategy = tf.distribute.get_strategy()
        
        # Assert that a strategy object is returned (analogous to successful backend assignment)
        self.assertIsNotNone(strategy)
        self.assertIsInstance(strategy, tf.distribute.Strategy)

        # Mimics the tensor creation and usage within the backend context
        with strategy.scope():
            foo = tf.constant([[1]])
            self.assertEqual(foo.shape, (1, 1))
            self.assertIsNotNone(foo.device)

if __name__ == '__main__':
    unittest.main()