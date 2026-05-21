import tensorflow as tf

# Define a simple model similar to the PyTorch SimpleModel
class SimpleModel(tf.keras.Model):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleModel, self).__init__()
        self.linear1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.linear2 = tf.keras.layers.Dense(output_size)

    def call(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# Check for GPU availability (analogous to checking custom backend availability)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Attempt to run the model using the XLA backend via tf.xla.experimental.compile
        # This parallels the usage of DataParallel to execute on a specific backend path
        
        model = SimpleModel()
        
        batch_size = 20
        # Create input data
        input_data = tf.random.normal((batch_size, 10))

        # Define the computation function to be compiled
        def computation_fn(x):
            return model(x)

        # Compile and run the computation using XLA
        # This is the TensorFlow equivalent API for running a model on a specific backend/compilation path
        outputs = tf.xla.experimental.compile(computation_fn, inputs=[input_data])
        
        # Extract the output (compile returns a list of tensors)
        output = outputs[0]

        # Verify the output shape matches expectations
        assert output.shape == (batch_size, 5), f"Expected shape (20, 5), got {output.shape}"
        print("success")

    except Exception as e:
        print(f"Test failed: {e}")
        raise
else:
    raise RuntimeError("No GPU available to test backend support")