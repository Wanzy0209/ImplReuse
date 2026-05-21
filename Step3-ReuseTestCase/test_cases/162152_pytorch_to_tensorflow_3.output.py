import tensorflow as tf

# Disable eager execution to ensure TF 1.x compatibility behavior for compat.v1 APIs
tf.compat.v1.disable_eager_execution()

# Mimicking SimpleModel: A class that generates the data to be processed
class TextGenerator:
    def __init__(self, text_content="Default Text"):
        self.text_content = text_content

    def get_tensor(self):
        return tf.constant(self.text_content)

# Mimicking the backend check
# The original code checks for a custom backend. Here we check for GPU availability 
# to maintain the logic of verifying hardware/backend support.
if tf.test.is_gpu_available():
    print(f"Detected GPU")

    # Mimicking model creation and moving to GPU 0
    generator = TextGenerator("Sample text for TensorBoard summary")
    
    # Place operations on the GPU
    with tf.device('/GPU:0'):
        text_tensor = generator.get_tensor()
        
        # Mimicking DataParallel(model, device_ids=[...])
        # Applying the target API: tf.compat.v1.summary.text
        # This wraps the tensor into a summary protobuf operation
        summary_op = tf.compat.v1.summary.text("custom_text_summary", text_tensor)

    # Mimicking input_data and output = model(input_data)
    # We need to run the session to execute the operation and get the output
    with tf.compat.v1.Session() as sess:
        # Initialize variables (standard practice in TF 1.x)
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Run the summary operation
        output = sess.run(summary_op)
        
        # Mimicking print("success") and verification
        # Verify that the output is a valid protobuf string (bytes)
        assert output is not None
        assert isinstance(output, bytes)
        print("success")
else:
    # Mimicking the raise in the original code if backend is not available
    raise RuntimeError("GPU not available or required backend not found")