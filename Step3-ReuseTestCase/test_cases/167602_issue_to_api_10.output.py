import torch
import tensorflow as tf
import os

def test_stable_diffusion_like_workflow_with_copy():
    """
    Test case adapted from Issue 167602 (PyTorch Stable Diffusion CUDNN error).
    
    This test preserves the original bug reproduction logic (loading a model, 
    moving to GPU, running inference in a loop, saving output) while leveraging 
    the similar API pattern (tf.io.gfile.copy) for handling the output file.
    """
    
    # Check for GPU availability (mimicking pipe.to("cuda"))
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("Warning: No GPU found. This test is intended to stress CUDNN on GPU.")
    else:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Define a model that uses Attention (mimicking the core of Stable Diffusion)
    # This aligns with the Original API Under Test: torch.nn.functional.scaled_dot_product_attention
    inputs = tf.keras.Input(shape=(10, 64))
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    model = tf.keras.Model(inputs=inputs, outputs=attention_output)

    # Prepare dummy input data
    batch_size = 1
    dummy_input = tf.random.normal((batch_size, 10, 64))

    # Run inference loop
    # Mimicking: for i in range(10): image = pipe(prompt).images[0]
    print("Starting inference loop...")
    for i in range(10):
        output = model(dummy_input, training=False)
    
    print("Inference complete.")

    # Save the result to a temporary file
    # Mimicking: image.save("astronaut_rides_horse.png")
    temp_file_path = "/tmp/tf_sd_output.bin"
    destination_path = "/tmp/tf_sd_output_copy.bin"
    
    try:
        # Serialize tensor to file
        tf.io.write_file(temp_file_path, tf.io.serialize_tensor(output))
        
        # Use the Similar API: tf.io.gfile.copy
        # This matches the code pattern from the similar API information
        print(f"Copying output from {temp_file_path} to {destination_path}...")
        tf.io.gfile.copy(temp_file_path, destination_path, overwrite=True)
        
        # Verify the copy operation succeeded
        assert tf.io.gfile.exists(destination_path), "Copy operation failed: destination file does not exist."
        
        # Verify content integrity
        original_content = tf.io.read_file(temp_file_path)
        copied_content = tf.io.read_file(destination_path)
        assert tf.reduce_all(tf.equal(original_content, copied_content)), "Content mismatch after copy."
        
        print("Test passed: Output generated and copied successfully.")
        
    finally:
        # Cleanup
        if tf.io.gfile.exists(temp_file_path):
            tf.io.gfile.remove(temp_file_path)
        if tf.io.gfile.exists(destination_path):
            tf.io.gfile.remove(destination_path)

if __name__ == "__main__":
    test_stable_diffusion_like_workflow_with_copy()