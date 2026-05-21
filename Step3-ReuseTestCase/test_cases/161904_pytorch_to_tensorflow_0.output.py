import torch
"""
Adapted Test Case for TensorFlow (tf.compat.v1.tpu.rewrite)

This test case adapts the PyTorch reproduction logic to TensorFlow.
It verifies if pipeline parallel stages, when compiled using 
tf.compat.v1.tpu.rewrite (analogous to torch.compile), execute correctly.

Run on a TPU node or a TPU-enabled environment:
python repro_tf.py
"""

import tensorflow as tf
import tf.compat.v1 as tfv1
import numpy as np

# Disable eager execution for tf.compat.v1.tpu.rewrite compatibility
tfv1.disable_eager_execution()

# Constants mimicking the original PyTorch script
BATCH_SIZE = 8
SEQ_LEN = 4096
VOCAB_SIZE = 128
HIDDEN_DIM = 32
NUM_LAYERS = 4

def get_stage_model(stage_id, num_stages=2):
    """
    Constructs a portion of the Transformer model to simulate pipeline stages.
    Analogous to pipeline_module_split in the PyTorch script.
    """
    layers = []
    
    # Calculate layer distribution
    layers_per_stage = NUM_LAYERS // num_stages
    start_layer = stage_id * layers_per_stage
    end_layer = (stage_id + 1) * layers_per_stage if stage_id < num_stages - 1 else NUM_LAYERS

    # Stage 0 includes Embedding
    if stage_id == 0:
        layers.append(tf.keras.layers.Embedding(VOCAB_SIZE, HIDDEN_DIM))

    # Add Linear (Dense) layers assigned to this stage
    for i in range(start_layer, end_layer):
        layers.append(tf.keras.layers.Dense(HIDDEN_DIM, use_bias=False, name=f"layer_{i}"))

    # Last stage includes Output projection
    if stage_id == num_stages - 1:
        layers.append(tf.keras.layers.Dense(VOCAB_SIZE, use_bias=False, name="output"))

    def forward_fn(x):
        for layer in layers:
            x = layer(x)
        return x

    return forward_fn

def main():
    # Initialize TPU system
    # Note: This requires a TPU environment. 
    # If running locally without TPU, this will raise an error regarding TPU system initialization.
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tfv1.tpu.initialize_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    except ValueError as e:
        print(f"TPU initialization failed (expected if not on TPU): {e}")
        print("Exiting as tf.compat.v1.tpu.rewrite requires TPU context.")
        return

    # Define Pipeline Stages
    # In the original bug, the model is split and then compiled.
    stage_0_fn = get_stage_model(0)
    stage_1_fn = get_stage_model(1)

    # Define the pipeline computation
    # This mimics the schedule.step() logic where data flows through stages.
    def pipeline_computation(input_ids):
        # Stage 0
        x = stage_0_fn(input_ids)
        # Stage 1
        x = stage_1_fn(x)
        return x

    # Input placeholder
    input_ids = tfv1.placeholder(tf.int32, shape=(BATCH_SIZE, SEQ_LEN), name="input_ids")

    print("Attempting to compile pipeline with tf.compat.v1.tpu.rewrite...")
    
    try:
        # This is the TensorFlow equivalent of torch.compile(model)
        # We wrap the pipeline logic in the rewrite function.
        compiled_op = tfv1.tpu.rewrite(
            pipeline_computation, 
            inputs=[input_ids],
            device_assignment=None
        )

        # Execution
        with tfv1.Session() as sess:
            sess.run(tfv1.global_variables_initializer())
            
            # Generate dummy input data
            dummy_input = np.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
            
            # Run the compiled pipeline
            result = sess.run(compiled_op, feed_dict={input_ids: dummy_input})
            
            print("Test Passed: Pipeline stages compiled and executed successfully with tf.compat.v1.tpu.rewrite.")
            print(f"Output shape: {result[0].shape}")

    except Exception as e:
        print(f"Test Failed: An error occurred during compilation or execution.")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()