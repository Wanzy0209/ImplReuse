import torch
import tensorflow as tf
import numpy as np
import time

def test_sparse_categorical_crossentropy_cuda_stability():
    """
    Test case adapted from Issue 167602 (Stable Diffusion CUDNN Errors).
    Replaces the PyTorch Stable Diffusion pipeline with the similar 
    TensorFlow API (tf.keras.metrics.SparseCategoricalCrossentropy) 
    to test for stability under repeated execution on GPU.
    """
    
    # Check for GPU availability, mimicking .to("cuda")
    gpus = tf.config.list_physical_devices('GPU')
    device_name = '/CPU:0'
    if gpus:
        try:
            # Restrict memory growth to avoid OOM similar to potential CUDA issues
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            device_name = '/GPU:0'
            print(f"Running on GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(e)

    with tf.device(device_name):
        # Initialize the metric (Similar API)
        # This replaces the StableDiffusionPipeline initialization
        scce = tf.keras.metrics.SparseCategoricalCrossentropy()

        # Setup dummy data
        # Mimicking the input tensors for a classification task
        # Using float16 to match the torch_dtype=torch.float16 in the original bug report
        batch_size = 8
        num_classes = 1000
        
        y_true = tf.random.uniform((batch_size,), maxval=num_classes, dtype=tf.int32)
        y_pred = tf.random.uniform((batch_size, num_classes), dtype=tf.float16)

        # Reproduce the loop logic from the original bug
        prompt = "a photo of an astronaut riding a horse on mars" # Kept for semantic context, though unused in TF metric
        start = time.time()
        
        for i in range(10):
            # Perform the operation (Similar API usage)
            # Replaces: image = pipe(prompt).images[0]
            loss = scce(y_true, y_pred)
            
            # Assert to catch CUDNN errors which often result in NaNs or Infs
            tf.debugging.assert_all_finite(loss, "Loss calculation resulted in NaN or Inf (Potential CUDNN Error)")
            
        print("Time taken: ", time.time() - start)
        print("Test passed: Metric calculated successfully over 10 iterations.")

if __name__ == "__main__":
    test_sparse_categorical_crossentropy_cuda_stability()