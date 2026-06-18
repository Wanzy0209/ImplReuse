import torch
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, WeightNormalization
from tensorflow.keras.optimizers import SGD

if __name__ == "__main__":
    # Reproduce the condition d > 64 which triggered the bug in PyTorch
    d = 65
    
    # PyTorch uses NCHW format (Batch, Channel, Height, Width)
    # TensorFlow uses NHWC format (Batch, Height, Width, Channel) by default
    x = tf.random.normal((1, 32, 32, 2))

    # Adapt the logic to use the target API: tf.keras.backend.name_scope
    # This replaces the torch.compile context, grouping operations under a specific name scope
    with tf.keras.backend.name_scope("compiled_model_scope"):
        # Define Conv2D layer
        # PyTorch: nn.Conv2d(2, d, 2) -> in_channels=2, out_channels=d, kernel_size=2
        conv_layer = Conv2D(filters=d, kernel_size=2, input_shape=(32, 32, 2))
        
        # Apply Weight Normalization
        # PyTorch: weight_norm(nn.Conv2d(...))
        # TensorFlow: WeightNormalization wrapper
        model = WeightNormalization(conv_layer)

    opt = SGD()

    # Training loop
    for _ in range(1000):
        with tf.GradientTape() as tape:
            # Forward pass
            # training=True is essential for layers like BatchNorm or Dropout, 
            # and generally good practice when mimicking .train() mode
            output = model(x, training=True)
            loss = tf.reduce_mean(output)

        # Backward pass
        # PyTorch: loss.backward()
        grads = tape.gradient(loss, model.trainable_variables)
        
        # Optimizer step
        # PyTorch: opt.step()
        opt.apply_gradients(zip(grads, model.trainable_variables))
        
        # PyTorch: opt.zero_grad() is handled implicitly by new GradientTape in TF

    print("Test passed: Backward pass completed successfully with name_scope.")