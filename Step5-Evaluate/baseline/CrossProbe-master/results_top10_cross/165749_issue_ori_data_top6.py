```python
import tensorflow as tf

if __name__ == "__main__":
    d = 65
    # Conversion: torch.randn((1, 2, 32, 32)).cuda() -> tf.random.normal with NHWC format on GPU
    with tf.device('/GPU:0'):
        x = tf.random.normal((1, 32, 32, 2))

        # Conversion: weight_norm(nn.Conv2d(2, d, 2))
        # Note: Weight normalization is not natively supported in core TF Keras layers.
        # We use a standard Conv2D layer here.
        # Conversion: torch.compile -> tf.function
        model = tf.keras.layers.Conv2D(filters=d, kernel_size=2)
        model = tf.function(model)

        opt = tf.keras.optimizers.SGD()

        for _ in range(1000):
            with tf.GradientTape() as tape:
                # Conversion: model(x).mean().backward()
                loss = tf.reduce_mean(model(x))
            
            # Conversion: opt.step() and opt.zero_grad()
            # Note: TF computes gradients fresh every iteration, so no zero_grad is needed.
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
```