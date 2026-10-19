```python
import os
import tensorflow as tf

# dynamo.config.optimize_ddp = True  # <== necessary for the bug
# Conversion: TensorFlow handles DDP optimization automatically via MirroredStrategy.

# LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
# dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
# Conversion: TensorFlow uses MirroredStrategy for single-node multi-GPU training.
# It automatically detects available GPUs and initializes the process group.
strategy = tf.distribute.MirroredStrategy()

# Conversion: PyTorch Autograd Function maps to tf.custom_gradient
@tf.custom_gradient
def simplest_double_fn(x):
    def grad(dy):
        return dy * 2
    return x * 2, grad

class DoubleLayer(tf.keras.layers.Layer):
    def call(self, x):
        return simplest_double_fn(x)

def main():
    # device = torch.device(f"cuda:{LOCAL_RANK}")
    # Conversion: Device placement is handled implicitly by strategy.scope()
    
    with strategy.scope():
        # model = nn.Sequential(nn.Conv2d(3,3,3,padding=1), DoubleLayer()).to(device)
        # Conversion: padding=1 with kernel_size=3 is equivalent to padding='same'
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='same'),
            DoubleLayer()
        ])

        # model = torch.compile(model)
        # Conversion: tf.function compiles the graph. We apply it to the training step.
        
        # model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
        # Conversion: MirroredStrategy handles the data parallelism and device mapping.
        
        opt = tf.keras.optimizers.SGD(learning_rate=1e-4)
        loss_fn = tf.keras.losses.MeanSquaredError()

        @tf.function # Equivalent to torch.compile for the training loop
        def train_step(x):
            with tf.GradientTape() as tape:
                out = model(x, training=True)
                loss = loss_fn(out, x)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

    for it in range(3):
        # x = torch.rand(2,3,256,256, device=device)
        # Conversion: Create random data. In TF, data is usually created on CPU and distributed.
        x = tf.random.uniform((2, 3, 256, 256))

        # out = model(x)
        # loss = F.mse_loss(out, x)
        # opt.zero_grad(set_to_none=True)
        # loss.backward()
        # opt.step()
        loss = train_step(x)

        # print(f"[rank{LOCAL_RANK}] iter={it+1} loss={loss.item():.6f}")
        # Conversion: TF MirroredStrategy runs on all replicas. 
        # Accessing specific rank ID for printing is non-trivial in standard TF loops.
        # We print the loss value (which is the same across replicas in this setup).
        print(f"iter={it+1} loss={loss:.6f}")

    # dist.destroy_process_group()
    # Conversion: Cleanup is handled automatically when the script exits.

if __name__ == "__main__":
    main()
```