import torch
import tensorflow as tf
import numpy as np

# Disable eager execution to use tf.compat.v1 graph mode features
tf.compat.v1.disable_eager_execution()

def test_string_input_producer_training():
    """
    Adapts the PyTorch ResNet-18 training loop test to TensorFlow 1.x style.
    Instead of torch.compile, we use the implicit graph compilation of TF1.
    Instead of direct tensor creation, we use tf.compat.v1.train.string_input_producer
    to feed data into the training graph.
    """
    
    # Constants from the original test
    BATCH_SIZE = 4
    NUM_CLASSES = 10
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 1

    # Create dummy filenames to simulate the input data pipeline
    # In the original PyTorch code, data was generated via torch.randn.
    # Here we use strings as required by string_input_producer.
    filenames = [f"dummy_image_{i}.jpg" for i in range(10)]

    # --- Use the specific API requested: tf.compat.v1.train.string_input_producer ---
    filename_queue = tf.compat.v1.train.string_input_producer(
        string_tensor=filenames,
        num_epochs=NUM_EPOCHS,
        shuffle=False,
        capacity=32
    )

    # Reader to read the files (mocking the data loading process)
    reader = tf.compat.v1.WholeFileReader()
    key, value = reader.read(filename_queue)

    # Decode and preprocess image (mocking the tensor creation)
    # We assume RGB images for simplicity
    image = tf.image.decode_jpeg(value, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]

    # Batch the data
    batch_images = tf.compat.v1.train.batch([image], batch_size=BATCH_SIZE, capacity=32)
    
    # Mock labels (random generation similar to torch.randint)
    batch_labels = tf.random.uniform([BATCH_SIZE], minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

    # --- Define a simple Model (mimicking ResNet-18 structure conceptually) ---
    def simple_cnn_model(x):
        # Conv1
        x = tf.compat.v1.layers.conv2d(x, filters=64, kernel_size=7, strides=2, padding='same', name='conv1')
        x = tf.compat.v1.layers.batch_normalization(x, name='bn1')
        x = tf.compat.v1.relu(x)
        # MaxPool
        x = tf.compat.v1.layers.max_pooling2d(x, pool_size=3, strides=2, padding='same', name='maxpool')
        # Flatten
        x = tf.compat.v1.layers.flatten(x)
        # FC
        logits = tf.compat.v1.layers.dense(x, units=NUM_CLASSES, name='fc')
        return logits

    logits = simple_cnn_model(batch_images)

    # --- Loss and Optimizer ---
    criterion = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=batch_labels, logits=logits)
    # Note: In TF1, loss.backward() and optimizer.step() are combined in optimizer.minimize
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(criterion)

    # --- Execution ---
    with tf.compat.v1.Session() as sess:
        # Initialize local variables (required for string_input_producer epochs)
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        # Coordinator for managing queue threads
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)

        try:
            # Run the training step (Forward + Backward pass)
            # This corresponds to the train(images, labels) call in the original PyTorch code
            _, loss_val = sess.run([train_op, criterion])
            
            # Assertion to verify the step completed successfully
            assert not np.isnan(loss_val), "Loss is NaN, training failed."
            print(f"Test Passed. Training step completed successfully. Loss: {loss_val}")

        except Exception as e:
            print(f"Test Failed. Error during training: {e}")
            raise
        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    test_string_input_producer_training()