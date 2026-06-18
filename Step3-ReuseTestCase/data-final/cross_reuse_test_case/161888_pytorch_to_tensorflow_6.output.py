import torch
import tensorflow as tf

print(tf.__version__)

# Create tensors mimicking the PyTorch input types
# PyTorch: tensor1 = torch.randint(low=-100, high=100, size=(9, 3, 7), dtype=torch.int16)
tensor1 = tf.random.uniform(
    shape=(9, 3, 7),
    minval=-100,
    maxval=100,
    dtype=tf.int16
)

# PyTorch: tensor2 = torch.randint(low=0, high=2, size=(1, 6, 4, 8), dtype=torch.bool)
# Note: TF random.uniform for int32 generates integers, then cast to bool
tensor2 = tf.cast(
    tf.random.uniform(
        shape=(1, 6, 4, 8),
        minval=0,
        maxval=2,
        dtype=tf.int32
    ),
    dtype=tf.bool
)

# Replicate the input structure containing invalid arguments
# PyTorch: input = [[[], 154691921484029491302139942063978250367, ()],{},[tensor1,tensor2],{}]
input_data = [
    [[], 154691921484029491302139942063978250367, ()],
    {},
    [tensor1, tensor2],
    {}
]

# Initialize the metric with invalid arguments
# Corresponds to: sensitivity=[], num_thresholds=huge_int, class_id=()
# PyTorch: r1 = torch.nn.MaxUnpool2d(*input[0],**input[1])
r1 = tf.keras.metrics.SpecificityAtSensitivity(*input_data[0], **input_data[1])

# Call the metric with invalid tensor arguments
# Corresponds to: y_true=tensor1, y_pred=tensor2
# PyTorch: r2 = r1(*input[2],**input[3])
r2 = r1(*input_data[2], **input_data[3])