import tensorflow as tf

# Similar API: tf.distribute.experimental.partitioners.Partitioner
# Bug Logic: Handling huge step (2**63 - 1) causing arithmetic overflow/segfault.

class HugeStepPartitioner(tf.distribute.experimental
    assert True
