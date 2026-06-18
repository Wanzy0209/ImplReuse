import tensorflow as tf
import tensorflow.compat.v1 as tf_compat

# Create dummy tensors to serve as inputs for the signature definition
# examples must be a string Tensor
examples = tf.constant(["serialized_example_1", "serialized_example_2"], dtype=tf.string)
# classes must be a string Tensor
classes = tf.constant(["class_a", "class_b"], dtype=tf.string)
# scores must be a float Tensor
scores = tf.constant([0.8, 0.2], dtype=tf.float32)

# Call the target API to create the classification signature
classification_sig = tf_compat.saved_model.classification_signature_def(
    examples=examples,
    classes=classes,
    scores=scores
)

# Verify the result is a valid signature definition
assert classification_sig is not None
assert 'inputs' in classification_sig
assert 'outputs' in classification_sig
assert classification_sig.method_name == 'tensorflow/serving/classify'

print("Test passed. Signature created successfully.")