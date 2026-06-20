import tensorflow as tf

# Disable eager execution to use compat.v1.Session
tf.compat.v1.disable_eager_execution()

# Create a simple graph
# Note: In a real-world scenario, this graph would contain LiteOp hints.
# For this minimal test case, we create a basic graph to verify the API call.
graph = tf.Graph()
with graph.as_default():
    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 10], name="input")

# Create a session containing the graph
with tf.compat.v1.Session(graph=graph) as sess:
    # Call the API to convert hints to stubs
    # This mirrors the usage of prune.remove in the original issue
    result_graph_def = tf.compat.v1.lite.experimental.convert_op_hints_to_stubs(session=sess)

    # Verify the result is a valid GraphDef
    assert result_graph_def is not None
    print("Conversion successful. Result GraphDef:", result_graph_def)