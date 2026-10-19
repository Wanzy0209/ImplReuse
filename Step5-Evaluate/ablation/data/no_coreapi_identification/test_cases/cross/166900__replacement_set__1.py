import torch
import tensorflow as tf

# Replacing torch.utils._pytree with tf.test.Benchmark
# We structure the test using the tf.test.Benchmark class to handle execution.

class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# Replacing torch.compile with tf.test.Benchmark
# We use a Benchmark class to wrap the logic, utilizing XLA (TF's compiler) 
# to mimic the compilation aspect of the original bug.

class TestDynamoEquivalent(tf.test.Benchmark):
    def benchmark_custom_object_graph(self):
        # Replacing torch.ones with torch.zeros -> translated to tf.zeros
        x = tf.zeros(3)
        obj = Foo()

        # The logic: obj.attr = {3: Bar()}, return x + 1
        # In TF graph mode (XLA), setting attributes on python objects is restricted.
        # We use tf.py_function to allow the side effect, similar to how Dynamo 
        # attempts to handle Python objects.
        
        @tf.function(jit_compile=True)
        def fn(x_input):
            # Wrap the side effect in py_function to allow it in the graph
            def side_effect():
                obj.attr = {3: Bar()}
                return x_input + 1
            
            return tf.py_function(side_effect, [], tf.float32)

        # Execute the compiled function
        # The benchmark_run method handles the execution loop
        self.run_op_benchmark(
            tf.compat.v1.Session(), 
            lambda: fn(x), 
            name="custom_obj_test"
        )

if __name__ == "__main__":
    # Instantiate and run the benchmark
    test = TestDynamoEquivalent()
    test.benchmark_custom_object_graph()