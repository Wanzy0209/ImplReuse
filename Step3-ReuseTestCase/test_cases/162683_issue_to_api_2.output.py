import torch
import tensorflow as tf
import time

# Ensure we are in a context where all_v2_summary_ops returns a list (Graph mode)
# to make the benchmark meaningful, as it returns None in eager execution.
tf.compat.v1.disable_eager_execution()

# Analogous to 'shapes' in the original bug report, we define different 
# configurations of summary operations to benchmark.
summary_configs = [10, 50]

def benchmark_summary_ops(num_summaries, repeat=500):
    """
    Benchmarks the performance of tf.compat.v1.summary.all_v2_summary_ops.
    This function mirrors the structure of the original torch.matmul benchmark,
    adapting the logic to test the retrieval of summary ops from the graph.
    """
    # Create a graph and populate it with summary ops
    g = tf.compat.v1.Graph()
    with g.as_default():
        for i in range(num_summaries):
            tf.compat.v1.summary.scalar(f'summary_{i}', tf.constant(1.0))
        
        # Warm up: Mimics the original warmup loop
        for _ in range(5000):
            _ = tf.compat.v1.summary.all_v2_summary_ops()

        # Run: Mimics the original timing loop
        times = []
        for i in range(repeat):
            start = time.time()
            ops = tf.compat.v1.summary.all_v2_summary_ops()
            end = time.time()
            
            # Functional assertion to ensure the API works as expected
            assert ops is not None, "all_v2_summary_ops should return a list in graph mode"
            assert len(ops) == num_summaries, f"Expected {num_summaries} ops, got {len(ops)}"

            if i > 100:
                times.append(round((end - start) * 1000 * 1000))
        
        times.sort()
        avg_time_us = sum(times) / len(times)
        return avg_time_us

if __name__ == "__main__":
    print(f"Running benchmark for tf.compat.v1.summary.all_v2_summary_ops...")
    for count in summary_configs:
        t = benchmark_summary_ops(count)
        print(f"Summary Count: {count}  ->  Avg Time: {t:.3f} us")