import torch
import tensorflow as tf
import unittest
import gc
import psutil

def mem_mb_print():
    gc.collect()
    mem = psutil.virtual_memory()
    process = psutil.Process().memory_info().rss

    # Replacement: torch.is_tensor -> torch.nn.utils.prune.is_pruned
    # Note: is_pruned expects a torch.nn.Module, so we check type to avoid crash
    pruned_count = sum(
        1 for o in gc.get_objects() 
        if isinstance(o, torch.nn.Module) and torch.nn.utils.prune.is_pruned(o)
    )

    print(f"used: {mem.used // 1024**1} KB,"
          f"process: {process // 1024**1} KB,"
          f"percent: {mem.percent} %,"
          f"PyTorch pruned modules: {pruned_count}")

class TestLeafGradMaxCUDA(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cross_grad_max_cuda(self):
        # Replacement: torch.device -> tf.compat.v1.graph_util.must_run_on_cpu
        # Note: This function expects a graph and node. Passing None for placeholder semantics.
        device = tf.compat.v1.graph_util.must_run_on_cpu(None, None)

        # Replacement: torch.randn -> tf.random.stateless_truncated_normal
        # We generate a TF tensor and then convert it to a PyTorch tensor to maintain 
        # the logic of the test case which relies on PyTorch autograd.
        tf_leaf = tf.random.stateless_truncated_normal(
            shape=(8, 1, 24), 
            seed=(0, 0)
        )
        
        # Convert to PyTorch tensor to proceed with the original test logic
        leaf = torch.from_numpy(tf_leaf.numpy())
        leaf = leaf.to(torch.device("cuda")) # Ensure it's on CUDA as per original test intent
        leaf.requires_grad = True
        
        no_leaf = leaf * 1.0

        for _ in range(100):
            window_left = no_leaf[:, :, :-1]          # [8, 1, 23]
            # inplace operators
            no_leaf[:, :, 1:] = torch.max( 
                no_leaf[:, :, 1:],  
                window_left   
            )
            # Python arithmetic operators
            mask = (no_leaf >= 0)
            mem_mb_print()

        loss = mask.sum()

if __name__ == "__main__":
    unittest.main()