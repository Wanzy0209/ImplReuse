import torch
import tensorflow as tf
import unittest
import gc
import psutil
import torch.nn.functional as F

def mem_mb_print():
    gc.collect()
    mem = psutil.virtual_memory()
    process = psutil.Process().memory_info().rss

    # Replacement: torch.is_tensor -> torch.atleast_3d
    # We use isinstance to ensure the object is a tensor before passing to atleast_3d
    # to maintain runnability, as atleast_3d expects a tensor-like input.
    count = sum(1 for o in gc.get_objects() if isinstance(o, torch.Tensor) and torch.atleast_3d(o))
    
    print(f"used: {mem.used // 1024**1} KB,"
          f"process: {process // 1024**1} KB,"
          f"percent: {mem.percent} %,"
          f"PyTorch tensors: {count}")

class TestLeafGradMaxCUDA(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cross_grad_max_cuda(self):
        # Replacement: torch.device -> tf.experimental.dtensor.create_mesh
        # We create a dummy mesh. Note: This is a TensorFlow object, not directly used by PyTorch.
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            mesh_device = physical_devices[0].name if physical_devices else "/cpu:0"
            device = tf.experimental.dtensor.create_mesh([("x", 1)], device=mesh_device)
        except Exception:
            # Fallback if TF setup fails or is not configured
            device = None

        # Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool3d
        # We need an input tensor for the pooling operation. We create a dummy input 
        # using torch.ones (not in replacement list) to feed into adaptive_max_pool3d.
        # We use "cuda" string directly for PyTorch device, as 'device' var is now a TF Mesh.
        dummy_input = torch.ones(8, 1, 24, device="cuda", requires_grad=True)
        leaf = F.adaptive_max_pool3d(dummy_input, output_size=(8, 1, 24))

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