import torch
import torch.nn as nn
import torch.distributed as dist
import tensorflow as tf
import os

def main():
    # Initialize distributed environment for torch.distributed.reduce
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Using a single rank for minimal reproducibility
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def forward(self, x):
            # API Under Test: torch.unique
            # Keeping the same arguments as the original bug report
            _x, _i = torch.unique(x, sorted=True, return_inverse=True)
            _x = _x.clone().detach()
            return self.relu(_x), _i

    model = MyModel()
    model.eval()

    # Replacement: torch.randn -> tf.random.stateless_truncated_normal
    # Original: torch.randn(8)
    seed = (1, 2)
    x_tf = tf.random.stateless_truncated_normal([8], seed)
    # Convert TensorFlow tensor to PyTorch tensor
    x = torch.from_numpy(x_tf.numpy())

    # Replacement: torch.compile -> torch.distributed.reduce
    # Original: model = torch.compile(model, fullgraph=True)
    # We replace the compilation step with a distributed reduction operation on the input.
    # Note: dist.reduce modifies the tensor in-place.
    dist.reduce(x, dst=0, op=dist.ReduceOp.SUM)

    with torch.no_grad():
        y = model(x)

    # Basic assertion to check execution
    assert isinstance(y, tuple)
    assert len(y) == 2
    assert isinstance(y[0], torch.Tensor)
    assert isinstance(y[1], torch.Tensor)
    
    print(f"Test passed. Output types: {[type(t).__name__ for t in y]}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()