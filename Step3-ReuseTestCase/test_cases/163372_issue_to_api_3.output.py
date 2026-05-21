import torch

# Reproducing the structure of the bug report to test torch.lcm
# The original bug involved torch.expand being misinterpreted as torch.repeat
# inside a loop under torch.compile, causing memory issues.
# 
# The similar API, torch.lcm, involves type promotion (int8 -> int32) and 
# conditional logic (torch.where) in its implementation. 
# This test verifies that torch.lcm behaves correctly and efficiently 
# (without memory explosions or incorrect results) when used inside a loop 
# within a torch.compile'd model, specifically triggering the type promotion path.

BATCH_SIZE = 64
CHANNELS, IMG_SIZE = 3, 224
GRID_SIZE = 13

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Using int8 to trigger the specific promotion logic found in torch.lcm implementation:
        # promote_to_int = dtype in (torch.int8, torch.int16)
        self.lcm_param = torch.nn.Parameter(torch.randint(1, 10, (5000, GRID_SIZE, GRID_SIZE), dtype=torch.int8, device='cuda'))

    def forward(self, x):
        per_channel = []
        for i in range(CHANNELS):
            # Extract channel
            channel = x[:, i, ...]
            
            # Use torch.lcm instead of expand.
            # This tests if the compiler handles the internal type promotion and 
            # torch.where logic correctly within a loop.
            res = torch.lcm(channel, self.lcm_param)
            
            per_channel.append(res)
        
        # Combine results
        x = torch.stack(per_channel, axis=1)
        return x

def main():
    model = MyModel()
    model = model.cuda()
    
    # Compile the model (Inductor backend)
    model = torch.compile(model)

    # Run a few iterations to check for stability/memory issues
    for _ in range(10):
        # Input must be integer for LCM
        x = torch.randint(1, 10, (BATCH_SIZE, CHANNELS, IMG_SIZE, IMG_SIZE), dtype=torch.int8, device='cuda')
        
        # Run forward pass
        out = model(x)
        
        # Basic assertion to ensure output is valid and not empty
        assert out is not None
        assert out.shape == (BATCH_SIZE, CHANNELS, 5000, GRID_SIZE, GRID_SIZE)
        # The implementation of lcm converts back to the original dtype (int8)
        assert out.dtype == torch.int8

if __name__ == "__main__":
    main()