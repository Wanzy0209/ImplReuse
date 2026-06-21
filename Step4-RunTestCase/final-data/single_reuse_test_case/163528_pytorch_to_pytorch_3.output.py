import torch
from torch import nn

class Foo(nn.Module):
    def __init__(
        self,
        rng_state: torch.Tensor,
    ) -> None:
        super().__init__()
        self.rng_state = rng_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapted call site: torch.set_rng_state / torch.cuda.set_rng_state
        # We check the device of the state to determine which API to use
        if self.rng_state.is_cuda:
            torch.cuda.set_rng_state(self.rng_state)
        else:
            torch.set_rng_state(self.rng_state)
        
        # Return a random tensor to verify that the RNG state was set correctly
        return torch.randn_like(x)
    

def test_device(device, x, rng_state):
    x = x.to(device)
    rng_state = rng_state.to(device)
    foo = Foo(rng_state).to(device)
    
    try:
        foo_compiled = torch.compile(Foo(rng_state).to(device), fullgraph=True)
    except Exception as e:
        print(f"device: {device}, Compilation failed: {e}")
        return

    # warm up
    with torch.no_grad():
        y_original = foo(x)
        y_compiled = foo_compiled(x)


    # proper inference
    with torch.no_grad():
        y_original = foo(x)
        y_compiled = foo_compiled(x)

    print(f'device: {device}, diff: {torch.max(torch.abs(y_original - y_compiled)).item()}')
    print('original', y_original[:5, :5])
    print('compiled', y_compiled[:5, :5])
    

def main():
    batch_size = 32
    feature_dim = 10
    
    # Setup dummy input
    torch.manual_seed(42)
    x = torch.randn(batch_size, feature_dim, dtype=torch.float32)

    # Setup a specific RNG state to be set inside the model
    torch.manual_seed(123)
    _ = torch.randn(10) # Advance the state
    cpu_state = torch.get_rng_state()

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        _ = torch.randn(10, device='cuda')
        cuda_state = torch.cuda.get_rng_state()
    else:
        cuda_state = None

    test_device('cpu', x, cpu_state)
    
    if cuda_state is not None:
        test_device('cuda', x, cuda_state)

if __name__ == '__main__':
    main()