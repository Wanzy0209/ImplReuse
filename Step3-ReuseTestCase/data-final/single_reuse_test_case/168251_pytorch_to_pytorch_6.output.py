import torch
import torch.nn as nn

class LobpcgModel(nn.Module):
    """
    A simple wrapper model for torch.lobpcg to mimic the structure of the VAE model.
    torch.lobpcg returns a tuple of (eigenvalues, eigenvectors).
    """
    def __init__(self):
        super(LobpcgModel, self).__init__()

    def forward(self, A):
        # torch.lobpcg returns a tuple (eigenvalues, eigenvectors)
        return torch.lobpcg(A, k=2)

def get_default_model():
    return LobpcgModel()

def get_sample_inputs():
    # Create a symmetric positive definite matrix
    n = 10
    A = torch.randn(n, n)
    A = A @ A.T + torch.eye(n) * 0.1
    return (A,)

def main():
    model = get_default_model()
    model.eval()
    inputs = get_sample_inputs()
    
    # Eager execution
    with torch.no_grad():
        output_eager = model(*inputs)
    
    print('Model executed successfully in eager mode!')
    print(f'Input shape: {inputs[0].shape}')
    
    # The original bug report highlights accessing .shape on the tuple output.
    # We verify this behavior here.
    try:
        print(f'Eager output shape: {output_eager.shape}')
    except AttributeError as e:
        print(f'Eager AttributeError: {e}')
        print(f'Actual output type: {type(output_eager)}')
        if isinstance(output_eager, tuple):
            print(f'Element 0 shape: {output_eager[0].shape}')
            print(f'Element 1 shape: {output_eager[1].shape}')

    # Compile execution
    # We check if torch.compile handles the tuple return of torch.lobpcg correctly
    # or if it introduces the issue described in the bug report.
    compiled_model = torch.compile(model)
    with torch.no_grad():
        output_compile = compiled_model(*inputs)
        
    print(f'Compile output type: {type(output_compile)}')
    
    # Attempt to access shape on the compiled output
    try:
        print(f'Compile shape: {output_compile.shape}')
    except AttributeError as e:
        print(f'Compile AttributeError: {e}')

if __name__ == '__main__':
    main()