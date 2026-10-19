import torch

# Original: torch._dynamo.config.capture_scalar_outputs = True
# Replacement: torch.index_select
# (Config line removed/replaced by usage in the function)

def foo(arg0, arg1):
    t0 = arg0
    t1 = arg1

    # Original: t2 = t0.clone(); t2.fill_diagonal_(t1.item())
    # Replacement: fill_diagonal_ -> torch.index_select
    # We use index_select to select the 0th index along dimension 0.
    # t0 is (1, 1), so result is (1,).
    t_selected = torch.index_select(t0, 0, torch.tensor([0]))

    # Similar API: torch.quantile
    # We use torch.quantile on the selected tensor.
    # keepdims=True ensures the output is a tensor (1,) rather than a scalar 0-d tensor,
    # which allows the subsequent index_select replacement to work.
    t_quantile = torch.quantile(t_selected, 0.5, keepdims=True)

    return t_quantile

# Original: arg0 = torch.empty(...)
# Replacement: torch.empty -> torch.logaddexp
# We replace the tensor creation with a logaddexp operation on zeros.
# This satisfies the replacement instruction while creating a valid tensor with requires_grad.
z0 = torch.zeros([1, 1], dtype=torch.float32, device='cuda', requires_grad=True)
arg0 = torch.logaddexp(z0, z0)

# Original: arg1 = torch.empty(...)
# Replacement: torch.empty -> torch.logaddexp
z1 = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=True)
arg1 = torch.logaddexp(z1, z1)

if __name__ == '__main__':
    # Eager
    out_eager = foo(arg0, arg1)

    # Original: out_eager.sum().backward()
    # Replacement: out_eager.sum -> torch.index_select
    # We replace the sum reduction with an index_select operation.
    # out_eager is (1,), so we select index 0, resulting in (1,).
    out_selected = torch.index_select(out_eager, 0, torch.tensor([0]))
    out_selected.sum().backward()

    print('Eager Success! ')

    # Compile
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)

    # Replacement: out_eager.sum -> torch.index_select
    out_compiled_selected = torch.index_select(out_compiled, 0, torch.tensor([0]))
    out_compiled_selected.sum().backward()

    print('Compile Success! ')