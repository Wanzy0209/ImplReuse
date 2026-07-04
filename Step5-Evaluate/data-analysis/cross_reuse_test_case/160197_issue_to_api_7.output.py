import torch
import warnings
import pytest

def test_lbfgs_closure_scalar_conversion_warning():
    """
    Test that reproduces the UserWarning raised by LBFGS when converting
    a tensor with requires_grad=True to a scalar internally.
    
    This test is based on Issue #160197. The LBFGS optimizer internally calls
    float(closure()), which triggers a warning in PyTorch 2.8.0+ if the
    returned loss tensor still requires gradients.
    """
    # Setup: Create random tensors and optimizer
    a, b = torch.rand((2, 32, 32))
    a.requires_grad_()
    optimizer = torch.optim.LBFGS([a])
    loss_fn = lambda x, y: (x - y).pow(2).mean()

    # Define the closure required by LBFGS
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(a, b)
        loss.backward()
        return loss

    # Assertion: Verify that the specific UserWarning is raised during the step
    # The warning message matches the behavior described in the bug report.
    with pytest.warns(UserWarning, match="converting a tensor with requires_grad=True to a scalar"):
        optimizer.step(closure)

    # Verify that optimization actually occurred (loss decreased)
    final_loss = loss_fn(a, b)
    # Note: Since we stop after one step (or the first warning), we just check it ran.
    # In a full test, we might check convergence, but here we focus on the warning.
    assert isinstance(final_loss, torch.Tensor)