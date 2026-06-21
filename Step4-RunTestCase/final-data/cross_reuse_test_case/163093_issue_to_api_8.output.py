import torch
import torch.nn as nn
import torch.optim as optim

def get_param_group_lrs(optimizer):
    """
    Helper to gather learning rates from param groups.
    This mirrors the pattern in tf.compat.v1.report_uninitialized_variables
    which gathers a list of variables to check their state.
    """
    return [group['lr'] for group in optimizer.param_groups]

def test_reduce_lr_on_plateau_preserves_tensor_lr():
    """
    Test that ReduceLROnPlateau preserves the tensor type of the learning rate
    to avoid triggering recompilation in torch.compile.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Linear(1, 1).to(device)

    # Initialize optimizer with a tensor LR
    initial_lr = torch.tensor(0.1, device=device)
    opt = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1, min_lr=0.001)

    # Verify initial type using the helper
    lrs = get_param_group_lrs(opt)
    assert isinstance(lrs[0], torch.Tensor), "Initial LR must be a tensor"

    # Define the compiled step function
    # We wrap the step logic to simulate the user's use case
    
    # Handle environments where torch.compile is not available (PyTorch < 2.0)
    if hasattr(torch, 'compile'):
        compile_decorator = torch.compile(fullgraph=False)
    else:
        # Identity decorator for older PyTorch versions
        def compile_decorator(func):
            return func

    @compile_decorator
    def step_fn(metric):
        opt.step()
        scheduler.step(metric)

    # Create metrics that will trigger the scheduler reduction
    # Patience=1. 
    # 1.0 -> 0.9 (improve) -> 0.8 (improve) -> 0.8 (wait) -> 0.8 (reduce)
    total_steps = 5
    metrics = torch.linspace(1.0, 0.8, total_steps, device=device)
    # Ensure the last few are equal to trigger patience logic
    metrics[-2:] = metrics[-1]

    for metric in metrics:
        step_fn(metric)

    # Check the LR after reduction
    final_lrs = get_param_group_lrs(opt)
    final_lr = final_lrs[0]

    # The bug causes 'lr' to become a float. We assert it remains a Tensor.
    assert isinstance(final_lr, torch.Tensor), \
        f"LR type changed to {type(final_lr)}. Expected Tensor to prevent recompilation."

    # Verify the value is correct (0.1 * 0.5 = 0.05)
    expected_lr = torch.tensor(0.05, device=device)
    assert torch.allclose(final_lr, expected_lr), \
        f"LR value {final_lr} does not match expected {expected_lr}"

if __name__ == "__main__":
    test_reduce_lr_on_plateau_preserves_tensor_lr()