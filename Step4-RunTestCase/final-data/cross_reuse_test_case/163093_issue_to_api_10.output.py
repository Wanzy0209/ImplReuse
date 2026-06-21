import torch
from torch import optim, nn

def test_reduce_lr_on_plateau_preserves_tensor_type():
    """
    Test case for Issue 163093.
    Verifies that ReduceLROnPlateau preserves the tensor type of the learning rate
    to avoid triggering recompilation when using torch.compile.
    Leverages torch.Tensor.view to create the initial learning rate tensor.
    """
    device = "cpu"
    model = nn.Linear(1, 1).to(device)

    # Leverage similar API: torch.Tensor.view
    # Create the learning rate as a tensor using view to ensure it is a tensor object.
    # This relates to the bug where the type of 'lr' changes from Tensor to float.
    lr_tensor = torch.tensor([0.1]).view(-1)

    opt = optim.Adam(model.parameters(), lr=lr_tensor)
    # patience=0 ensures reduction happens on the first non-improving step
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=0, min_lr=0.001)

    # Define the function logic
    def fn(metric):
        opt.step()
        scheduler.step(metric)

    # Apply torch.compile if available (PyTorch 2.0+)
    # If not available, run the function as is to verify the core logic (LR type preservation)
    if hasattr(torch, 'compile'):
        fn = torch.compile(fullgraph=False)(fn)

    # Step 1: metric improves (or stays same initially)
    fn(1.0)
    
    # Step 2: metric does not improve, triggers reduction
    # Bug: param_group["lr"] becomes float, triggering recompilation.
    fn(1.0)

    # Assertion: Check that the learning rate remains a tensor
    current_lr = opt.param_groups[0]['lr']
    assert isinstance(current_lr, torch.Tensor), (
        f"Expected lr to be Tensor, but got {type(current_lr)}. "
        "This indicates the bug where ReduceLROnPlateau converts tensor LR to float."
    )

    # Assertion: Check the value is correct
    expected_lr = 0.05
    assert torch.isclose(current_lr, torch.tensor(expected_lr)), (
        f"Expected lr value {expected_lr}, but got {current_lr}"
    )

if __name__ == "__main__":
    test_reduce_lr_on_plateau_preserves_tensor_type()