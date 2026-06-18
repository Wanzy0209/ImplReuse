import torch
import torch.nn as nn
import torch.optim as optim

def test_reduce_lr_on_plateau_preserves_tensor_type():
    """
    Test that ReduceLROnPlateau preserves the tensor type of the learning rate.
    
    This test addresses the bug where _reduce_lr sets param_group["lr"] to a float,
    triggering recompilation if the optimizer uses a tensor LR.
    """
    # Setup model and optimizer with a tensor learning rate
    model = nn.Linear(1, 1)
    lr_tensor = torch.tensor(0.1)
    opt = optim.Adam(model.parameters(), lr=lr_tensor)
    
    # Verify initial LR is a tensor
    assert isinstance(opt.param_groups[0]['lr'], torch.Tensor), "Initial LR must be a tensor"

    # Setup scheduler
    # patience=0 ensures reduction happens immediately if metric doesn't improve
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=0, min_lr=0.001)

    # Step 1: metric = 1.0
    scheduler.step(1.0)
    
    # Step 2: metric = 0.5 (improvement)
    scheduler.step(0.5)
    
    # Step 3: metric = 1.0 (worsening, triggers reduction)
    # Expected LR: 0.1 * 0.5 = 0.05
    scheduler.step(1.0)

    # Check the type of the learning rate after reduction
    current_lr = opt.param_groups[0]['lr']
    
    # The bug causes current_lr to become a float.
    # The fix should ensure it remains a tensor to avoid recompilation.
    assert isinstance(current_lr, torch.Tensor), \
        f"LR type changed from Tensor to {type(current_lr)}, which triggers recompilation."
    
    # Check the value is correct
    assert torch.allclose(current_lr, torch.tensor(0.05)), \
        f"LR value is incorrect. Expected 0.05, got {current_lr}"

if __name__ == "__main__":
    test_reduce_lr_on_plateau_preserves_tensor_type()
    print("Test passed.")