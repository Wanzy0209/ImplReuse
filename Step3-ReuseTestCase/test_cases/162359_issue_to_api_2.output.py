import torch
from torch import optim

def test_sequential_lr_tensor_aliasing():
    """
    Test case for Issue 162359: SequentialLR aliases lr Tensor with initial_lr.
    
    This test verifies that when using a Tensor learning rate (similar to how 
    tf.lookup.KeyValueTensorInitializer accepts tensor inputs), the SequentialLR 
    scheduler does not corrupt the base_lrs of chained schedulers by aliasing 
    the optimizer's lr with initial_lr.
    """
    # 1. Use a tensor learning rate. 
    # This mirrors the pattern of initializing structures with tensors 
    # (e.g., tf.lookup.KeyValueTensorInitializer).
    lr_val = 1.0
    x = torch.tensor(0.0, device='meta')
    lr_tensor = torch.tensor(lr_val)
    opt = optim.AdamW([x], lr=lr_tensor)

    # 2. Initialize our chained schedulers.
    milestone = 10
    start_factor = 0.5
    warmup = optim.lr_scheduler.LinearLR(opt, start_factor=start_factor, total_iters=milestone)
    decay = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

    # Capture the original base_lr for the decay scheduler
    expected_decay_base_lr = decay.base_lrs[0].clone()

    # 3. Initialize our SequentialLR.
    # In the bug, SequentialLR.__init__ aliases group["lr"] with group["initial_lr"].
    # The subsequent _initial_step() call to LinearLR modifies the lr in-place,
    # which mutates initial_lr and the base_lrs of the 'decay' scheduler.
    scheduler = optim.lr_scheduler.SequentialLR(
        opt, 
        schedulers=[warmup, decay], 
        milestones=[milestone]
    )

    # 4. Verify that the base_lrs of the chained scheduler are not corrupted.
    # The bug would cause decay.base_lrs[0] to be scaled by start_factor (0.5).
    # The fix ensures decay.base_lrs[0] remains the original lr_val (1.0).
    
    # Check that the optimizer's initial_lr was NOT mutated by the warmup step
    assert opt.param_groups[0]['initial_lr'] == lr_val, (
        f"Optimizer initial_lr was corrupted. Expected {lr_val}, "
        f"got {opt.param_groups[0]['initial_lr']}"
    )

    # Check that the chained scheduler's base_lr was NOT mutated
    assert decay.base_lrs[0] == expected_decay_base_lr, (
        f"Chained scheduler base_lr was corrupted. Expected {expected_decay_base_lr}, "
        f"got {decay.base_lrs[0]}"
    )
    
    # Verify that the current LR is correctly set by the warmup scheduler
    assert opt.param_groups[0]['lr'] == start_factor * lr_val, (
        f"Current LR is incorrect. Expected {start_factor * lr_val}, "
        f"got {opt.param_groups[0]['lr']}"
    )

    print("Test passed: SequentialLR correctly handles Tensor learning rates without aliasing corruption.")

if __name__ == "__main__":
    test_sequential_lr_tensor_aliasing()