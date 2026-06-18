import torch
from torch import optim
import math

def test_sequentiallr_tensor_aliasing_bug():
    """
    Reproduces the bug where SequentialLR aliases the lr Tensor with initial_lr,
    corrupting base_lrs for chained schedulers when using a Tensor learning rate.
    """
    # 1. Use a tensor learning rate.
    lr = 1.0
    x = torch.tensor(0.0, device='meta')
    opt = optim.AdamW([x], lr=torch.tensor(lr))

    # 2. Initialize our chained schedulers.
    milestone, total_steps = 40, 100
    start_factor, end_factor = 0.2, 1.0
    warmup = optim.lr_scheduler.LinearLR(opt, start_factor, end_factor, total_iters=milestone)
    decay = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - milestone)

    # Note that each scheduler's base_lr aliases with the optimizer's initial_lr.
    assert warmup.base_lrs[0].data_ptr() == opt.param_groups[0]['initial_lr'].data_ptr()
    assert decay.base_lrs[0].data_ptr() == opt.param_groups[0]['initial_lr'].data_ptr()

    # 3. Initialize our SequentialLR.
    scheduler = optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, decay], milestones=[milestone])

    # SequentialLR.__init__ aliases our optimizer's lr and initial_lr.
    # for group in self.optimizer.param_groups:
    #     group["lr"] = group["initial_lr"]
    assert opt.param_groups[0]['lr'].data_ptr() == opt.param_groups[0]['initial_lr'].data_ptr()

    # Which means they're also aliased with the base_lrs of each scheduler!
    assert (warmup.base_lrs[0].data_ptr() 
            == decay.base_lrs[0].data_ptr()
            == opt.param_groups[0]['initial_lr'].data_ptr() 
            == opt.param_groups[0]['lr'].data_ptr())

    # It also calls _initial_step() on our LinearLR, setting all tensors to start_factor * lr.
    # This is the corruption point: the in-place update affects all aliased tensors.
    assert (start_factor * lr
            == opt.param_groups[0]['lr'].item()
            == opt.param_groups[0]['initial_lr'].item()
            == warmup.base_lrs[0].item()
            == decay.base_lrs[0].item())

    # 4. Verify the impact on the chained scheduler (CosineAnnealingLR).
    # Step through the warmup phase
    for _ in range(milestone):
        scheduler.step()

    # Step into the decay phase
    scheduler.step()
    
    current_lr = scheduler.get_last_lr()[0].item()
    
    # Calculate what the LR would be if base_lrs were corrupted (buggy behavior)
    # The base_lr for decay is now start_factor * lr (0.2) instead of lr (1.0)
    t = 1
    T_max = total_steps - milestone
    corrupted_base_lr = start_factor * lr
    expected_buggy_lr = corrupted_base_lr * (1 + math.cos(math.pi * t / T_max)) / 2
    
    # Assert that the bug is present: the LR is scaled by start_factor incorrectly
    assert abs(current_lr - expected_buggy_lr) < 1e-6, \
        f"Expected buggy LR {expected_buggy_lr}, got {current_lr}"

    print("Bug reproduced successfully: base_lrs were corrupted by aliasing.")

if __name__ == "__main__":
    test_sequentiallr_tensor_aliasing_bug()