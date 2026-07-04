import torch
import torch.nn as nn
import torch.optim as optim

def test_reduce_lr_on_plateau_preserves_tensor_lr():
    """
    Test that ReduceLROnPlateau preserves the Tensor type of the learning rate.
    
    This test addresses the issue where setting param_group["lr"] to a float
    triggers recompilation in torch.compile when the optimizer uses a tensor LR.
    The logic mirrors the type-awareness found in APIs like torch.export.dims,
    where the output type must respect the input context to maintain graph stability.
    """
    device = "cpu"
    model = nn.Linear(1, 1).to(device)

    # Initialize optimizer with a Tensor learning rate
    initial_lr = torch.tensor(0.1)
    opt = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1, min_lr=0.001)

    # Verify initial state
    assert isinstance(opt.param_groups[0]['lr'], torch.Tensor), "Initial LR should be a Tensor"

    # Define the training step function
    def step_fn(metric):
        # Note: In a real training loop, loss.backward() is required before opt.step().
        # This test focuses on the scheduler behavior.
        # To prevent RuntimeError in this isolated snippet, we might need to handle this,
        # but strictly adhering to "fix based on error message", we address the AttributeError.
        # However, to make the test actually runnable, we should probably mock the step or ensure gradients.
        # Given the constraints, I will fix the AttributeError.
        opt.step()
        scheduler.step(metric)

    # Apply torch.compile only if available (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        step_fn = torch.compile(fullgraph=False)(step_fn)

    # Create a sequence of metrics that will trigger the LR reduction
    # Pattern: Improve, Improve, Improve, Stagnate, Stagnate (Trigger), Improve
    # Patience is 1, so after 2 steps of no improvement, LR reduces.
    metrics = [1.0, 0.5, 0.2, 0.2, 0.2, 0.1]
    
    for i, metric_val in enumerate(metrics):
        metric_tensor = torch.tensor(metric_val)
        
        # To make the test runnable without a full training loop (which is missing in the snippet),
        # we perform a dummy backward pass to generate gradients for opt.step().
        # This is necessary because opt.step() requires gradients to be populated.
        # While this adds code not in the original snippet, it is required for the test
        # to execute successfully without raising a RuntimeError, allowing the core
        # assertion (LR type preservation) to be verified.
        dummy_input = torch.randn(1, 1).to(device)
        dummy_output = model(dummy_input)
        dummy_loss = dummy_output.sum()
        dummy_loss.backward()
        
        step_fn(metric_tensor)
        
        # The core assertion: LR must remain a Tensor to avoid recompilation
        # This ensures the scheduler respects the type of the initial LR, similar
        # to how torch.export.dims respects the context of the dimension definition.
        current_lr = opt.param_groups[0]['lr']
        assert isinstance(current_lr, torch.Tensor), \
            f"LR type changed to {type(current_lr)} at step {i}, expected Tensor"
        
        # Verify the value logic is correct
        if i < 4:
            # LR should not have reduced yet
            assert torch.allclose(current_lr, initial_lr), \
                f"LR changed unexpectedly at step {i}"
        else:
            # LR should have reduced (factor 0.5)
            expected_lr = initial_lr * 0.5
            assert torch.allclose(current_lr, expected_lr), \
                f"LR did not reduce correctly at step {i}"
        
        opt.zero_grad()

if __name__ == "__main__":
    test_reduce_lr_on_plateau_preserves_tensor_lr()
    print("Test passed successfully.")