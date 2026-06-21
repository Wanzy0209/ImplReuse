import torch

# Configuration from the original bug report
# Fix: Check if _dynamo exists before accessing it to handle different PyTorch versions
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(751735337)

def get_inputs():
    """Generates inputs matching the shapes and types from the original fuzzer output."""
    return (
        torch.randint(0, 10, (15, 108, 4), dtype=torch.int16, device='cuda'),
        torch.randint(0, 10, (11,), dtype=torch.int64, device='cuda'),
        torch.randint(0, 10, (3, 27), dtype=torch.int16, device='cuda'),
        torch.randint(0, 10, (1, 27), dtype=torch.int16, device='cuda'),
        torch.randint(0, 10, (1, 27), dtype=torch.int16, device='cuda'),
        torch.randint(0, 10, (1, 27), dtype=torch.int16, device='cuda'),
        torch.randint(0, 10, (1,), dtype=torch.int64, device='cuda'),
        None
    )

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, sentinel):
    # Setup from the original test case
    var_node_4 = arg_0
    var_node_3 = torch.chunk(var_node_4, 4, dim=1)[0]
    var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0]
    var_node_1 = torch.squeeze(var_node_2)
    
    # Note: Added device='cuda' to torch.full calls to match the original context
    var_node_8 = torch.full((13, 27), 3, dtype=torch.int16, device='cuda')
    var_node_9 = arg_1
    _input_size_var_node_7 = var_node_8.size(0)
    _index_var_node_7 = torch.randint(0, _input_size_var_node_7, (11,), device=var_node_8.device)
    var_node_7 = torch.index_select(var_node_8, 0, _index_var_node_7)
    var_node_6 = torch.clamp(var_node_7, min=-1.0, max=1.0)
    
    var_node_12 = arg_2
    var_node_11 = torch.clamp(var_node_12, min=-1.0, max=1.0)
    var_node_13 = torch.full((1,), 3, dtype=torch.int64, device='cuda')
    _input_size_var_node_10 = var_node_11.size(0)
    _index_var_node_10 = torch.randint(0, _input_size_var_node_10, (1,), device=var_node_11.device)
    var_node_10 = torch.index_select(var_node_11, 0, _index_var_node_10)
    
    var_node_16 = arg_3
    var_node_17 = arg_4
    var_node_18 = arg_5
    var_node_15 = torch.cat([var_node_16, var_node_17, var_node_18], dim=0)
    var_node_20 = arg_6
    var_node_19 = torch.clamp(var_node_20, min=None, max=1.0)
    _input_size_var_node_14 = var_node_15.size(0)
    _index_var_node_14 = torch.randint(0, _input_size_var_node_14, (1,), device=var_node_15.device)
    var_node_14 = torch.index_select(var_node_15, 0, _index_var_node_14)
    
    var_node_22 = torch.full((4, 27), 3, dtype=torch.int16, device='cuda')
    var_node_24 = torch.full((4,), 3, dtype=torch.int64, device='cuda')
    var_node_25 = torch.full((2,), 3, dtype=torch.int64, device='cuda')
    _input_size_var_node_23 = var_node_24.size(0)

    # Adaptation: Replace torch.gather with torch.ceil
    # We apply torch.ceil to var_node_6 (int16). 
    # For integer types, torch.ceil is expected to return a clone.
    return torch.ceil(var_node_6)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
    else:
        args = get_inputs()
        
        # Run in eager mode
        expected = fuzzed_program(*args)
        
        # Run in compiled mode
        compiled_fn = torch.compile(fuzzed_program)
        result = compiled_fn(*args)
        
        # Verify results match
        assert torch.equal(expected, result), f"Eager and compiled results diverged.\nEager: {expected}\nCompiled: {result}"
        print("Test passed: torch.ceil behaves consistently in eager and compiled modes.")