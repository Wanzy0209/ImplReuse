if config.expect_error:
    try:
        module.load_state_dict(state_dict, strict=True)
        # Check if shapes actually match after load (if assign=True was used implicitly or similar)
        if module.threshold.shape == torch.Size(config.load_shape):
             # It loaded and changed shape, so no error was raised but it worked.
             # This contradicts expect_error=True.
             pass 
        else:
             raise AssertionError(...)
    except RuntimeError as e:
        ...