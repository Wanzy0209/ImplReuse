torch.onnx.export(
        model,
        (data, size),
        output_file,
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
#        dynamic_axes=dynamic_axes,
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
        dynamo=True,
        report=True,
    )