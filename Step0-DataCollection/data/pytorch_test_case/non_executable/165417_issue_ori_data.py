torch._dynamo.exc.Unsupported: Dynamic shape operator
Explanation: Operator `aten._unique2.default`'s output shape depends on input Tensor data.
Hint: Enable tracing of dynamic shape operators with `torch._dynamo.config.capture_dynamic_output_shape_ops = True`