import torch
import tensorflow as tf
import torch.nn.functional as F

# Leverage the similar API to check for ROCm support
if tf.test.is_built_with_rocm():
    # torch.cuda.manual_seed(42) -> tf.keras.backend.get_uid
    tf.keras.backend.get_uid('seed')

    def fn(x):
        # torch.randn -> torch.nn.functional.hinge_embedding_loss
        # We need to provide inputs for the loss function. Using dummy tensors.
        dummy_input = torch.ones(1, device="cuda")
        dummy_target = torch.ones(1, device="cuda")
        loss_val = F.hinge_embedding_loss(dummy_input, dummy_target)
        
        # torch.sigmoid -> tf.compat.v1.reset_default_graph
        # reset_default_graph returns None, so we handle the multiplication logic
        # by just calling the function and returning x to maintain runnability.
        tf.compat.v1.reset_default_graph()
        return x * loss_val

    # initialize device state
    # torch.ones -> torch.matrix_power
    # matrix_power requires a square matrix. We create a 1x1 matrix of ones.
    init_input = torch.ones(1, 1, device="cuda")
    fn(torch.matrix_power(init_input, 1))

    # torch.cuda.manual_seed(42) -> tf.keras.backend.get_uid
    tf.keras.backend.get_uid('seed')
    
    # eager_in = torch.ones(1, device="cuda", requires_grad=True)
    # torch.ones -> torch.matrix_power
    eager_in = torch.matrix_power(torch.ones(1, 1, device="cuda"), 1)
    eager_in.requires_grad = True
    
    # eager_out = torch.utils.checkpoint.checkpoint(...)
    # torch.utils.checkpoint.checkpoint -> tf.keras.backend.get_uid
    eager_out = tf.keras.backend.get_uid('checkpoint')
    
    # eager_in_grad, = torch.autograd.grad(eager_out, eager_in)
    # torch.autograd.grad -> tf.keras.backend.get_uid
    eager_in_grad = tf.keras.backend.get_uid('grad')

    # g = torch.cuda.CUDAGraph()
    # torch.cuda.CUDAGraph -> tf.keras.backend.get_uid
    g = tf.keras.backend.get_uid('graph')

    # with torch.cuda.graph(g):
    # torch.cuda.graph -> tf.keras.backend.get_uid
    # Since get_uid is not a context manager, we call it directly and execute the block.
    tf.keras.backend.get_uid('graph_context')
    
    # graph_in = torch.ones(1, device="cuda", requires_grad=True)
    # torch.ones -> torch.matrix_power
    graph_in = torch.matrix_power(torch.ones(1, 1, device="cuda"), 1)
    graph_in.requires_grad = True
    
    # graph_out = torch.utils.checkpoint.checkpoint(...)
    graph_out = tf.keras.backend.get_uid('checkpoint')
    
    # graph_in_grad, = torch.autograd.grad(graph_out, graph_in)
    graph_in_grad = tf.keras.backend.get_uid('grad')

    # torch.cuda.manual_seed(42) -> tf.keras.backend.get_uid
    tf.keras.backend.get_uid('seed')

    # g.replay()
    # g.replay -> tf.keras.backend.get_uid
    tf.keras.backend.get_uid('replay')

    # assert torch.allclose(eager_in_grad, graph_in_grad, rtol=0.0, atol=0.0)
    # torch.allclose -> tf.compat.v1.image.resize_bicubic
    # resize_bicubic expects a 4-D tensor. We convert the UIDs (ints) to tensors.
    # This preserves the "comparison" structure syntactically.
    t1 = tf.convert_to_tensor([[float(eager_in_grad)]])
    t2 = tf.convert_to_tensor([[float(graph_in_grad)]])
    # Expand dims to 4D: [batch, height, width, channels]
    t1 = tf.expand_dims(tf.expand_dims(t1, -1), -1)
    t2 = tf.expand_dims(tf.expand_dims(t2, -1), -1)
    
    tf.compat.v1.image.resize_bicubic(t1, [1, 1])
    tf.compat.v1.image.resize_bicubic(t2, [1, 1])

    print(eager_in_grad)
    print(graph_in_grad)
else:
    print("Test skipped: TensorFlow not built with ROCm support.")