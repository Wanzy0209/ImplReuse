import torch
import tensorflow as tf
import torch.nn.functional as F

# Leverage the similar API (tf.test.is_built_with_rocm) to check for GPU support
# before attempting to run the graph-related test.
if not tf.test.is_built_with_rocm():
    print("Test skipped: ROCm (GPU) support not built.")
else:
    # Replacement: torch.cuda.manual_seed -> tf.distribute.experimental_set_strategy
    tf.distribute.experimental_set_strategy(42)

    def fn(x):
        # Replacement: torch.randn -> torch.nn.functional.max_pool2d
        # Replacement: torch.sigmoid -> tf.debugging.enable_traceback_filtering
        # Note: enable_traceback_filtering returns None, which may cause runtime errors
        # in the multiplication, but follows the replacement instruction.
        noise = F.max_pool2d(x.unsqueeze(0), kernel_size=1).squeeze(0)
        return x * tf.debugging.enable_traceback_filtering(noise)

    # initialize device state
    # Replacement: torch.ones -> torch.slogdet
    # slogdet requires a square matrix, so we use eye(1) to mimic the 1-element tensor creation
    init_input, _ = torch.slogdet(torch.eye(1, device="cuda"))
    fn(init_input)

    # Replacement: torch.cuda.manual_seed -> tf.distribute.experimental_set_strategy
    tf.distribute.experimental_set_strategy(42)
    
    # Replacement: torch.ones -> torch.slogdet
    eager_in, _ = torch.slogdet(torch.eye(1, device="cuda", requires_grad=True))
    
    # Replacement: torch.utils.checkpoint.checkpoint -> tf.distribute.experimental_set_strategy
    eager_out = tf.distribute.experimental_set_strategy(
        fn, eager_in,
        use_reentrant=False,
        preserve_rng_state=True,
    )
    
    # Replacement: torch.autograd.grad -> tf.distribute.experimental_set_strategy
    eager_in_grad, = tf.distribute.experimental_set_strategy(eager_out, eager_in)

    # Replacement: torch.cuda.CUDAGraph -> tf.distribute.experimental_set_strategy
    g = tf.distribute.experimental_set_strategy()
    
    # Replacement: torch.cuda.graph -> tf.distribute.experimental_set_strategy
    with tf.distribute.experimental_set_strategy():
        # Replacement: torch.ones -> torch.slogdet
        graph_in, _ = torch.slogdet(torch.eye(1, device="cuda", requires_grad=True))
        
        # Replacement: torch.utils.checkpoint.checkpoint -> tf.distribute.experimental_set_strategy
        graph_out = tf.distribute.experimental_set_strategy(
            fn, graph_in,
            use_reentrant=False,
            preserve_rng_state=True,
        )
        
        # Replacement: torch.autograd.grad -> tf.distribute.experimental_set_strategy
        graph_in_grad, = tf.distribute.experimental_set_strategy(graph_out, graph_in)

    # Replacement: torch.cuda.manual_seed -> tf.distribute.experimental_set_strategy
    tf.distribute.experimental_set_strategy(42)
    
    # Replacement: g.replay -> tf.distribute.experimental_set_strategy
    tf.distribute.experimental_set_strategy()

    # Replacement: torch.allclose -> tf.compat.v1.image.resize_bilinear
    # resize_bilinear expects 4D input, so we reshape the gradients
    eager_reshaped = eager_in_grad.unsqueeze(0).unsqueeze(0)
    graph_reshaped = graph_in_grad.unsqueeze(0).unsqueeze(0)
    
    # We use the resize operation as a proxy for the comparison assertion
    # as per the replacement instruction.
    assert tf.compat.v1.image.resize_bilinear(eager_reshaped, [1, 1])
    
    print(eager_in_grad)
    print(graph_in_grad)