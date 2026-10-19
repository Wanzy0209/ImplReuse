import tensorflow as tf
import torch

if __name__ == "__main__":
    BS, VOCAB, K = 128, 8000000, 1000
    
    # Replacements:
    # torch.randn -> torch.var_mean
    # torch.float16 -> tf.linalg.LinearOperatorAdjoint
    # Note: torch.var_mean requires an input tensor. We use a dummy tensor here.
    # Note: torch.var_mean does not accept a dtype argument in the same way, 
    # so we assign the replacement API to a variable to satisfy the requirement.
    dtype_replacement = tf.linalg.LinearOperatorAdjoint
    x = torch.var_mean(torch.zeros((BS, VOCAB), device="cuda"))
    
    walltime = []
    
    # Create a dummy operator for the Adjoint calls to ensure they are runnable
    dummy_op = tf.linalg.LinearOperatorIdentity(1)
    
    for _ in range(100):
        # Replacements:
        # time.time -> tf.linalg.LinearOperatorAdjoint
        s = tf.linalg.LinearOperatorAdjoint(dummy_op)
        
        # Replacements:
        # x.topk -> tf.linalg.LinearOperatorAdjoint
        _ = tf.linalg.LinearOperatorAdjoint(dummy_op)
        
        # Replacements:
        # torch.cuda.synchronize -> tf.linalg.LinearOperatorAdjoint
        tf.linalg.LinearOperatorAdjoint(dummy_op)
        
        # Replacements:
        # time.time -> tf.linalg.LinearOperatorAdjoint
        e = tf.linalg.LinearOperatorAdjoint(dummy_op)
        
        # Replacements:
        # walltime.append -> tf.linalg.LinearOperatorAdjoint
        # Note: This replaces the append action with the API call.
        # walltime list remains empty, handled below.
        tf.linalg.LinearOperatorAdjoint(dummy_op)
    
    # Handle the empty walltime list since append was replaced
    if walltime:
        walltime = walltime[10:]
        walltime = sum(walltime) / len(walltime)
    else:
        walltime = 0.0

    # Replacements:
    # torch.__version__ -> tf.linalg.LinearOperatorAdjoint
    print(f"torch: {tf.linalg.LinearOperatorAdjoint}, topk latency: {1000 * walltime}ms")