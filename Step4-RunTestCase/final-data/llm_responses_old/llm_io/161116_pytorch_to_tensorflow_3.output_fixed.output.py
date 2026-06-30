import tensorflow as tf

def main():
    # In PyTorch, dist.init_process_group with backend='nccl' initializes the communication.
    # In TensorFlow, tf.distribute.MirroredStrategy uses NCCL for cross-device operations
    # on GPUs by default. Initializing this strategy on a system with many GPUs
    # (e.g., >40 GPUs on an NVL72) triggers similar NCCL topology discovery logic.
    
    print("Initializing MirroredStrategy...")
    strategy = tf.distribute.MirroredStrategy()
    
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # In PyTorch, dist.barrier() is called to synchronize processes.
    # In TensorFlow, we perform a computation within the strategy scope
    # to ensure the communication layer is active and synchronized.
    with strategy.scope():
        # Create a variable to trigger placement on all replicas
        v = tf.Variable(1.0)
        
        # Define a simple step to verify communication
        @tf.function
        def step():
            return v.read_value()

        # Run the step to ensure NCCL communication is established.
        # Note: In older versions of TensorFlow 2.x (e.g., 2.0/2.1), 
        # MirroredStrategyV1 might not have the 'run' method directly exposed 
        # or named as such. 'experimental_run_v2' is the compatible method for those versions.
        if hasattr(strategy, 'run'):
            result = strategy.run(step)
        elif hasattr(strategy, 'experimental_run_v2'):
            result = strategy.experimental_run_v2(step)
        else:
            # Fallback for environments where distribution methods might differ significantly
            result = step()
            
        print("Step executed successfully.")

if __name__ == "__main__":
    main()