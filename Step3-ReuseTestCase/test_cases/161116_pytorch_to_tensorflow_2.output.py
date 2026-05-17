import torch
import tensorflow as tf
import os

def main():
    # In TensorFlow, MirroredStrategy handles the initialization of the 
    # distributed environment (typically using NCCL for GPUs) automatically.
    # Unlike PyTorch, we don't manually set the device based on LOCAL_RANK 
    # for each process; MirroredStrategy manages the visible devices.
    
    # Initialize the strategy. This step corresponds to torch.distributed.init_process_group.
    # It will attempt to initialize NCCL across all available GPUs.
    # On an NVL72 with >40 GPUs, this is where the potential segfault might occur
    # if the underlying NCCL implementation has similar issues.
    strategy = tf.compat.v1.distribute.MirroredStrategy()
    
    print(f"Number of devices found: {strategy.num_replicas_in_sync}")

    # Corresponds to dist.barrier() and subsequent operations.
    # We enter the scope to ensure the communication backend is actively used.
    with strategy.scope():
        # Creating a variable forces the strategy to initialize the communication
        # and replicate the variable across devices.
        v = tf.Variable(1.0)
        # A simple assignment to verify the graph execution on the devices.
        v.assign(2.0)
        
    print("Strategy initialized and variable created successfully.")

if __name__ == "__main__":
    main()