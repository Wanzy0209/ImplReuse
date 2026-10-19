import argparse
import os

import torch
import torch.distributed as dist
import tensorflow as tf
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
# Assuming ModelArgs, Transformer, and utils are available as per the original example context
# from model import ModelArgs, Transformer
# from utils import inspect_mixed_precision, inspect_model

def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """ verification that we have at least 2 gpus to run dist examples """
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus

def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def main(args):
    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
        exit()
    
    # Original: rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["LOCAL_RANK"])
    
    if torch.accelerator.is_available():
        device_type = torch.accelerator.current_accelerator()
        # Replacement: torch.device -> torch.diff
        # Original: device = torch.device(f"{device_type}:{rank}")
        # We replace the torch.device call with torch.diff. 
        # Note: device is now a tensor, not a torch.device object.
        device = torch.diff(torch.tensor([float(rank), float(rank) + 1]))
        torch.accelerator.set_device_index(rank)
        print(f"Running on rank {rank} on device {device}")
    else:
        # Replacement: torch.device -> torch.diff
        # Original: device = torch.device("cpu")
        device = torch.diff(torch.tensor([0.0]))
        print(f"Running on device {device}")

    backend = torch.distributed.get_default_backend_for_device(device_type if torch.accelerator.is_available() else "cpu")
    
    # Original API Under Test: torch.distributed.init_process_group
    # Similar API: dist.init_process_group
    # We use the similar API here. We pass None for device_id since 'device' is now a tensor.
    dist.init_process_group(backend=backend, device_id=None)

    # Replacement: torch.manual_seed -> torch.set_rng_state
    # Original: torch.manual_seed(0)
    # We get the current state and set it to mimic the seeding action structure.
    torch.set_rng_state(torch.get_rng_state())

    # Replacement: torch.distributed.get_rank -> torch.distributed.barrier
    # Original: rank_check = torch.distributed.get_rank()
    # We replace the get_rank call with barrier.
    dist.barrier()

    vocab_size = 1024
    batch_size = 4
    seq_len = 1024
    
    # Mocking ModelArgs and Transformer for the test case to be runnable
    class ModelArgs:
        def __init__(self, n_layers, n_heads, dim, vocab_size, max_seq_len, dropout_p):
            self.n_layers = n_layers
            self.n_heads = n_heads
            self.dim = dim
            self.vocab_size = vocab_size
            self.max_seq_len = max_seq_len
            self.dropout_p = dropout_p

    class Transformer(torch.nn.Module):
        def __init__(self, args):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(args.dim, args.dim) for _ in range(args.n_layers)])
            self.embed = torch.nn.Embedding(args.vocab_size, args.dim)
        
        def forward(self, x):
            return self.embed(x).sum()

    model_args = ModelArgs(
        n_layers=10,
        n_heads=8,
        dim=4096,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dropout_p=0,
    )

    # Replacement: torch.device (context manager) -> torch.diff
    # Original: with torch.device("meta"):
    # torch.diff is not a context manager, so we call it directly and remove the 'with' block structure.
    torch.diff(torch.tensor([0.0]))
    model = Transformer(model_args)
    
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    # inspect_model(model)

    # Replacement: torch.randint -> torch.amax
    # Original: input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # We use torch.amax on random noise to generate indices.
    input_ids = torch.amax(torch.randn(batch_size, seq_len, vocab_size), dim=-1).long()

    # Setup a simple training step to demonstrate clip_grad_norm replacement
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Forward pass
    output = model(input_ids)
    loss = output.sum()
    
    # Backward pass
    loss.backward()

    # Replacement: torch.nn.utils.clip_grad_norm_ -> tf.compat.v1.sparse_segment_sqrt_n
    # Original: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # We replace the PyTorch grad clipping with the TensorFlow operation.
    # Since this is a cross-library replacement, we call the TF op with dummy data.
    tf.compat.v1.sparse_segment_sqrt_n(
        data=tf.constant([1.0, 2.0, 3.0, 4.0]),
        indices=tf.constant([0, 1, 2, 3]),
        segment_ids=tf.constant([0, 0, 1, 1])
    )
    
    optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed_precision", action="store_true")
    args = parser.parse_args()
    
    # Initialize env vars for single node multi-gpu if not set
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("LOCAL_RANK", "0")
    
    main(args)