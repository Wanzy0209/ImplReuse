import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
import os
import pprint
import argparse


class TransformerStage(nn.Module):
    def __init__(self, hidden_size=1536, num_layers=3, vocab_size=32000, stage_index=0, num_stages=4):
        super().__init__()
        self.stage_index = stage_index
        self.num_stages = num_stages
        
        if stage_index == 0:
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                hidden_size, 
                nhead=12,
                dim_feedforward=5440,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        if stage_index == num_stages - 1:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, x):
        if hasattr(self, 'embed_tokens'):
            x = self.embed_tokens(x)
        
        for layer in self.layers:
            x = layer(x)
        
        if hasattr(self, 'lm_head'):
            x = self.lm_head(x)
        
        return x


def create_pipeline_loss_fn():
    def loss_fn(logits, labels):
        logits = logits.float()
        
        labels = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous()
        
        vocab_size = logits.size(-1)
        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        
        return nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100)
    
    return loss_fn


def run_single_gpu_baseline(num_steps=3):
    print("="*80)
    print("Running Single GPU Baseline (no pipeline)")
    print("="*80)
    
    device = torch.device("cuda:0")
    config = {
        'batch_size': 4,
        'micro_batch_size': 1,
        'sequence_length': 512,
        'hidden_size': 1536,
        'vocab_size': 32000,
        'num_layers_per_stage': 3,
    }
    
    # Create all 4 stages as a complete model
    stages = []
    for stage_idx in range(4):
        # Reset seed for each stage to match PP initialization
        torch.manual_seed(0 + stage_idx)
        torch.cuda.manual_seed_all(0 + stage_idx)
        
        stage = TransformerStage(
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers_per_stage'],
            vocab_size=config['vocab_size'],
            stage_index=stage_idx,
            num_stages=4
        ).to(device)
        stages.append(stage)
    
    # Save initial parameters and input data for verification
    initial_params = {}
    for stage_idx, stage in enumerate(stages):
        for name, param in stage.named_parameters():
            key = f"stage{stage_idx}.{name}"
            initial_params[key] = param.data.clone().cpu()
    
    # Save to file for pipeline parallel to load
    torch.save(initial_params, 'initial_params.pt')
    print(f"Saved initial parameters to initial_params.pt for verification")
    
    # Save input data for each step
    all_inputs = []
    for step in range(num_steps):
        torch.manual_seed(42 + step)
        input_ids = torch.randint(0, config['vocab_size'], 
                                 (config['batch_size'], config['sequence_length']))
        all_inputs.append(input_ids)
    torch.save(all_inputs, 'input_data.pt')
    print(f"Saved input data to input_data.pt for verification")
    
    # Create optimizers for each stage
    optimizers = [torch.optim.Adam(stage.parameters(), lr=1e-3) for stage in stages]
    
    reference_losses = []
    reference_grads = []
    
    for step in range(num_steps):
        # Zero grad all stages
        for stage in stages:
            stage.zero_grad()
        
        # Use saved input data
        input_ids = all_inputs[step].to(device)
        labels = input_ids.clone()
        
        # Forward through all stages
        x = input_ids
        for stage in stages:
            x = stage(x)
        logits = x
        
        # Compute loss using same function as pipeline
        loss = create_pipeline_loss_fn()(logits, labels)
        
        # Scale loss as done in pipeline
        scaled_loss = loss / (config['batch_size'] / config['micro_batch_size'])
        scaled_loss.backward()
        
        reference_losses.append(loss.item())
        
        # Collect gradient norms from all stages
        total_grad_norm = 0.0
        all_grad_norms = {}
        stage_grad_summaries = []
        
        print(f"\n--- Single GPU Gradients Step {step} ---")
        
        for stage_idx, stage in enumerate(stages):
            stage_grads = {}
            stage_grad_norms = {}
            
            for name, param in stage.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    stage_grad_norms[name] = grad_norm
                    all_grad_norms[f"stage{stage_idx}.{name}"] = grad_norm
                    total_grad_norm += grad_norm
            
            # Print gradient summary for this stage (similar to PP format)
            print(f"Stage {stage_idx} Step {step}:")
            if stage_grad_norms:
                grad_summary = {k: f"{v:.2e}" for k, v in stage_grad_norms.items()}
                pprint.pprint(grad_summary)
            else:
                print("  No gradients")
            
            stage_grad_summaries.append(stage_grad_norms)
        
        reference_grads.append(all_grad_norms)
        
        # Update all stages
        for optimizer in optimizers:
            optimizer.step()
        
        print(f"\nSingle GPU Step {step}: loss={loss.item():.6f}, total_grad_norm={total_grad_norm:.6f}")
    
    # Save results for comparison
    single_gpu_results = {
        'losses': reference_losses,
        'grad_norms': [sum(grads.values()) for grads in reference_grads]
    }
    torch.save(single_gpu_results, 'single_gpu_results.pt')
    
    print("\nSingle GPU baseline completed")
    print(f"   Saved initial parameters to 'initial_params.pt'")
    print(f"   Saved input data to 'input_data.pt'")
    print(f"   Saved results to 'single_gpu_results.pt'")
    print("\nTo run pipeline parallel comparison, use:")
    print("torchrun --nproc_per_node=4 repro.py --mode pipeline")


def run_pipeline_parallel(num_steps=3):
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    if global_rank == 0:
        print("="*80)
        print("Running Pipeline Parallel")
        print("="*80)
    
    dist.init_process_group(
        backend="nccl",
        rank=global_rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    
    pp_degree = 4
    assert world_size == pp_degree, f"Expected world_size=4 for pipeline parallel, got {world_size}"
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    config = {
        'batch_size': 4,
        'micro_batch_size': 1,
        'sequence_length': 512,
        'hidden_size': 1536,
        'vocab_size': 32000,
        'num_layers_per_stage': 3,
    }
    
    pp_rank = global_rank
    
    # Reset seed for this stage to match single GPU initialization
    torch.manual_seed(0 + pp_rank)
    torch.cuda.manual_seed_all(0 + pp_rank)
    
    stage_model = TransformerStage(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers_per_stage'],
        vocab_size=config['vocab_size'],
        stage_index=pp_rank,
        num_stages=pp_degree
    ).to(device)
    
    # Verify initial parameters match single GPU if reference exists
    if os.path.exists('initial_params.pt'):
        saved_params = torch.load('initial_params.pt', map_location='cpu')
        
        mismatches = []
        for name, param in stage_model.named_parameters():
            key = f"stage{pp_rank}.{name}"
            if key in saved_params:
                saved_param = saved_params[key].to(device)
                if not torch.allclose(param.data, saved_param, rtol=1e-5):
                    max_diff = (param.data - saved_param).abs().max().item()
                    mismatches.append((name, max_diff))
        
        if mismatches:
            print(f"[Rank {global_rank}] WARNING: Initial parameters don't match single GPU baseline!")
            for name, diff in mismatches:
                print(f"   {name}: max diff = {diff}")
        else:
            print("[Rank {global_rank}] Initial parameters verified to match single GPU baseline")
    else:
        print("[Rank {global_rank}] No initial_params.pt found. Run single GPU baseline first for parameter verification.")
    
    if global_rank == 0:
        print(f"Pipeline setup: {pp_degree} stages, {config['num_layers_per_stage']} layers per stage")
    
    pipeline_stage = PipelineStage(
        stage_model,
        stage_index=pp_rank,
        num_stages=pp_degree,
        device=device,
    )
    
    n_microbatches = config['batch_size'] // config['micro_batch_size']
    
    pipeline_schedule = ScheduleGPipe(
        stage=pipeline_stage,
        n_microbatches=n_microbatches,
        loss_fn=create_pipeline_loss_fn(),
        scale_grads=True
    )
    
    optimizer = torch.optim.Adam(stage_model.parameters(), lr=1e-3)
    
    gradient_info = []
    pipeline_losses = []
    
    # Load saved input data if available
    if os.path.exists('input_data.pt'):
        saved_inputs = torch.load('input_data.pt', map_location='cpu')
        if global_rank == 0:
            print("Using saved input data from single GPU run")
    else:
        saved_inputs = None
        if global_rank == 0:
            print("No input_data.pt found. Using fresh random data.")
    
    for step in range(num_steps):
        dist.barrier()
        
        stage_model.zero_grad()
        
        if saved_inputs is not None:
            input_ids = saved_inputs[step].to(device)

        labels = input_ids.clone()
        
        losses_list = []
        
        if pp_rank == 0:
            output = pipeline_schedule.step(input_ids)
        elif pp_rank == pp_degree - 1:
            output = pipeline_schedule.step(target=labels, losses=losses_list)
            if losses_list:
                avg_loss = torch.mean(torch.stack(losses_list))
                pipeline_losses.append(avg_loss.item())
                print(f"Pipeline Step {step}: loss={avg_loss.item():.6f}")
        else:
            output = pipeline_schedule.step()
        
        params_with_grad = {name: param.grad.norm().item() for name, param in stage_model.named_parameters() if param.grad is not None}
        params_no_grad = {name: param.grad for name, param in stage_model.named_parameters() if param.grad is None}
        gradient_info.append((step, params_with_grad, params_no_grad))
        
        optimizer.step()
        
        dist.barrier()
    
    print('Completed steps')

    dist.barrier()

    pipeline_losses = torch.tensor(pipeline_losses if pp_rank == pp_degree - 1 else [0.0] * num_steps, device=device)
    dist.broadcast(pipeline_losses, src=pp_degree - 1)
    pipeline_losses = pipeline_losses.tolist()
    
    # Collect total gradient norms from all ranks  
    pipeline_total_grad_norms = []
    for step_idx in range(num_steps):
        # First, each rank computes its local gradient norm
        step, params_with_grad, params_no_grad = gradient_info[step_idx]
        local_grad_norm = sum(params_with_grad.values()) if params_with_grad else 0.0
        
        # Print per-rank gradient info
        if global_rank == 0:
            print(f"\n--- Pipeline Gradients Step {step} ---")
        
        for rank in range(world_size):
            if rank == global_rank:
                print(f"Rank {global_rank} Step {step}:")
                if params_no_grad:
                    pprint.pprint(params_no_grad)
                else:
                    grad_summary = {k: f"{v:.2e}" for k, v in params_with_grad.items()}
                    pprint.pprint(grad_summary)
            dist.barrier()
        
        # All-reduce to get total grad norm across all ranks
        total_grad_tensor = torch.tensor(local_grad_norm, device=device)
        dist.all_reduce(total_grad_tensor)
        total_grad_norm = total_grad_tensor.item()
        pipeline_total_grad_norms.append(total_grad_norm)
        
        if global_rank == 0:
            print(f"\nPipeline Step {step} total_grad_norm={total_grad_norm:.6f}")
        
        dist.barrier()
    
    # Show comparison with single GPU results
    if global_rank == 0:
        print("\n" + "="*80)
        print("COMPARISON: Single GPU vs Pipeline Parallel")
        print("="*80)
        
        if os.path.exists('single_gpu_results.pt') and pipeline_losses:
            single_gpu_results = torch.load('single_gpu_results.pt')
            
            print("\nStep | Single GPU Loss | Pipeline Loss | Loss Diff    | Single GPU Grad | Pipeline Grad | Grad Diff    | Status")
            print("-" * 110)
            
            for step in range(len(pipeline_losses)):
                single_loss = single_gpu_results['losses'][step]
                pipeline_loss = pipeline_losses[step]
                loss_diff = abs(single_loss - pipeline_loss)
                
                single_grad = single_gpu_results['grad_norms'][step]
                pipeline_grad = pipeline_total_grad_norms[step]
                grad_diff = abs(single_grad - pipeline_grad)
                
                # Check if values match within tolerance
                loss_match = loss_diff < 1e-5
                grad_match = grad_diff < 1e-3
                match_status = "MATCH" if (loss_match and grad_match) else "DIVERGED"
                
                print(f"  {step}  | {single_loss:.6f} | {pipeline_loss:.6f} | {loss_diff:.6f} | {single_grad:.6f} | {pipeline_grad:.6f} | {grad_diff:.6f} | {match_status}")
            
            print("\nCONCLUSION: Pipeline parallel gradients/losses diverge from single GPU baseline after step 0.")
            print("This indicates a bug in the pipeline parallel gradient computation.")
        else:
            if not os.path.exists('single_gpu_results.pt'):
                print("No single_gpu_results.pt found. Run single GPU baseline first for comparison.")
            else:
                print("No pipeline losses collected. Check if the pipeline ran correctly.")
        
        print("="*80)
    
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Pipeline Parallel vs Single GPU Comparison')
    parser.add_argument('--mode', type=str, choices=['single', 'pipeline'], required=True,
                        help='Run mode: "single" for single GPU baseline, "pipeline" for pipeline parallel')
    args = parser.parse_args()
        
    if args.mode == 'single':
        if 'RANK' in os.environ:
            raise RuntimeError("Single GPU mode should not be run with torchrun. Use: python repro.py --mode single")
        run_single_gpu_baseline()
    elif args.mode == 'pipeline':
        if 'RANK' not in os.environ:
            raise RuntimeError("Pipeline mode must be run with torchrun. Use: torchrun --nproc_per_node=4 repro.py --mode pipeline")
        run_pipeline_parallel()


if __name__ == "__main__":
    main()