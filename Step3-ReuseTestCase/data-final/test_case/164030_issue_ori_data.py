with torch.no_grad():
        expert_counts.scatter_(1, topk_ids, 1)
        tokens_per_expert = expert_counts.sum(dim=0)