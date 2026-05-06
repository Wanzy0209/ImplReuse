temp = nn.Parameter(torch.tensor(0.0))
    def score_mod(score, b, h, q, kv):
        #print(f"const: {const}")
        score = score + temp
        return score