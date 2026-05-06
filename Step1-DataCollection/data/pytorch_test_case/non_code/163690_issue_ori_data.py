(attn): Attention(
          (q_proj): LoRALinear(
            (original_layer): Linear(in_features=4096, out_features=4096, bias=False)
            (dropout): Identity()
          )
          (k_proj): LoRALinear(
            (original_layer): Linear(in_features=4096, out_features=4096, bias=False)
            (dropout): Identity()
          )
          (v_proj): LoRALinear(
            (original_layer): Linear(in_features=4096, out_features=4096, bias=False)
            (dropout): Identity()
          )
          (proj): LoRALinear(
            (original_layer): Linear(in_features=4096, out_features=4096, bias=True)
            (dropout): Identity()
          )
          (proj_drop): Dropout(p=0.0, inplace=False)
        )