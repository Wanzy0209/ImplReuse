AssertionError: s52 (could be from ["L['position_ids']._base.size()[0]"]) not in {
s53: ["L['attention_mask'].size()[1]", "L['attention_mask'].stride()[0]"],
s58: ["L['cache_position'].size()[0]", "L['position_ids']._base.size()[0]"],
s55: ["L['input_embeds'].size()[1]"],
s9: ["L['position_ids'].size()[1]", "L['position_ids'].stride()[0]"],
s52: []
}.  If this assert is failing, it could be due to the issue described in https://github.com/pytorch/pytorch/pull/90665