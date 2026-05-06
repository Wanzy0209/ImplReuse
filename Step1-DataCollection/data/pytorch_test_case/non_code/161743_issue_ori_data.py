{
    "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "bnb_config": {
        "load_in_8bit": false,
        "load_in_4bit": false,
        "bnb_4bit_use_double_quant": true,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16"
    },
    "ACCESS_TOKEN": "insert your token",
    "dataset_name": "rajpurkar/squad",
    "batch_size": 8,
    "max_length": 100,
    "do_sample": false, 
    "num_beams": 1,
    "cache_implementation": "static",
    "torch.compile": false
}