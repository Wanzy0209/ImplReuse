import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as fc

def test_stable_diffusion_with_crossed_column_feature():
    """
    Test case that preserves the original Stable Diffusion reproduction logic
    while leveraging the similar API (tf.feature_column.crossed_column) 
    to process the input prompt structure.
    """
    
    # --- Leverage Similar API: tf.feature_column.crossed_column ---
    # We reuse the pattern of defining a feature interaction (crossing) 
    # to structure the input prompt, mirroring the argument passing pattern
    # found in the similar API definition.
    
    prompt_text = "a photo of an astronaut riding a horse on mars"
    
    # Define categorical features to be crossed
    # This simulates handling the input data with the similar API's logic
    prompt_feature = fc.categorical_column_with_hash_bucket("prompt", hash_bucket_size=1000)
    style_feature = fc.categorical_column_with_identity("style", num_buckets=10)
    
    # Instantiate the crossed_column (The Similar API)
    # This mirrors the structural pattern: API_Name(keys, config_args)
    crossed_col = fc.crossed_column([prompt_feature, style_feature], hash_bucket_size=5000)
    
    # Verify the API object is created successfully
    assert crossed_col is not None, "Failed to create crossed_column feature"

    # --- Preserve Original Bug Reproduction Logic ---
    # Setup and run Stable Diffusion on CUDA to check for CUDNN errors
    
    model_id = "stabilityai/stable-diffusion-2-1"
    
    try:
        # Initialize pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")

        # Run inference loop (reproducing the bug scenario)
        # Note: While we used the TF API to define the feature structure above,
        # we pass the raw string to the PyTorch pipeline as required by its API.
        image = pipe(prompt_text).images[0]
        
        # Assertion to verify successful execution (Bug is not present)
        assert image is not None, "Pipeline returned no image"
        print("Test Passed: Stable Diffusion executed without CUDNN errors.")
        
    except RuntimeError as e:
        # Catch specific CUDNN errors mentioned in the bug report
        if "CUDNN" in str(e):
            print(f"Test Failed: CUDNN Error detected - {e}")
            raise AssertionError(f"CUDNN Error encountered: {e}")
        else:
            raise

if __name__ == "__main__":
    test_stable_diffusion_with_crossed_column_feature()