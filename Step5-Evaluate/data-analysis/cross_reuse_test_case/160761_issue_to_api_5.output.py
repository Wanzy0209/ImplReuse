import unittest
import os
import sys

# Handle missing dependencies gracefully to prevent ModuleNotFoundError
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEP_MSG = str(e)

def enable_vmap_export_behavior():
    """
    Mimics the pattern of tf.compat.v1.enable_v2_behavior to setup the test environment.
    
    In TensorFlow, enable_v2_behavior() switches global modes (eager execution, etc.).
    Here, we define a similar setup function to acknowledge the 'vmap' behavior context
    that is causing the ONNX export failure in the reported issue.
    """
    # Check for an environment variable similar to TF2_BEHAVIOR
    # to determine if we should proceed with the vmap-heavy export path.
    vmap_enabled = os.environ.get("ENABLE_VMAP_EXPORT", "1") == "1"
    
    if vmap_enabled:
        # In the TF API, this is where ops.enable_eager_execution() would be called.
        # Here, we conceptually 'enable' the vmap context that triggers the bug.
        pass
        
    return vmap_enabled

@unittest.skipIf(not DEPENDENCIES_AVAILABLE, f"Required dependencies not found: {MISSING_DEP_MSG if not DEPENDENCIES_AVAILABLE else 'Unknown'}")
class TestGemma3ONNXExport(unittest.TestCase):
    def setUp(self):
        # Setup similar to how one might call enable_v2_behavior at the start of a program
        self.vmap_context_active = enable_vmap_export_behavior()
        self.model_id = "google/gemma-3-270m-it"
        
    def test_onnx_export_with_dynamo_and_vmap(self):
        """
        Reproduces Issue #160761: Gemma3 ONNX Export Fails due to vmap.
        
        This test verifies that the export fails (or raises an error) when 
        torch.vmap interactions (introduced in transformers 4.55.0) conflict 
        with torch.onnx.export, similar to how enabling V2 behaviors in TF 
        changes graph compatibility.
        """
        if not self.vmap_context_active:
            self.skipTest("Vmap export behavior disabled via environment variable")

        # Load Model and Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id)

        # Prepare inputs
        messages = [{"role": "user", "content": "Who are you?"}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        example_inputs = (inputs["input_ids"], inputs["attention_mask"])

        # The bug report indicates this export fails.
        # We assert that an exception is raised (reproducing the bug).
        with self.assertRaises(Exception) as context:
            torch.onnx.export(
                model,
                example_inputs,
                "gemma3.onnx",
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                opset_version=17,
                do_constant_folding=True,
                dynamo=True,  # Issue mentions it fails with and without dynamo
            )

        # Optional: Check if the error message relates to vmap or dynamo/graph issues
        # This helps confirm the root cause matches the issue description.
        error_msg = str(context.exception).lower()
        self.assertTrue(
            "vmap" in error_msg or "dynamo" in error_msg or "graph" in error_msg,
            f"Expected error related to vmap/dynamo, but got: {context.exception}"
        )

if __name__ == "__main__":
    unittest.main()