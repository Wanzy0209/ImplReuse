```python
import sys
sys.path.append('third_party/Matcha-TTS')
import argparse
import os
import tensorflow as tf
from hyperpyyaml import load_hyperpyyaml
# Conversion: Importing specific internal APIs mentioned in the prompt
from tensorflow.python.framework.func_graph import func_graph_from_py_func
from tensorflow.python.eager.context import get_device_name

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
THIRD_PARTY = os.path.join(ROOT_DIR, "third_party", "Matcha-TTS")
if THIRD_PARTY not in sys.path:
    sys.path.append(THIRD_PARTY)


class HiFTInferenceWrapper(tf.keras.Model):
    def __init__(self, hift_module: tf.keras.Model):
        super().__init__()
        self.hift = hift_module

    def call(
        self,
        speech_feat: tf.Tensor,
        cache_source: tf.Tensor,
    ):
        speech, cache_out = self.hift.inference(
            speech_feat=speech_feat,
            cache_source=cache_source,
        )
        return speech, cache_out


def load_hift(model_dir: str) -> tf.keras.Model:
    config_path = os.path.join(model_dir, "cosyvoice2.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={
                "qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN"),
                "llm": None,
            },
        )
    hift = configs["hift"]
    
    # Conversion: torch.load -> tf.keras.models.load_model or load_weights
    # Note: The prompt suggests make_variable, but for loading a full model architecture 
    # and weights, load_weights is the standard TensorFlow equivalent.
    # We assume the weights are available in TensorFlow format (e.g., .h5 or checkpoint).
    weight_path = os.path.join(model_dir, "hift_weights.h5")
    if os.path.exists(weight_path):
        hift.load_weights(weight_path)
    else:
        print(f"Warning: TF weights not found at {weight_path}. Using random initialization.")
        
    return hift


def build_example_inputs(
    mel_frames: int,
    cache_frames: int,
) -> tuple:
    cache_frames = max(cache_frames, 1)
    # Conversion: torch.randn -> tf.random.normal
    # Conversion: torch.zeros -> tf.zeros
    # Device placement is handled implicitly or via tf.device context in TF
    speech_feat = tf.random.normal(
        [1, 80, mel_frames],
        dtype=tf.float32,
    )
    cache_source = tf.zeros(
        [1, 1, cache_frames],
        dtype=tf.float32,
    )
    return speech_feat, cache_source


def main():
    parser = argparse.ArgumentParser(
        description="Export CosyVoice2 HiFT vocoder to TensorFlow SavedModel."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="..",
        help="Directory that contains cosyvoice2.yaml and hift_weights.h5",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hift_model",
        help="SavedModel output path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for exporting",
    )
    # Opset version is specific to ONNX, not applicable to TF SavedModel export
    parser.add_argument(
        "--mel-frames",
        type=int,
        default=120,
        help="Example mel length (frames)",
    )
    parser.add_argument(
        "--cache-frames",
        type=int,
        default=3840,
        help="Example cache source length (samples)",
    )
    args = parser.parse_args()

    # Conversion: torch.cuda.is_available -> tf.config.list_physical_devices
    gpus = tf.config.list_physical_devices('GPU')
    use_cuda = len(gpus) > 0 and args.device == "cuda"
    
    # Conversion: torch.device -> get_device_name (or string construction)
    # We determine the device string to use in the context manager
    device_name = "/GPU:0" if use_cuda else "/CPU:0"
    
    # Conversion: torch.set_grad_enabled
    # In TensorFlow, gradients are not calculated by default during inference (eager execution).
    # We do not need an explicit disable flag unless using GradientTape.

    with tf.device(device_name):
        hift = load_hift(args.model_dir)
        wrapper = HiFTInferenceWrapper(hift)

        example_inputs = build_example_inputs(
            mel_frames=args.mel_frames,
            cache_frames=args.cache_frames,
        )

        # Conversion: torch.onnx.export -> func_graph_from_py_func
        # func_graph_from_py_func traces a python function into a TF Graph.
        # This is the low-level mechanism used by tf.function.
        
        # Note: dynamic_axes in ONNX correspond to tensor shapes in TF. 
        # TF SavedModels handle dynamic shapes automatically based on input tensors.
        
        # We trace the wrapper's call method
        graph = func_graph_from_py_func(
            name="hift_graph",
            python_func=wrapper.call,
            args=example_inputs,
            kwargs={}
        )
        
        # To actually "export" this to a usable format, we save the wrapper as a SavedModel.
        # The graph generated above is encapsulated in the SavedModel.
        tf.saved_model.save(wrapper, args.output)
        
        print(f"HiFT TensorFlow model exported to {args.output}")


if __name__ == "__main__":
    main()
```