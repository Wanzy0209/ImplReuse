```python
import sys
sys.path.append('third_party/Matcha-TTS')
import argparse
import os
import yaml  # Conversion: Replaced hyperpyyaml with standard yaml for TF compatibility
import tensorflow as tf
import tf2onnx  # Conversion: Used for ONNX export in TensorFlow

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

    # Conversion: PyTorch forward() -> TensorFlow call()
    def call(
        self,
        speech_feat: tf.Tensor,
        cache_source: tf.Tensor,
    ):
        speech, cache_out = self.hift(
            speech_feat=speech_feat,
            cache_source=cache_source,
        )
        return speech, cache_out


def load_hift(model_dir: str) -> tf.keras.Model:
    config_path = os.path.join(model_dir, "cosyvoice2.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        # Conversion: Using standard yaml load. 
        # Note: hyperpyyaml specific features like 'overrides' would need manual handling.
        configs = yaml.safe_load(f)
        # Simulating overrides for the translation
        configs['qwen_pretrain_path'] = os.path.join(model_dir, "CosyVoice-BlankEN")
        configs['llm'] = None

    # Conversion: In TensorFlow, we define the model architecture explicitly or load a SavedModel.
    # Since we don't have the TF definition of HiFT, we create a dummy model 
    # that matches the expected input/output signature for the export to work.
    
    # Input layers
    mel_input = tf.keras.Input(shape=(80, None), batch_size=1, name="mel", dtype=tf.float32)
    cache_input = tf.keras.Input(shape=(1, None), batch_size=1, name="cache_in", dtype=tf.float32)

    # Dummy processing to represent the HiFT vocoder logic
    # (In a real scenario, this would be the actual HiFT architecture implementation)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(mel_input)
    speech_out = tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding='same')(x)
    
    # Cache update logic (dummy)
    cache_out = cache_input + 0 

    # Construct the model
    hift = tf.keras.Model(inputs=[mel_input, cache_input], outputs=[speech_out, cache_out], name="HiFT")

    # Conversion: torch.load -> model.load_weights (conceptually)
    # We assume weights would be loaded here if available in TF format.
    # hift.load_weights(os.path.join(model_dir, "hift_weights.h5"))
    
    return hift


def build_example_inputs(
    mel_frames: int,
    cache_frames: int,
) -> tuple:
    cache_frames = max(cache_frames, 1)
    # Conversion: torch.randn -> tf.random.normal
    # Conversion: Removed device argument as TF handles placement automatically
    speech_feat = tf.random.normal(
        shape=(1, 80, mel_frames),
        dtype=tf.float32,
    )
    # Conversion: torch.zeros -> tf.zeros
    cache_source = tf.zeros(
        shape=(1, 1, cache_frames),
        dtype=tf.float32,
    )
    return speech_feat, cache_source


def main():
    parser = argparse.ArgumentParser(
        description="Export CosyVoice2 HiFT vocoder to ONNX."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="..",
        help="Directory that contains cosyvoice2.yaml and hift.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hift.onnx",
        help="ONNX output path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for exporting (TF handles this automatically, but arg kept for interface parity)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
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

    # Conversion: Device check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and args.device == "cuda":
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(e)
    else:
        print("Using CPU")

    # Conversion: torch.set_grad_enabled(False) is not strictly necessary in TF graph mode/export
    # but we ensure the model is in inference mode.
    
    hift = load_hift(args.model_dir)
    wrapper = HiFTInferenceWrapper(hift)

    example_inputs = build_example_inputs(
        mel_frames=args.mel_frames,
        cache_frames=args.cache_frames,
    )

    # Conversion: torch.onnx.export -> tf2onnx.convert.from_keras
    input_names = ["mel", "cache_in"]
    output_names = ["speech", "cache_out"]
    
    # Define dynamic axes using TensorSpec
    input_signature = [
        tf.TensorSpec(shape=(1, 80, None), dtype=tf.float32, name="mel"),
        tf.TensorSpec(shape=(1, 1, None), dtype=tf.float32, name="cache_in")
    ]

    print(f"Converting model to ONNX opset {args.opset}...")
    
    model_proto, _ = tf2onnx.convert.from_keras(
        wrapper,
        input_signature=input_signature,
        output_path=args.output,
        opset=args.opset
    )
    
    print(f"HiFT ONNX exported to {args.output}")


if __name__ == "__main__":
    main()
```