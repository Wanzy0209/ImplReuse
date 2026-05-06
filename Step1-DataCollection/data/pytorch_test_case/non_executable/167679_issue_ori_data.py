conda create -n mps_torch python=3.11 -y
conda activate mps_torch
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu

python - << 'EOF'
import sys, platform, torch
print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("macOS:", platform.mac_ver()[0])
print("platform:", platform.platform())
print("arch:", platform.machine())
print("is_macos_or_newer(14,0):",
      getattr(torch.backends.mps, "is_macos_or_newer", lambda *a: "no-func")(14,0))
print("MPS built?:", torch.backends.mps.is_built())
print("MPS available?:", torch.backends.mps.is_available())
EOF