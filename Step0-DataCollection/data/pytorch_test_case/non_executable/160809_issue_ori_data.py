FROM intel/intel-extension-for-pytorch:2.8.10-xpu

# Install basic dependencies + OpenGL libraries
RUN apt-get update && apt-get install -y \
    git wget python3-venv \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create persistent workspace directory for ComfyUI
WORKDIR /workspace

# Clone ComfyUI (latest)
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

# Install ComfyUI dependencies in system environment (so they persist with volume)
WORKDIR /workspace/ComfyUI
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Default command will run ComfyUI
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "9002"]