FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel
RUN apt-get update && apt-get install -y curl
RUN curl -sSL https://github.com/pytorch/extension-cpp/archive/0ec4969.tar.gz | tar -xvz && \
        sed -i 's/torch.cuda.is_available()/True/g' extension-cpp-0ec4969c7bc8e15a8456e5eb9d9ca0a7ec15bc95/setup.py
WORKDIR extension-cpp-0ec4969c7bc8e15a8456e5eb9d9ca0a7ec15bc95
ENV TORCH_CUDA_ARCH_LIST=10.1
RUN pip install --break-system-packages --no-build-isolation -e .