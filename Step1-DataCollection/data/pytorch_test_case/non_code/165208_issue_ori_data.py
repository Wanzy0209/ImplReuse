git submodule sync
git submodule update --init --recursive

conda install cmake ninja
pip install -r requirements.txt

pip install mkl-static mkl-include
.ci/docker/common/install_magma_conda.sh 12.6
make triton

export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=8
python3.12 setup.py develop