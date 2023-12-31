#!/bin/bash
set -e
set -x

WORK_DIR=$PWD

# Prerequsite: conda, python 3.8 or above; cmake; LLVM-13

# Step 0: Create conda environment & install related packages
conda create -n ipex python=3.8 -y
conda activate ipex
conda install pytorch==1.13.0 torchvision==0.14.0 -c pytorch

# Step 1: Build IPEX
git clone --branch release/1.13 https://github.com/intel/intel-extension-for-pytorch
cd intel-extension-for-pytorch
git submodule update --init --recursive
# Specify Anonymous Compiler version
cd third_party/ideep/mkl-dnn/
git remote add origin_inner https://github.com/oneapi-src/oneDNN
git fetch origin_inner dev-graph-beta-3-paper
git checkout origin_inner/dev-graph-beta-3-paper
# Build
cd ../../../
sed -i '/set(DNNL_GRAPH_LLVM_CONFIG "/ s/llvm-config-13/\/llvm-project\/install\/bin\/llvm-config/g' csrc/cpu/CMakeLists.txt    # Enable LLVM
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
python setup.py install

# Step 2: Pull model zoo
cd ..
git clone --branch pytorch-r1.13-models https://github.com/IntelAI/models.git

# Step 3: General setup
pip install scikit-learn
pip install onnx
pip install lark-parser hypothesis
pip install tensorboardX
conda install numpy ninja pyyaml mkl mkl-include setuptools cffi typing_extensions future six requests dataclasses psutil
conda install -c conda-forge jemalloc
conda install intel-openmp

# install torch-ccl
git clone https://github.com/intel/torch-ccl.git &&cd torch-ccl
git checkout ccl_torch1.13
git submodule sync 
git submodule update --init --recursive
python setup.py install
cd ..

export LD_PRELOAD=${LD_PRELOAD}:/usr/lib/x86_64-linux-gnu/libstdc++.so.6