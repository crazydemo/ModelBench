#!/bin/bash
set -e
set -x

# Step 1: Install Dependency
cd models
export MODEL_DIR=$(pwd)
cd quickstart/recommendation/pytorch/dlrm
pip install future tqdm
cd inference/cpu

# Step2: Set ENV
export LD_PRELOAD=${LD_PRELOAD}:/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_PRELOAD="/miniconda/envs/ipex/lib/libjemalloc.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export LD_PRELOAD=/miniconda/envs/ipex/lib/libiomp5.so:$LD_PRELOAD

# Step3: Modify scipts
sed -i 's/OMP_NUM_THREADS=1/OMP_NUM_THREADS=32/' inference_performance.sh
sed -i 's/-share-weight-instance=$Cores/-share-weight-instance=0/' inference_performance.sh

