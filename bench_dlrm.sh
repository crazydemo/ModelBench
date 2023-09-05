#!/bin/bash
set -e
set -x

# Step 1: Set IPEX for int8
cd  intel-extension-for-pytorch
sed -i '/thread_local bool llga_fp32_bf16_enabled/ s/true/false/g' csrc/jit/codegen/onednn/interface.cpp
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
python setup.py install
cd ..

# Step2: Set ENV
cd models
export MODEL_DIR=$(pwd)
cd quickstart/recommendation/pytorch/dlrm/inference/cpu
export LD_PRELOAD=${LD_PRELOAD}:/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_PRELOAD="/miniconda/envs/ipex/lib/libjemalloc.so":$LD_PRELOAD
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export LD_PRELOAD=/miniconda/envs/ipex/lib/libiomp5.so:$LD_PRELOAD

# Step3: Bench
# bench int8 with batch size=32 / 512
_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=0 numactl -C 0-31 -m 0 python -u ${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz --data-set=terabyte --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 --arch-sparse-feature-size=128 --max-ind-range=40000000 --ipex-interaction --numpy-rand-seed=727 --inference-only --num-batches=1000 --print-freq=10 --print-time --mini-batch-size=32 --share-weight-instance=32 --num-cpu-cores=32 --int8 --int8-configure=${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/int8_configure.json | tee int8_bs32_onednn_graph_compiler.log 2>&1

_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=1 numactl -C 0-31 -m 0 python -u ${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz --data-set=terabyte --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 --arch-sparse-feature-size=128 --max-ind-range=40000000 --ipex-interaction --numpy-rand-seed=727 --inference-only --num-batches=1000 --print-freq=10 --print-time --mini-batch-size=32 --share-weight-instance=32 --num-cpu-cores=32 --int8 --int8-configure=${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/int8_configure.json | tee int8_bs32_onednn_primitives.log 2>&1

_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=0 numactl -C 0-31 -m 0 python -u ${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz --data-set=terabyte --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 --arch-sparse-feature-size=128 --max-ind-range=40000000 --ipex-interaction --numpy-rand-seed=727 --inference-only --num-batches=1000 --print-freq=10 --print-time --mini-batch-size=512 --share-weight-instance=32 --num-cpu-cores=32 --int8 --int8-configure=${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/int8_configure.json | tee int8_bs512_onednn_graph_compiler.log 2>&1

_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=1 numactl -C 0-31 -m 0 python -u ${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz --data-set=terabyte --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 --arch-sparse-feature-size=128 --max-ind-range=40000000 --ipex-interaction --numpy-rand-seed=727 --inference-only --num-batches=1000 --print-freq=10 --print-time --mini-batch-size=512 --share-weight-instance=32 --num-cpu-cores=32 --int8 --int8-configure=${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/int8_configure.json | tee int8_bs512_onednn_primitives.log 2>&1


# prepare ipex for bench fp32
cd  ../../../../../../../intel-extension-for-pytorch
sed -i '/thread_local bool llga_fp32_bf16_enabled/ s/false/true/g' csrc/jit/codegen/onednn/interface.cpp
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
python setup.py install
cd ..

# navigate to bert_large dir
cd quickstart/recommendation/pytorch/dlrm/inference/cpu
export LD_PRELOAD=${LD_PRELOAD}:/usr/lib/x86_64-linux-gnu/libstdc++.so.6
sed -i 's/dlrm = ipex.optimize/#dlrm = ipex.optimize/g' ${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py
sed -i 's/dlrm, optimizer = ipex.optimize/#dlrm, optimizer = ipex.optimize/g' ${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py

# bench fp32 bert large with batchsize = 32 / 512
_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=0 numactl -C 0-31 -m 0 python -u ${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz --data-set=terabyte --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 --arch-sparse-feature-size=128 --max-ind-range=40000000 --ipex-interaction --numpy-rand-seed=727 --inference-only --num-batches=1000 --print-freq=10 --print-time --mini-batch-size=32 --share-weight-instance=32 | tee fp32_bs32_onednn_graph_compiler.log 2>&1

_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=1 numactl -C 0-31 -m 0 python -u ${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz --data-set=terabyte --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 --arch-sparse-feature-size=128 --max-ind-range=40000000 --ipex-interaction --numpy-rand-seed=727 --inference-only --num-batches=1000 --print-freq=10 --print-time --mini-batch-size=32 --share-weight-instance=32 | tee fp32_bs32_onednn_primitives.log 2>&1

_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=0 numactl -C 0-31 -m 0 python -u ${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz --data-set=terabyte --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 --arch-sparse-feature-size=128 --max-ind-range=40000000 --ipex-interaction --numpy-rand-seed=727 --inference-only --num-batches=1000 --print-freq=10 --print-time --mini-batch-size=512 --share-weight-instance=32 | tee fp32_bs512_onednn_graph_compiler.log 2>&1

_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=1 numactl -C 0-31 -m 0 python -u ${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz --data-set=terabyte --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 --arch-sparse-feature-size=128 --max-ind-range=40000000 --ipex-interaction --numpy-rand-seed=727 --inference-only --num-batches=1000 --print-freq=10 --print-time --mini-batch-size=512 --share-weight-instance=32 | tee fp32_bs512_onednn_primitives.log 2>&1



