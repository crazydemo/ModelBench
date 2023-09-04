# ModelBench
Instructions on benching AI model with anonymous compiler
[![DOI](https://zenodo.org/badge/679274457.svg)](https://zenodo.org/badge/latestdoi/679274457)

## Docker
[Docker](https://drive.google.com/drive/folders/1sLo7OFIqAmQb-zYzTo-rf-fKAFU3ajpG?usp=sharing)

## Setup ENV
use following cmd to install ipex and intel ai model zoo.
```
source setup_env.sh
```

## Bench Bert Large
Please refer to [Simplified Bert Large Inference](https://github.com/crazydemo/ModelBench/blob/main/simpilified_bert_large_inference.md). The detailed guidance is in [Detailed Bert Large Inference](https://github.com/IntelAI/models/blob/pytorch-r1.13-models/quickstart/language_modeling/pytorch/bert_large/inference/cpu/README.md)

## Bench DLRM
Please refer to [Simplified DLRM Inference](https://github.com/crazydemo/ModelBench/blob/main/simplified_dlrm_inference.md). The detailed guidance is in [Detailed DLRM Inference](https://github.com/IntelAI/models/blob/pytorch-r1.13-models/quickstart/recommendation/pytorch/dlrm/inference/cpu/README.md)
## Manual Config
By default, only int8 mode of IPEX will run into oneDNN Graph. If you want to benchmark the performance under fp32 mode with oneDNN Graph and Anonymous Compiler, you need to manually do some modification on IPEX and Model Zoo source code.
For IPEX, `navigate to [IPEX_ROOT]/csrc/cpu/jit/codegen/onednn/interface.cpp and change the value of llga_fp32_bf16_enabled from false to true`:
```
bool llga_fp32_bf16_enabled = true;
```
Then re-build IPEX:
```
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
python setup.py install
```
For Model Zoo Scripts, you need to search for the model’s main script
and comment out the `ipex.optimize` in the corresponding dtype’s condition
branch. For instance, as for DLRM model under fp32 mode, `navigate to dlrm_s_pytorch.py`, comment out
the following code:
```
dlrm = ipex.optimize(dlrm, dtype=torch.float, ...)
```
Note that, the above changes need to be reverted when you are running in int8 mode. As for benchmark AI models, you can follow the instructions on [Bert Large](https://github.com/IntelAI/models/blob/pytorch-r1.13-models/quickstart/language_modeling/pytorch/bert_large/inference/cpu/README.md) and [DLRM](https://github.com/IntelAI/models/blob/pytorch-r1.13-models/quickstart/recommendation/pytorch/dlrm/inference/cpu/README.md) to prepare the model specified dependency and
datasets. For bench DLRM model, you need to modify the below two args to switch to throughput mode (32 cores per instance).
```
OMP_NUM_THREADS=1 => OMP_NUM_THREADS=32
-share-weight-instance=$Cores => -share-weight-instance=0
```
## Switch Anonymous Compiler and oneDNN
By default, anonymous compiler is enabled on IPEX. To get oneDNN's performance data on end-to-end model, you need to set the below environment variable:
```
export _DNNL_DISABLE_COMPILER_BACKEND=1
```
## Evaluation and expected result
To check whether the Anonymous Compiler is enabled or not, you can check the graph verbose (via export ONEDNN GRAPH VERBOSE=1). You will see the below outputs from verbose once compiler backend is enabled.
```
onednn_graph_verbose,info,backend,0:compiler_backend
onednn_graph_verbose,info,backend,1:dnnl_backend
```


