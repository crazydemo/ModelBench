<!--- 0. Title -->
# PyTorch BERT Large inference

<!-- 10. Description -->
## Description

This document has instructions for running BERT Large SQuAD1.1 inference using
Intel-optimized PyTorch.

## Bare Metal
Make sure you have set up the general dependecies via `source setup_env.sh`
You should now at:
1. conda env: ipex
2. locate at /home/ModelBench

Then you can run
```
source bench_bert.sh
```
 The bench results are saved in `ModelBench/models/quickstart/language_modeling/pytorch/bert_large/inference/cpu`, named as `int8_bs32_onednn_graph_compiler.log`, or `fp32_bs32_onednn_primitives.log`, etc.
## More Details
Please refer to [link](https://github.com/IntelAI/models/blob/pytorch-r1.13-models/quickstart/language_modeling/pytorch/bert_large/inference/cpu/README.md).

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.
