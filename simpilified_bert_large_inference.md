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
source setup_bert_env.sh
```

## Quick Start Scripts

|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | bash run_multi_instance_throughput.sh fp32 | bash run_multi_instance_realtime.sh fp32 | bash run_accuracy.sh fp32 |
| BF32        | bash run_multi_instance_throughput.sh bf32 | bash run_multi_instance_realtime.sh bf32 | bash run_accuracy.sh bf32 |
| BF16        | bash run_multi_instance_throughput.sh bf16 | bash run_multi_instance_realtime.sh bf16 | bash run_accuracy.sh bf16 |
| FP16        | bash run_multi_instance_throughput.sh fp16 | bash run_multi_instance_realtime.sh fp16 | bash run_accuracy.sh fp16 |
| INT8        | bash run_multi_instance_throughput.sh int8 | bash run_multi_instance_realtime.sh int8 | bash run_accuracy.sh int8 |

## Run the model

You can use the following cmd to run the model on single socket with batchsize=56.
```
numactl -C 0-31 -m 0 /home/zhangyan/miniconda3/envs/ipex_reproduce/bin/python -u ./transformers/examples/legacy/question-answering/run_squad.py --benchmark --int8 --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name ${FINETUNED_MODEL} --do_eval --do_lower_case --predict_file ${EVAL_DATA_FILE} --per_gpu_eval_batch_size 56 --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter 15 --use_jit --perf_run_iters 40 --int8_config configure.json
```

## More Details
Please refer to [link](https://github.com/IntelAI/models/blob/pytorch-r1.13-models/quickstart/language_modeling/pytorch/bert_large/inference/cpu/README.md).

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.
