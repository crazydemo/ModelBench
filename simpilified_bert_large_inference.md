<!--- 0. Title -->
# PyTorch BERT Large inference

<!-- 10. Description -->
## Description

This document has instructions for running BERT Large SQuAD1.1 inference using
Intel-optimized PyTorch.

## Bare Metal
### Prepare model
```
cd <clone of the model zoo>/quickstart/language_modeling/pytorch/bert_large/inference/cpu
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.18.0
git apply ../enable_ipex_for_squad.diff
pip install -e ./
cd ../

```
### Model Specific Setup
* Install dependency
```
conda install intel-openmp
```

* Download dataset

Please following this [link](https://github.com/huggingface/transformers/tree/v3.0.2/examples/question-answering) to get dev-v1.1.json

* Download fine-tuned model
```
mkdir bert_squad_model
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json -O bert_squad_model/config.json
wget https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin  -O bert_squad_model/pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt -O bert_squad_model/vocab.txt
```

* Set ENV to use AMX if you are using SPR
```
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
```

* Set ENV for model and dataset path, and optionally run with no network support
```
export FINETUNED_MODEL=#path/bert_squad_model
export EVAL_DATA_FILE=#/path/dev-v1.1.json
  
### correct EVAL_DATA_FILE
change EVAL_DATA_FILE=${VAL_DATA_FILE}
  
### [optional] Pure offline mode to benchmark:
change --tokenizer_name to #path/bert_squad_model in scripts before running
e.g. --tokenizer_name ${FINETUNED_MODEL} in run_multi_instance_throughput.sh
  
```

* [optional] Do calibration to get quantization config if you want do calibration by yourself.
```
export INT8_CONFIG=#/path/configure.json
run_calibration.sh
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
numactl -C 0-31 -m 0 /home/zhangyan/miniconda3/envs/ipex_reproduce/bin/python -u ./transformers/examples/legacy/question-answering/run_squad.py --benchmark --int8 --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name ${FINETUNED_MODEL} --do_eval --do_lower_case --predict_file ${VAL_DATA_FILE} --per_gpu_eval_batch_size 56 --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter 15 --use_jit --perf_run_iters 40 --int8_config configure.json
```

## More Details
Please refer to [link](https://github.com/IntelAI/models/blob/pytorch-r1.13-models/quickstart/language_modeling/pytorch/bert_large/inference/cpu/README.md).

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.
