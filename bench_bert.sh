#!/bin/bash
set -e
set -x

# Step 1: Prepare Model
cd models/quickstart/language_modeling/pytorch/bert_large/inference/cpu
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.18.0
git apply ../enable_ipex_for_squad.diff
pip install -e ./
cd ../

# Step2: Install Dependency
conda install intel-openmp

# Step3: Download datasets
mkdir squad1.1
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O squad1.1/dev-v1.1.json
cp squad1.1/dev-v1.1.json .

# Ste4: Download fine-tuned model
mkdir bert_squad_model
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-config.json -O bert_squad_model/config.json
wget https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin  -O bert_squad_model/pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt -O bert_squad_model/vocab.txt

# # Step5: Set ENV
export FINETUNED_MODEL=$PWD/bert_squad_model
export EVAL_DATA_FILE=$PWD/dev-v1.1.json

# Step6: Bench
# bench int8 with batch size=32 / 128
_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=0 numactl -C 0-31 -m 0 python -u ./transformers/examples/legacy/question-answering/run_squad.py --benchmark --int8 --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name ${FINETUNED_MODEL} --do_eval --do_lower_case --predict_file ${EVAL_DATA_FILE} --per_gpu_eval_batch_size 32 --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter 15 --use_jit --perf_run_iters 40 --int8_config configure.json | tee int8_bs32_onednn_graph_compiler.log 2>&1

_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=1 numactl -C 0-31 -m 0 python -u ./transformers/examples/legacy/question-answering/run_squad.py --benchmark --int8 --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name ${FINETUNED_MODEL} --do_eval --do_lower_case --predict_file ${EVAL_DATA_FILE} --per_gpu_eval_batch_size 32 --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter 15 --use_jit --perf_run_iters 40 --int8_config configure.json | tee int8_bs32_onednn_primitives.log 2>&1

_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=0 numactl -C 0-31 -m 0 python -u ./transformers/examples/legacy/question-answering/run_squad.py --benchmark --int8 --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name ${FINETUNED_MODEL} --do_eval --do_lower_case --predict_file ${EVAL_DATA_FILE} --per_gpu_eval_batch_size 128--learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter 15 --use_jit --perf_run_iters 40 --int8_config configure.json | tee int8_bs128_onednn_graph_compiler.log 2>&1

_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=1 numactl -C 0-31 -m 0 python -u ./transformers/examples/legacy/question-answering/run_squad.py --benchmark --int8 --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name ${FINETUNED_MODEL} --do_eval --do_lower_case --predict_file ${EVAL_DATA_FILE} --per_gpu_eval_batch_size 128 --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter 15 --use_jit --perf_run_iters 40 --int8_config configure.json | tee int8_bs128_onednn_primitives.log 2>&1


# prepare ipex for bench fp32
cd  ../../../../../../../intel-extension-for-pytorch
sed -i '/thread_local bool llga_fp32_bf16_enabled/ s/false/true/g' csrc/jit/codegen/onednn/interface.cpp
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
python setup.py install
cd ..

# navigate to bert_large dir
cd models/quickstart/language_modeling/pytorch/bert_large/inference/cpu
export LD_PRELOAD=${LD_PRELOAD}:/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# bench fp32 bert large with batchsize = 32 / 128
_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=0 numactl -C 0-31 -m 0 python -u ./transformers/examples/legacy/question-answering/run_squad.py --benchmark --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name ${FINETUNED_MODEL} --do_eval --do_lower_case --predict_file ${EVAL_DATA_FILE} --per_gpu_eval_batch_size 32 --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter 15 --use_jit --perf_run_iters 40 --int8_config configure.json | tee fp32_bs32_onednn_graph_compiler.log 2>&1

_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=1 numactl -C 0-31 -m 0 python -u ./transformers/examples/legacy/question-answering/run_squad.py --benchmark --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name ${FINETUNED_MODEL} --do_eval --do_lower_case --predict_file ${EVAL_DATA_FILE} --per_gpu_eval_batch_size 32 --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter 15 --use_jit --perf_run_iters 40 --int8_config configure.json | tee fp32_bs32_onednn_primitives.log 2>&1

_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=0 numactl -C 0-31 -m 0 python -u ./transformers/examples/legacy/question-answering/run_squad.py --benchmark --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name ${FINETUNED_MODEL} --do_eval --do_lower_case --predict_file ${EVAL_DATA_FILE} --per_gpu_eval_batch_size 128 --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter 15 --use_jit --perf_run_iters 40 --int8_config configure.json | tee fp32_bs128_onednn_graph_compiler.log 2>&1

_ONEDNN_GRAPH_DISABLE_COMPILER_BACKEND=1 numactl -C 0-31 -m 0 python -u ./transformers/examples/legacy/question-answering/run_squad.py --benchmark --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name ${FINETUNED_MODEL} --do_eval --do_lower_case --predict_file ${EVAL_DATA_FILE} --per_gpu_eval_batch_size 128 --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter 15 --use_jit --perf_run_iters 40 --int8_config configure.json | tee fp32_bs128_onednn_primitives.log 2>&1
