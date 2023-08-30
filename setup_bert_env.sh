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

# Step6: modify bench shell
sed -i 's/--tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad/--tokenizer_name ${FINETUNED_MODEL}/' run_multi_instance_throughput.sh

