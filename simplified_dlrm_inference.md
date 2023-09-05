<!--- 0. Title -->
# PyTorch DLRM inference

<!-- 10. Description -->
## Description

This document has instructions for running DLRM inference using
Intel-optimized PyTorch for bare metal.

## Bare Metal

## Datasets

### Criteo Terabyte Dataset

The [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) is
used to run DLRM. To download the dataset, you will need to visit the Criteo website and accept
their terms of use:
[https://labs.criteo.com/2013/12/download-terabyte-click-logs/](https://labs.criteo.com/2013/12/download-terabyte-click-logs/).
Copy the download URL into the command below as the `<download url>` and
replace the `<dir/to/save/dlrm_data>` to any path where you want to download
and save the dataset.
```bash
export DATASET_DIR=/dlrm_data

mkdir ${DATASET_DIR} && cd ${DATASET_DIR}
curl -O <download url>/day_{$(seq -s , 0 23)}.gz
gunzip day_*.gz
```
If the `day_*.gz` is empty, please download them from website, i.e. using chrome to download.
The raw data will be automatically preprocessed and saved as `day_*.npz` to
the `DATASET_DIR` when DLRM is run for the first time. On subsequent runs, the
scripts will automatically use the preprocessed data.

### Model Specific Setup

Make sure you have set up the general dependecies via `source setup_env.sh`
You should now at:
1. conda env: ipex
2. locate at /home/ModelBench

Then you can run
```
source bench_dlrm.sh
```
 The bench results are saved in `ModelBench/models/quickstart/language_modeling/pytorch/bert_large/inference/cpu`, named as `int8_bs32_onednn_graph_compiler.log`, or `fp32_bs32_onednn_primitives.log`, etc.

## More Details
Please refer to [link](https://github.com/IntelAI/models/blob/pytorch-r1.13-models/quickstart/recommendation/pytorch/dlrm/inference/cpu/README.md).

<!--- 80. License -->
## License

[LICENSE](/LICENSE)