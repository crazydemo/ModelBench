<!--- 0. Title -->
# PyTorch DLRM inference

<!-- 10. Description -->
## Description

This document has instructions for running DLRM inference using
Intel-optimized PyTorch for bare metal.

## Bare Metal

### Model Specific Setup

Make sure you have set up the general dependecies via `source setup_env.sh`
You should now at:
1. conda env: ipex
2. locate at /home/ModelBench

Then you can run
```
source setup_bert_env.sh
```

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
export DATASET_DIR=<dir/to/save/dlrm_data>

mkdir ${DATASET_DIR} && cd ${DATASET_DIR}
curl -O <download url>/day_{$(seq -s , 0 23)}.gz
gunzip day_*.gz
```
The raw data will be automatically preprocessed and saved as `day_*.npz` to
the `DATASET_DIR` when DLRM is run for the first time. On subsequent runs, the
scripts will automatically use the preprocessed data.

## Quick Start Scripts

```bash

# Env vars
export PRECISION=<specify the precision to run>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
```

| Script name | Description |
|-------------|-------------|
| `inference_performance.sh` | Run inference to verify performance for the specified precision (fp32, bf32, bf16, or int8). |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (fp32, bf32, bf16, or int8). |

## More Details
Please refer to [link](https://github.com/IntelAI/models/blob/pytorch-r1.13-models/quickstart/recommendation/pytorch/dlrm/inference/cpu/README.md).

<!--- 80. License -->
## License

[LICENSE](/LICENSE)