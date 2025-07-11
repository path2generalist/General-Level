# GenBench Scoring System - User Guide

<div align="center">
<p><a href="README_ZH.md">中文</a> | <a href="README_EN.md">English</a></p>
</div>

---

This system is designed to evaluate large models' performance on the General-Bench multimodal task set. Users can complete prediction, scoring, and final score calculation with just one command.

## Environment Setup

- Python 3.9 or higher
- Recommended to install dependencies in advance (such as pandas, numpy, openpyxl, etc.)
- For Video Generation evaluation, follow the steps in video_generation_evaluation/README.md to install dependencies
- For Video Comprehension evaluation, follow the steps in [sa2va](https://github.com/magic-research/Sa2VA) README.md to install dependencies

## Dataset Download

- **Open Set (Public Dataset)**: Please download all data from [HuggingFace General-Bench-Openset](https://huggingface.co/datasets/General-Level/General-Bench-Openset), extract and place in the `General-Bench-Openset/` directory.
- **Close Set (Private Dataset)**: Please download all data from [HuggingFace General-Bench-Closeset](https://huggingface.co/datasets/General-Level/General-Bench-Closeset), extract and place in the `General-Bench-Closeset/` directory.

## One-Click Run

Simply run the main script `run.sh` to complete the entire process:

```bash
bash run.sh
```

This command will sequentially complete:
1. Generate predictions for all modalities
2. Calculate scores for each task
3. Calculate final Level scores

## Step-by-Step Run (Optional)

If you only need to run specific steps, use the `--step` parameter:

- Run only step 1 (generate predictions):
  ```bash
  bash run.sh --step 1
  ```
- Run only steps 1 and 2:
  ```bash
  bash run.sh --step 12
  ```
- Run only steps 2 and 3:
  ```bash
  bash run.sh --step 23
  ```
- No parameter defaults to executing all steps (equivalent to `--step 123`)

- Step 1: Generate prediction results (prediction.json), stored in the same directory as annotation.json for each dataset
- Step 2: Calculate scores for each task, stored in outcome/{model_name}_result.xlsx
- Step 3: Calculate the final Level scores for the relevant models

> **Note:**
> - When using **Close Set (Private Dataset)**, only run step 1 (i.e., `bash run.sh --step 1`) and submit the generated prediction.json to the system.
> - When using **Open Set (Public Dataset)**, run steps 1, 2, and 3 sequentially (i.e., `bash run.sh --step 123`) to complete the entire evaluation process.

## Results Viewing

- Prediction results (prediction.json) will be output to the corresponding dataset folder for each task, at the same level as annotation.json.
- Scoring results (such as Qwen2.5-7B-Instruct_result.xlsx) will be output to the outcome/ directory.
- Final Level scores will be printed directly in the terminal.

## Directory Description

- `General-Bench-Openset/`: Public dataset directory
- `General-Bench-Closeset/`: Private dataset directory
- `outcome/`: Output results directory
- `references/`: Reference template directory
- `run.sh`: Main run script (recommended for users to use only this script)

## Common Issues

- If dependencies are missing, please install the corresponding Python packages according to the error messages.
- If you need to customize model or data paths, you can edit the relevant variables in the `run.sh` script.

---

For further assistance, please contact the system maintainers or refer to the detailed development documentation. 