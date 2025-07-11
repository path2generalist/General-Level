# GenBench 评分系统 - 用户使用说明

<div align="center">
<p><a href="README_EN.md">English</a> | <a href="README_ZH.md">中文</a></p>
</div>

---

本系统用于评估大模型在 General-Bench 多模态任务集上的表现。用户只需一条命令即可完成预测、评分和最终得分计算。

## 环境准备

- Python 3.9 及以上
- 推荐提前安装依赖（如 pandas, numpy, openpyxl 等）
- Video Generation评测，需要按照video_generation_evaluation/README.md中的步骤安装依赖
- Video Comprehension评测，需要按照[sa2va](https://github.com/magic-research/Sa2VA)中的README.md中的步骤安装依赖。

## 数据集下载

- **Open Set（公开数据集）**：请从 [HuggingFace General-Bench-Openset](https://huggingface.co/datasets/General-Level/General-Bench-Openset) 下载全部数据，解压后放入 `General-Bench-Openset/` 目录。
- **Close Set（私有数据集）**：请从 [HuggingFace General-Bench-Closeset](https://huggingface.co/datasets/General-Level/General-Bench-Closeset) 下载全部数据，解压后放入 `General-Bench-Closeset/` 目录。

## 一键运行

请直接运行主脚本 `run.sh`，即可完成全部流程：

```bash
bash run.sh
```

该命令将依次完成：
1. 生成各模态预测结果
2. 计算各任务得分
3. 计算最终 Level 得分

## 分步运行（可选）

如只需运行部分步骤，可使用 `--step` 参数：

- 只运行第1步（生成预测）：
  ```bash
  bash run.sh --step 1
  ```
- 只运行第1、2步：
  ```bash
  bash run.sh --step 12
  ```
- 只运行第2、3步：
  ```bash
  bash run.sh --step 23
  ```
- 不加参数默认全部执行（等价于 `--step 123`）

- 步骤1：生成预测结果prediction.json，存在每一个数据集的annotation.json同级目录下
- 步骤2：计算每个任务的得分，存在outcome/{model_name}_result.xlsx中
- 步骤3：计算相关模型的Level得分

> **注意：**
> - 使用 **Close Set（私有数据集）** 时，只需运行 step1（即 `bash run.sh --step 1`），并将生成的 prediction.json 提交到系统。
> - 使用 **Open Set（公开数据集）** 时，需依次运行 step1、step2、step3（即 `bash run.sh --step 123`），完成全部评测流程。

## 结果查看

- 预测结果（prediction.json）会输出到每个任务对应的数据集文件夹下，与 annotation.json 同级。
- 评分结果（如 Qwen2.5-7B-Instruct_result.xlsx）会输出到 outcome/ 目录。
- 最终 Level 得分会直接在终端打印输出。

## 目录说明

- `General-Bench-Openset/`：公开数据集目录
- `General-Bench-Closeset/`：私有数据集目录
- `outcome/`：输出结果目录
- `references/`：参考模板目录
- `run.sh`：主运行脚本（推荐用户只用此脚本）

## 常见问题

- 如遇依赖缺失，请根据报错信息安装相应 Python 包。
- 如需自定义模型或数据路径，可编辑 `run.sh` 脚本中的相关变量。

---

如需进一步帮助，请联系系统维护者或查阅详细开发文档。 