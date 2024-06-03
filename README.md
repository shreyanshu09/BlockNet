<div align="center">
    
# Unveiling the Power of Integration: Block Diagram Summarization through Local-Global Fusion

[![Conference](https://img.shields.io/badge/ACL-2024-blue)](#how-to-cite)
[![Demo](https://img.shields.io/badge/Demo-Gradio-brightgreen)](#demo)

Official Implementation of BlockNet Model

</div>

## Introduction

BlockNet is an advanced framework designed to summarize block diagrams by integrating local and global information for both English and Korean languages. It employs an OCR-based algorithm that follows a divide-and-conquer principle to extract detailed local information from smaller sections of block diagrams. For global information extraction, it utilizes an OCR-free transformer architecture trained with the BD-EnKo dataset and public data, capturing the overall structure and relationships within the diagrams. By leveraging Large Language Models (LLMs), BlockNet seamlessly synthesizes these detailed and holistic insights to produce coherent and contextually accurate summaries.

<img width="946" alt="image" src="misc/overview.png">

The official implementation of BD-EnKo Dataset and Generator is available in this [GitHub](https://github.com/shreyanshu09/BD-EnKo) repository.

## Pre-trained Models and Web Demos

The links to the pre-trained backbones are here:
- [`donut-base`](https://huggingface.co/naver-clova-ix/donut-base/tree/official): trained with 64 A100 GPUs (~2.5 days), number of layers (encoder: {2,2,14,2}, decoder: 4), input size 2560x1920, swin window size 10, IIT-CDIP (11M) and SynthDoG (English, Chinese, Japanese, Korean, 0.5M x 4).
- [`donut-proto`](https://huggingface.co/naver-clova-ix/donut-proto/tree/official): (preliminary model) trained with 8 V100 GPUs (~5 days), number of layers (encoder: {2,2,18,2}, decoder: 4), input size 2048x1536, swin window size 8, and SynthDoG (English, Japanese, Korean, 0.4M x 3).

Please see [our paper](#how-to-cite) for more details.

## SynthDoG datasets

![image](misc/sample_synthdog.png)

The links to the SynthDoG-generated datasets are here:

- [`synthdog-en`](https://huggingface.co/datasets/naver-clova-ix/synthdog-en): English, 0.5M.
- [`synthdog-zh`](https://huggingface.co/datasets/naver-clova-ix/synthdog-zh): Chinese, 0.5M.
- [`synthdog-ja`](https://huggingface.co/datasets/naver-clova-ix/synthdog-ja): Japanese, 0.5M.
- [`synthdog-ko`](https://huggingface.co/datasets/naver-clova-ix/synthdog-ko): Korean, 0.5M.

To generate synthetic datasets with our SynthDoG, please see `./synthdog/README.md` and [our paper](#how-to-cite) for details.

## Software installation

Clone this repository and install the dependencies:
```bash
git clone https://github.com/shreyanshu09/BlockNet.git
cd BlockNet/
conda create -n blocknet python=3.9
conda activate blocknet
pip install -r requirements.txt
```

We tested with CUDA (11.8):
- [torch](https://github.com/pytorch/pytorch) == 1.12.1 
- [torchvision](https://github.com/pytorch/vision) == 0.13.1
- [pytorch-lightning](https://github.com/Lightning-AI/lightning) == 2.1.3
- [transformers](https://github.com/huggingface/transformers) == 4.21.1
- [timm](https://github.com/rwightman/pytorch-image-models) == 0.5.4


## Getting Started

### Data

This repository assumes the following structure of dataset:
```bash
> tree dataset_name
dataset_name
├── test
│   ├── metadata.jsonl
│   ├── {image_path0}
│   ├── {image_path1}
│             .
│             .
├── train
│   ├── metadata.jsonl
│   ├── {image_path0}
│   ├── {image_path1}
│             .
│             .
└── validation
    ├── metadata.jsonl
    ├── {image_path0}
    ├── {image_path1}
              .
              .

> cat dataset_name/test/metadata.jsonl
{"file_name": {image_path0}, "ground_truth": "{\"gt_parse\": {ground_truth_parse}, ... {other_metadata_not_used} ... }"}
{"file_name": {image_path1}, "ground_truth": "{\"gt_parse\": {ground_truth_parse}, ... {other_metadata_not_used} ... }"}
     .
     .
```

- The structure of `metadata.jsonl` file is in [JSON Lines text format](https://jsonlines.org), i.e., `.jsonl`. Each line consists of
  - `file_name` : relative path to the image file.
  - `ground_truth` : string format (json dumped), the dictionary contains either `gt_parse` or `gt_parses`. Other fields (metadata) can be added to the dictionary but will not be used.
- `donut` interprets all tasks as a JSON prediction problem. As a result, all `donut` model training share a same pipeline. For training and inference, the only thing to do is preparing `gt_parse` or `gt_parses` for the task in format described below.


### Training

This is the configuration of Donut model training on [CORD](https://github.com/clovaai/cord) dataset used in our experiment. 
We ran this with a single NVIDIA A100 GPU.

```bash
python train.py --config config/train_cord.yaml \
                --pretrained_model_name_or_path "naver-clova-ix/donut-base" \
                --dataset_name_or_paths '["naver-clova-ix/cord-v2"]' \
                --exp_version "test_experiment"    
  .
  .                                                                                                                                                                                                                                         
Prediction: <s_menu><s_nm>Lemon Tea (L)</s_nm><s_cnt>1</s_cnt><s_price>25.000</s_price></s_menu><s_total><s_total_price>25.000</s_total_price><s_cashprice>30.000</s_cashprice><s_changeprice>5.000</s_changeprice></s_total>
Answer: <s_menu><s_nm>Lemon Tea (L)</s_nm><s_cnt>1</s_cnt><s_price>25.000</s_price></s_menu><s_total><s_total_price>25.000</s_total_price><s_cashprice>30.000</s_cashprice><s_changeprice>5.000</s_changeprice></s_total>
Normed ED: 0.0
Prediction: <s_menu><s_nm>Hulk Topper Package</s_nm><s_cnt>1</s_cnt><s_price>100.000</s_price></s_menu><s_total><s_total_price>100.000</s_total_price><s_cashprice>100.000</s_cashprice><s_changeprice>0</s_changeprice></s_total>
Answer: <s_menu><s_nm>Hulk Topper Package</s_nm><s_cnt>1</s_cnt><s_price>100.000</s_price></s_menu><s_total><s_total_price>100.000</s_total_price><s_cashprice>100.000</s_cashprice><s_changeprice>0</s_changeprice></s_total>
Normed ED: 0.0
Prediction: <s_menu><s_nm>Giant Squid</s_nm><s_cnt>x 1</s_cnt><s_price>Rp. 39.000</s_price><s_sub><s_nm>C.Finishing - Cut</s_nm><s_price>Rp. 0</s_price><sep/><s_nm>B.Spicy Level - Extreme Hot Rp. 0</s_price></s_sub><sep/><s_nm>A.Flavour - Salt & Pepper</s_nm><s_price>Rp. 0</s_price></s_sub></s_menu><s_sub_total><s_subtotal_price>Rp. 39.000</s_subtotal_price></s_sub_total><s_total><s_total_price>Rp. 39.000</s_total_price><s_cashprice>Rp. 50.000</s_cashprice><s_changeprice>Rp. 11.000</s_changeprice></s_total>
Answer: <s_menu><s_nm>Giant Squid</s_nm><s_cnt>x1</s_cnt><s_price>Rp. 39.000</s_price><s_sub><s_nm>C.Finishing - Cut</s_nm><s_price>Rp. 0</s_price><sep/><s_nm>B.Spicy Level - Extreme Hot</s_nm><s_price>Rp. 0</s_price><sep/><s_nm>A.Flavour- Salt & Pepper</s_nm><s_price>Rp. 0</s_price></s_sub></s_menu><s_sub_total><s_subtotal_price>Rp. 39.000</s_subtotal_price></s_sub_total><s_total><s_total_price>Rp. 39.000</s_total_price><s_cashprice>Rp. 50.000</s_cashprice><s_changeprice>Rp. 11.000</s_changeprice></s_total>
Normed ED: 0.039603960396039604                                                                                                                                  
Epoch 29: 100%|█████████████| 200/200 [01:49<00:00,  1.82it/s, loss=0.00327, exp_name=train_cord, exp_version=test_experiment]
```

Some important arguments:

- `--config` : config file path for model training.
- `--pretrained_model_name_or_path` : string format, model name in Hugging Face modelhub or local path.
- `--dataset_name_or_paths` : string format (json dumped), list of dataset names in Hugging Face datasets or local paths.
- `--result_path` : file path to save model outputs/artifacts.
- `--exp_version` : used for experiment versioning. The output files are saved at `{result_path}/{exp_version}/*`

### Test

With the trained model, test images and ground truth parses, you can get inference results and accuracy scores.

```bash
python test.py --dataset_name_or_path naver-clova-ix/cord-v2 --pretrained_model_name_or_path ./result/train_cord/test_experiment --save_path ./result/output.json
100%|█████████████| 100/100 [00:35<00:00,  2.80it/s]
Total number of samples: 100, Tree Edit Distance (TED) based accuracy score: 0.9129639764131697, F1 accuracy score: 0.8406020841373987
```

Some important arguments:

- `--dataset_name_or_path` : string format, the target dataset name in Hugging Face datasets or local path.
- `--pretrained_model_name_or_path` : string format, the model name in Hugging Face modelhub or local path.
- `--save_path`: file path to save predictions and scores.


## License

The content of this project is licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
