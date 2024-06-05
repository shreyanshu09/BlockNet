<div align="center">
    
# Unveiling the Power of Integration: Block Diagram Summarization through Local-Global Fusion

[![Conference](https://img.shields.io/badge/ACL-2024-blue)](#how-to-cite)
[![Demo](https://img.shields.io/badge/Demo-Gradio-brightgreen)](#demo)

**Official Implementation of BlockNet Model**
</div>

## Introduction

BlockNet is an advanced framework designed to summarize block diagrams by integrating local and global information for both English and Korean languages. It employs an OCR-based algorithm that follows a divide-and-conquer principle to extract detailed local information from smaller sections of block diagrams. For global information extraction, it utilizes an OCR-free transformer architecture trained with the BD-EnKo dataset and public data, capturing the overall structure and relationships within the diagrams. By leveraging Large Language Models (LLMs), BlockNet seamlessly synthesizes these detailed and holistic insights to produce coherent and contextually accurate summaries.

<p align="center">
  <img src="misc/blocknet.png" alt="blocknet" />
</p>

## Pre-trained Models

The links to the pre-trained backbones are here:
- [`local_information_extractor`](https://huggingface.co/shreyanshu09/block_diagram_symbol_detection): This model is trained using an object detection model based on YOLOv5, which offers essential capabilities for detecting various objects in an image. Using the CBD, FCA, and FCB dataset, which includes annotations for different shapes and arrows in a diagram, we train the model to recognize six labels: arrow, terminator, process, decision, data, and text.

- [`global_information_extractor`](https://huggingface.co/shreyanshu09/block_diagram_global_information): This model is trained using a transformer encoder and decoder architecture, based on the configuration specified in [Donut](https://arxiv.org/abs/2111.15664), to extract the overall summary of block diagram images. It supports both English and Korean languages. The straightforward architecture comprises a visual encoder module and a text decoder module, both based on the Transformer architecture.

## Datasets

Sample data from the BD-EnKo dataset is available [here](https://github.com/shreyanshu09/BlockNet/tree/main/BD-EnKo_dataset_sample).

The official implementation of BD-EnKo Dataset and Generator is available in this [GitHub](https://github.com/shreyanshu09/BD-EnKo) repository.


<p align="center">
  <img src="misc/dataset_sample.png" alt="dataset_sample" />
</p>

The links to the datasets are here:

- [`BD-EnKo`](https://huggingface.co/datasets/shreyanshu09/BD-EnKo): 83,394 samples.
- [`Complete Dataset`](https://huggingface.co/datasets/shreyanshu09/Block_Diagram): 84,925 samples.

To generate synthetic datasets with our method, please check [here](https://github.com/shreyanshu09/BD-EnKo) for details.

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

> cat dataset_name/train/metadata.jsonl
{"file_name": {image_path0}, "ground_truth": "{\"gt_parse\": {\"c2t\": \"{ground_truth_parse}\"}}"}
{"file_name": {image_path1}, "ground_truth": "{\"gt_parse\": {\"c2t\": \"{ground_truth_parse}\"}}"}
     .
     .
```

- The structure of `metadata.jsonl` file is in [JSON Lines text format](https://jsonlines.org), i.e., `.jsonl`. Each line consists of
  - `file_name` : relative path to the image file.
  - `ground_truth` : string format (json dumped), the dictionary contains `gt_parse`.

### Training

*Coming soon*


### Inference

Download the pre-trained models and place them in their respective folders:
- [`local_information_extractor`](https://huggingface.co/shreyanshu09/block_diagram_symbol_detection) inside `local_model/block_diagram_symbol_detection`
- [`global_information_extractor`](https://huggingface.co/shreyanshu09/block_diagram_global_information) inside `global_model/block_diagram_global_information`

There are different ways to run the models:

1. **Python file (.py)**:
   - Change the paths in `main.py`
   - Run the script: `python main.py`

2. **Jupyter Notebook (.ipynb)**:
   - The `all_in_one.ipynb` file contains all the code within a single Jupyter notebook and allows for testing within the notebook.
   - Simply change all the paths and run the cells.

3. **Gradio App**:
   - Open the `app.ipynb` or `app.py` file and update the file paths.
   - Finally, run all the cells.
   - 

## Contact

If you have any questions about this work, please contact **[Shreyanshu Bhushan](https://github.com/shreyanshu09)** using the following email addresses: **shreyanshubhushan@gmail.com**.


## License

The content of this project itself is licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
