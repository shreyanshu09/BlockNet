<div align="center">
    
# Unveiling the Power of Integration: Block Diagram Summarization through Local-Global Fusion

[![Paper](https://img.shields.io/badge/Paper-arxiv.2111.15664-red)](https://arxiv.org/abs/2111.15664)
[![Conference](https://img.shields.io/badge/ECCV-2022-blue)](#how-to-cite)
[![Demo](https://img.shields.io/badge/Demo-Gradio-brightgreen)](#demo)
[![Demo](https://img.shields.io/badge/Demo-Colab-orange)](#demo)
[![PyPI](https://img.shields.io/pypi/v/donut-python?color=green&label=pip%20install%20donut-python)](https://pypi.org/project/donut-python)
[![Downloads](https://static.pepy.tech/personalized-badge/donut-python?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/donut-python)

Official Implementation of BlockNet and BD-EnKo

</div>

## Introduction

**BlockNet** is a new framework that summarizes block diagrams, working for both English and Korean. It combines detailed (local) and overall (global) information from the diagrams using a large language model (LLM). For local information, it uses an Optical Character Recognition (OCR) algorithm that breaks the task into smaller parts. For global information, it uses a transformer model trained with a large block diagram dataset, without needing OCR.

**Donut** 🍩, **Do**cume**n**t **u**nderstanding **t**ransformer, is a new method of document understanding that utilizes an OCR-free end-to-end Transformer model. Donut does not require off-the-shelf OCR engines/APIs, yet it shows state-of-the-art performances on various visual document understanding tasks, such as visual document classification or information extraction (a.k.a. document parsing). 
In addition, we present **SynthDoG** 🐶, **Synth**etic **Do**cument **G**enerator, that helps the model pre-training to be flexible on various languages and domains.

<img width="946" alt="image" src="misc/overview.png">

## Pre-trained Models and Web Demos

Gradio web demos are available! [![Demo](https://img.shields.io/badge/Demo-Gradio-brightgreen)](#demo) [![Demo](https://img.shields.io/badge/Demo-Colab-orange)](#demo)
|:--:|
|![image](misc/screenshot_gradio_demos.png)|
- You can run the demo with `./app.py` file.
- Sample images are available at `./misc` and more receipt images are available at [CORD dataset link](https://huggingface.co/datasets/naver-clova-ix/cord-v2).
- Web demos are available from the links in the following table.
- Note: We have updated the Google Colab demo (as of June 15, 2023) to ensure its proper working.

|Task|Sec/Img|Score|Trained Model|<div id="demo">Demo</div>|
|---|---|---|---|---|
| [CORD](https://github.com/clovaai/cord) (Document Parsing)   |   0.7 /<br> 0.7 /<br> 1.2   |  91.3 /<br> 91.1 /<br> 90.9    | [donut-base-finetuned-cord-v2](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2/tree/official) (1280) /<br> [donut-base-finetuned-cord-v1](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v1/tree/official) (1280) /<br> [donut-base-finetuned-cord-v1-2560](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v1-2560/tree/official) | [gradio space web demo](https://huggingface.co/spaces/naver-clova-ix/donut-base-finetuned-cord-v2),<br>[google colab demo (updated at 23.06.15)](https://colab.research.google.com/drive/1NMSqoIZ_l39wyRD7yVjw2FIuU2aglzJi?usp=sharing) |
| [Train Ticket](https://github.com/beacandler/EATEN) (Document Parsing)   |   0.6   |  98.7    | [donut-base-finetuned-zhtrainticket](https://huggingface.co/naver-clova-ix/donut-base-finetuned-zhtrainticket/tree/official) | [google colab demo (updated at 23.06.15)](https://colab.research.google.com/drive/1YJBjllahdqNktXaBlq5ugPh1BCm8OsxI?usp=sharing) |
| [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip) (Document Classification)     |  0.75   |   95.3      | [donut-base-finetuned-rvlcdip](https://huggingface.co/naver-clova-ix/donut-base-finetuned-rvlcdip/tree/official) | [gradio space web demo](https://huggingface.co/spaces/nielsr/donut-rvlcdip),<br>[google colab demo (updated at 23.06.15)](https://colab.research.google.com/drive/1iWOZHvao1W5xva53upcri5V6oaWT-P0O?usp=sharing) |
| [DocVQA Task1](https://rrc.cvc.uab.es/?ch=17) (Document VQA) |  0.78       | 67.5 | [donut-base-finetuned-docvqa](https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa/tree/official) | [gradio space web demo](https://huggingface.co/spaces/nielsr/donut-docvqa),<br>[google colab demo (updated at 23.06.15)](https://colab.research.google.com/drive/1oKieslZCulFiquequ62eMGc-ZWgay4X3?usp=sharing) |

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

## Updates

**_2023-06-15_** We have updated all Google Colab demos to ensure its proper working.<br>
**_2022-11-14_** New version 1.0.9 is released (`pip install donut-python --upgrade`). See [1.0.9 Release Notes](https://github.com/clovaai/donut/releases/tag/1.0.9).<br>
**_2022-08-12_** Donut 🍩 is also available at [huggingface/transformers 🤗](https://huggingface.co/docs/transformers/main/en/model_doc/donut) (contributed by [@NielsRogge](https://github.com/NielsRogge)). `donut-python` loads the pre-trained weights from the `official` branch of the model repositories. See [1.0.5 Release Notes](https://github.com/clovaai/donut/releases/tag/1.0.5).<br>
**_2022-08-05_** A well-executed hands-on tutorial on donut 🍩 is published at [Towards Data Science](https://towardsdatascience.com/ocr-free-document-understanding-with-donut-1acfbdf099be) (written by [@estaudere](https://github.com/estaudere)).<br>
**_2024-05-30_** First Commit, We release our code, model, and dataset.

## Software installation

[![PyPI](https://img.shields.io/pypi/v/donut-python?color=green&label=pip%20install%20donut-python)](https://pypi.org/project/donut-python)
[![Downloads](https://static.pepy.tech/personalized-badge/donut-python?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/donut-python)

```bash
pip install donut-python
```

or clone this repository and install the dependencies:
```bash
git clone https://github.com/clovaai/donut.git
cd donut/
conda create -n donut_official python=3.7
conda activate donut_official
pip install .
```

We tested [donut-python](https://pypi.org/project/donut-python/1.0.1) == 1.0.1 with:
- [torch](https://github.com/pytorch/pytorch) == 1.11.0+cu113 
- [torchvision](https://github.com/pytorch/vision) == 0.12.0+cu113
- [pytorch-lightning](https://github.com/Lightning-AI/lightning) == 1.6.4
- [transformers](https://github.com/huggingface/transformers) == 4.11.3
- [timm](https://github.com/rwightman/pytorch-image-models) == 0.5.4

**Note**: From several reported issues, we have noticed increased challenges in configuring the testing environment for `donut-python` due to recent updates in key dependency libraries. While we are actively working on a solution, we have updated the Google Colab demo (as of June 15, 2023) to ensure its proper working. For assistance, we encourage you to refer to the following demo links: [CORD Colab Demo](https://colab.research.google.com/drive/1NMSqoIZ_l39wyRD7yVjw2FIuU2aglzJi?usp=sharing), [Train Ticket Colab Demo](https://colab.research.google.com/drive/1YJBjllahdqNktXaBlq5ugPh1BCm8OsxI?usp=sharing), [RVL-CDIP Colab Demo](https://colab.research.google.com/drive/1iWOZHvao1W5xva53upcri5V6oaWT-P0O?usp=sharing), [DocVQA Colab Demo](https://colab.research.google.com/drive/1oKieslZCulFiquequ62eMGc-ZWgay4X3?usp=sharing).

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

#### For Document Classification
The `gt_parse` follows the format of `{"class" : {class_name}}`, for example, `{"class" : "scientific_report"}` or `{"class" : "presentation"}`.
- Google colab demo is available [here](https://colab.research.google.com/drive/1xUDmLqlthx8A8rWKLMSLThZ7oeRJkDuU?usp=sharing).
- Gradio web demo is available [here](https://huggingface.co/spaces/nielsr/donut-rvlcdip).

#### For Document Information Extraction
The `gt_parse` is a JSON object that contains full information of the document image, for example, the JSON object for a receipt may look like `{"menu" : [{"nm": "ICE BLACKCOFFEE", "cnt": "2", ...}, ...], ...}`.
- More examples are available at [CORD dataset](https://huggingface.co/datasets/naver-clova-ix/cord-v2).
- Google colab demo is available [here](https://colab.research.google.com/drive/1o07hty-3OQTvGnc_7lgQFLvvKQuLjqiw?usp=sharing).
- Gradio web demo is available [here](https://huggingface.co/spaces/naver-clova-ix/donut-base-finetuned-cord-v2).

#### For Document Visual Question Answering
The `gt_parses` follows the format of `[{"question" : {question_sentence}, "answer" : {answer_candidate_1}}, {"question" : {question_sentence}, "answer" : {answer_candidate_2}}, ...]`, for example, `[{"question" : "what is the model name?", "answer" : "donut"}, {"question" : "what is the model name?", "answer" : "document understanding transformer"}]`.
- DocVQA Task1 has multiple answers, hence `gt_parses` should be a list of dictionary that contains a pair of question and answer.
- Google colab demo is available [here](https://colab.research.google.com/drive/1Z4WG8Wunj3HE0CERjt608ALSgSzRC9ig?usp=sharing).
- Gradio web demo is available [here](https://huggingface.co/spaces/nielsr/donut-docvqa).

#### For (Pseudo) Text Reading Task
The `gt_parse` looks like `{"text_sequence" : "word1 word2 word3 ... "}`
- This task is also a pre-training task of Donut model.
- You can use our **SynthDoG** 🐶 to generate synthetic images for the text reading task with proper `gt_parse`. See `./synthdog/README.md` for details.

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

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

```
