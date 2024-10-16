# Multimodal Learning with Q-Former and MCAN for Visual Question Answering

This repository contains the implementation of **"Multimodal Learning with Q-Former and MCAN for Visual Question Answering"**. Our proposed model addresses the limitations of Q-Former by integrating the **Multimodal Co-Attention Network (MCAN)** and introducing a **Question-Aware Prompt** during the fine-tuning process to improve the model's performance on Visual Question Answering (VQA) tasks.

## Introduction

Visual Question Answering (VQA) is a challenging task that requires a model to understand and reason over both textual (questions) and visual (images) information to generate accurate answers. **Q-Former** has been widely used for VQA, utilizing **Cross-Attention** to model the interaction between questions and images. However, Q-Former’s **single-layer attention mechanism** struggles to capture complex and detailed relationships, limiting its performance in tasks requiring deeper reasoning.

To overcome these limitations, we propose a model that integrates **Q-Former** with the **Multimodal Co-Attention Network (MCAN)**, a multi-layer attention mechanism capable of capturing deeper interactions between questions and images. Furthermore, during the fine-tuning phase, we introduce a **Question-Aware Prompt** that provides additional context to the questions, further enhancing the model’s understanding and performance.

## Methodology

![image](imgs/model_Architecture_train.png)

Our approach builds upon the strengths of Q-Former while addressing its limitations through the integration of MCAN and Question-Aware Prompting.

### Q-Former and MCAN Integration

The base architecture uses **Q-Former** to model the initial interaction between the question and the image via **Cross-Attention**. While this effectively handles basic interactions, more complex relationships are not fully captured. To address this, we integrated **MCAN**, which employs multiple layers of **Self-Attention** and **Cross-Attention** to progressively refine the question-image interaction. This multi-layered approach allows the model to capture both high-level relationships and fine-grained details, improving overall reasoning capability.

### Fine-tuning with Question-Aware Prompts

![image](imgs/model_finetuning.png)

During fine-tuning, we enhance the model’s ability to comprehend the question by incorporating **Question-Aware Prompts**. These prompts provide additional background knowledge or possible answer candidates, allowing the model to better understand the question's context. This leads to improved performance, especially on complex questions where deeper reasoning is required.

<details>
  <summary>Train & Eval</summary>
  
  ## Training & Inference
  
  ### Train
  After downloading the training datasets and specifying their path in [dataset configs](daiv/configs/datasets/), we are ready for training!
  
  #### 0. Setting Environments
  ```Shell
  conda create -n fusion python=3.9
  ```
  ```Shell
  git clone 
  ```
  ```Shell
  cd BLIVA
  ```
  ```Shell
  pip install -e .
  ```
  if packaging error occurs, then:
  ```Shell
  pip install setuptools==69.5.1
  ```

  ### Training
  
  #### 1. Pretraining of Dm-Former
  ```Shell
  python train.py --cfg-path train_configs/pretrain_stage1.yaml
  ```
  #### 2. Pretraining of visual assistant branch
  
  ```Shell
  python train.py --cfg-path train_configs/pretrain_stage2.yaml
  ```
  #### 3. Instruction Finetuning 
  ```Shell
  python train.py --cfg-path train_configs/finetune_stage2.yaml
  ```
  ### Evaluation
  
  #### Evaluation of Stage2 
  ```Shell
  python evaluate.py --cfg-path train_configs/pretrain_stage2_eval.yaml
  ```
  
  ```Shell
  python evaluate.py --cfg-path train_configs/finetune_stage2_eval.yaml
  ```
  
  #### Training with MCAN output (prophet) - okvqa
  ```Shell
  python train.py --cfg-path train_configs/finetune_stage2_t5_vqa.yaml
  ```
  ```Shell
  python evaluate.py --cfg-path train_configs/eval_stage2_vqa.yaml
  ```

</details>

## Results

We evaluated our model on standard VQA datasets such as **OK-VQA** and **AOK-VQA**, with pre-training performed on **COCO** and **Visual Genome** datasets. The following table presents the accuracy results comparing different models and the impact of incorporating **Question-Aware Prompts**.

| Model           | Accuracy (Only-Question) | Accuracy (Question-Aware Prompt) |
|-----------------|--------------------------|----------------------------------|
| InstructBLIP    | 49.2%                     | 55.65%                          |
| MCAN            | 52.56%                    | -                                |
| Ours            | 50%                       | 56.1%                           |

### Results Analysis

The results demonstrate that integrating **MCAN** and utilizing **Question-Aware Prompts** significantly improves performance on VQA tasks. Our model achieved a **6.1% increase in accuracy** compared to the baseline (Q-Former with only questions). This improvement highlights the effectiveness of **Question-Aware Prompts**, which provide valuable context, helping the model better understand and reason about the question. Additionally, **MCAN**’s multi-layered attention mechanism outperforms the single-layer **Q-Former**, effectively capturing complex interactions between the question and image, and leading to more accurate answers.
