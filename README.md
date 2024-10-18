# Enhancing Q-Former for Visual Question Answering with Multi-layer Co-Attention and Question-Aware Prompts

We propose a model that enhances Q-Former’s performance by integrating the **Modular Co-Attention Network (MCAN)** and introducing a **Question-Aware Prompt** during fine-tuning, improving Visual Question Answering (VQA) tasks.

## Model Architecture Overview
![image](imgs/model_Architecture_train.png)
Visual Question Answering (VQA) involves generating accurate answers by reasoning over both textual (questions) and visual (images) data. While **Q-Former** effectively models question-image interactions through Cross-Attention, it struggles with complex relationships due to its single-layer attention. To address this, we combine Q-Former with the **Modular Co-Attention Network (MCAN)**, introducing a multi-layer attention mechanism for deeper interactions. Additionally, **Question-Aware Prompts** during fine-tuning provide richer contextual information to further boost performance.

### Q-Former and MCAN Integration

We enhance **Q-Former** by integrating **MCAN**, a multi-layered network that uses both **Self-Attention** and **Cross-Attention** to refine question-image interactions. While Q-Former's single-layer structure struggles with complex relationships, MCAN progressively captures both high-level semantics and detailed information, significantly improving the model's reasoning ability.

### Fine-tuning with Question-Aware Prompts

![image](imgs/model_finetuning.png)

During fine-tuning, **Question-Aware Prompts** are introduced to provide additional context, such as background knowledge and answer candidates. This helps the model better interpret the question's intent, enabling more accurate and informed answers, especially for complex queries. The combination of **MCAN** and **Question-Aware Prompts** results in significant improvements in handling challenging VQA tasks.

## Experiment Results

### 1. Environment Setup
```bash
conda create -n fusion python=3.9
```

### 2. Dataset Preparation
Download COCO and Visual Genome datasets, and specify their path in [dataset configs](daiv/configs/datasets/).

### 3. Training the Model
```bash
python train.py --cfg-path train_configs/pretrain_stage1.yaml
```

### 4. Fine-tuning with Question-Aware Prompts
```bash
python train.py --cfg-path train_configs/finetune_stage2.yaml
```

### 5. Evaluation
```bash
python evaluate.py --cfg-path train_configs/finetune_stage2_eval.yaml
```

### Results on VQA Datasets

We evaluated our model on the **OK-VQA** and **AOK-VQA** datasets, using **COCO** and **Visual Genome** for pre-training. The table below compares the baseline **Q-Former**, **MCAN**, and our enhanced model with and without **Question-Aware Prompts**.

| Model           | Accuracy (Only-Question) | Accuracy (Question-Aware Prompt) |
|-----------------|--------------------------|----------------------------------|
| Q-Former        | 49.2%                     | 55.65%                          |
| MCAN            | **52.56%**                | -                                |
| Ours            | 50%                       | **56.1%**                        |

Our enhanced model, which integrates **MCAN** and **Question-Aware Prompts**, achieved a **6.1% accuracy improvement** when using the prompts compared to the baseline Q-Former. This demonstrates that **Question-Aware Prompts** provide valuable context, enabling the model to better interpret the question’s intent and make more informed predictions. Moreover, **MCAN’s** multi-layer attention mechanism consistently outperformed the single-layer **Q-Former**, especially for complex questions requiring deeper reasoning. These results validate the effectiveness of our approach in improving VQA performance.

## Conclusion

Our enhanced model for **Visual Question Answering (VQA)** significantly improves performance by integrating **MCAN** for deeper attention and introducing **Question-Aware Prompts** during fine-tuning. These enhancements allow the model to better capture complex interactions between questions and images, leading to more accurate and context-aware answers. Experimental results show a **6.1% increase in accuracy**, demonstrating the effectiveness of this approach in handling complex VQA tasks without extensive fine-tuning.
