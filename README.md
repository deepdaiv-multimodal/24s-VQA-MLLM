# Enhancing Q-Former for Knowledge-Based Visual Question Answering with Multi-Layer Co-Attention and Question-Aware Prompts

We propose a novel approach to improve Knowledge-Based Visual Question Answering (KB-VQA) by integrating **Q-Former** with the **Multimodal Co-Attention Network (MCAN)** and introducing **Question-Aware Prompts** during fine-tuning. This enhances the model's ability to handle complex multimodal reasoning tasks, requiring both visual and external knowledge.

## Model Overview
![image](https://github.com/user-attachments/assets/f3f6f41f-1ac4-43b8-b4d7-f99369a655ea)

In **Knowledge-Based Visual Question Answering (KB-VQA)**, generating accurate answers requires the model to reason over both visual data (images) and external knowledge sources. Our approach builds upon **Q-Former**, a model known for extracting visual features using Cross-Attention, by addressing its limitations in handling complex question-image relationships with a single-layer attention mechanism. By integrating **MCAN**'s multi-layered attention, we capture deeper interactions between the image and the question. Additionally, **Question-Aware Prompts** during fine-tuning allow the model to incorporate more context, improving overall reasoning and answer accuracy.

### Q-Former and MCAN Integration

Our approach enhances **Q-Former** by combining it with **MCAN**, a multi-layered attention network that applies **Self-Attention** and **Cross-Attention** to better capture the intricate relationships between images and questions. **MCAN** progressively refines both visual and textual inputs, leading to a significant improvement in the model's reasoning capabilities, especially for questions that require external knowledge and complex reasoning.

### Fine-Tuning with Question-Aware Prompts
![image](imgs/model_finetuning.png)

We introduce **Question-Aware Prompts** during fine-tuning to provide additional contextual information, such as past answers (**Answer-Aware Examples**) and candidate answers (**Answer Candidates**). These prompts help the model better interpret the intent of the question and select more accurate answers, especially for complex or knowledge-dependent queries.

#### Question-Aware Prompt Structure
![image](https://github.com/user-attachments/assets/53953d81-aa90-48cf-83c7-257e7d3eed87)

The **Question-Aware Prompt** structure provides a framework for the model to combine context from multiple sources. It includes:
- **Answer Candidates**: A list of possible answers, each with a confidence score.
- **Answer-Aware Examples**: Previous cases with similar questions, answers, and contexts.

The prompt enables the model to integrate background knowledge, helping the model generate answers that are more contextually relevant and informed.

## Experiment Results

### 1. Environment Setup
```bash
conda create -n kbvqa python=3.9
```

### 2. Dataset Preparation
Download COCO, OK-VQA, and AOK-VQA datasets. Specify their paths in [dataset configs](daiv/configs/datasets/).

### 3. Training the Model
```bash
python train.py --cfg-path train_configs/pretrain_stage1.yaml
```

### 4. Fine-Tuning with Question-Aware Prompts
```bash
python train.py --cfg-path train_configs/finetune_stage2.yaml
```

### 5. Evaluation
```bash
python evaluate.py --cfg-path train_configs/finetune_stage2_eval.yaml
```

### Results on OK-VQA and AOK-VQA Datasets

We evaluated our model on the **OK-VQA** and **AOK-VQA** datasets. The table below shows a comparison of the baseline **Q-Former**, **MCAN**, and our enhanced model with and without Our **Question-Aware Prompts**.

| Model           | Accuracy (Only-Question) | Accuracy (Question-Aware Prompt) |
|-----------------|--------------------------|----------------------------------|
| Q-Former        | 49.2%                     | 55.65%                          |
| MCAN            | **52.56%**                | -                                |
| Ours            | 50%                       | **56.1%**                        |

Our model, enhanced with **MCAN** and **Question-Aware Prompts**, shows a **6.9% improvement in accuracy** compared to the baseline **Q-Former**. This demonstrates the effectiveness of multi-layer attention and prompt-based fine-tuning for handling complex VQA tasks.

## Conclusion

By integrating **MCAN** with **Q-Former** and introducing **Question-Aware Prompts**, our model significantly improves its ability to reason over complex question-image relationships and utilize external knowledge sources for more accurate answers. The results demonstrate a **6.9% increase in accuracy**, validating the effectiveness of our approach in enhancing performance for Knowledge-Based Visual Question Answering (KB-VQA) tasks.
