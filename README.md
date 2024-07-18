# Improving VQA Using MLLM
![image](https://github.com/pej0918/BLIVA/assets/79118751/d3de9fc7-cbda-4fb1-ba88-202ac09ee28f)


## Train

After downloading the training datasets and specify their path in [dataset configs](daiv/configs/datasets/), we are ready for training!

0. Setting Environments
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
if packaging error, then
```Shell
pip install setuptools==69.5.1
```

1. pretraining of Dm-Former
```Shell
python train.py --cfg-path train_configs/pretrain_stage1.yaml
```

2. Pretraining of visual assistant branch

```Shell
python train.py --cfg-path train_configs/pretrain_bliva_vicuna.yaml
```

3. Instruction Finetuning 

```Shell
python 
```

