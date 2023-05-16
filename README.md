# DLFinalProject
The goal is to finetune the models RoBerta, Bert and DistilBert for sentiment classification tasks. By implementing the tweet_eval dataset and three pretrained models with transformer, we conclude that different language models can have varying levels of ability to interpret emotions.

## Prerequisites:
```
pip install transformers
pip install datasets
```
The suggested version of PyTorch is torch 1.13.1+cu116 on 2 RTX8000 (48GB)

# Executing the project:
For base base training
```
python main.py --batch_size 32
```
For improved training,
```
python main.py --batch_size 128 --DP --mixed
```
For sample testing,
```
python result_testing.py
```

## Pretrained model and dataset used:
* BERT, RoBERTa-base, DistilBERT
* "tweet_eval" dataset
