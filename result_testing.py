from DataLoad import dataloader
import torch
import os
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
#from scipy.special import softmax
import csv
import urllib.request
device = "cuda"

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Get tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# load the task model
model = torch.load("emotion.pt").cuda()
# Preprocess sentence
text = "I love you"
text = preprocess(text)
encoded_input = tokenizer(text, padding="max_length", truncation=True)
inputTen = torch.Tensor([encoded_input['input_ids']]).to(torch.int64).to(device)
maskTen = torch.Tensor([encoded_input['attention_mask']]).to(torch.int64).to(device)
# Generate output
output = model(inputTen,maskTen).logits
output = torch.nn.functional.softmax(output, dim=-1)
print("Emotion probability is:")
print(output.to("cpu").tolist())
