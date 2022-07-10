from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split

from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('kykim/bert-kor-base')
model = BertForSequenceClassification.from_pretrained('kykim/bert-kor-base', num_labels = 2).to('cuda')

# Data Load
class data(Dataset) :
    def __init__(self) :
        data = pd.read_csv("ratings_test.txt", sep='\t', engine='python' ,encoding="utf-8")
        self.len = data.shape[0]
        self.comment = data['document']
        self.labels = data['label']
    
    def __getitem__(self, index):
        return self.comment[index], self.labels[index]

    def __len__(self) :
        return self.len

dataset = data()
total_size = len(dataset)
train_size = int(total_size * 0.8)
test_size = int(total_size - train_size )

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True, drop_last = True)
test_dataloader = DataLoader(test_dataset, batch_size = 16, shuffle = True, drop_last = True)

# Model
inputs = tokenizer(["이 영화 너무 재밌는 것 같아요!!", "이 영화 왜 만든거임?"], return_tensors = "pt", padding=True).to('cuda')
labels = torch.tensor([1, 0]).to('cuda')

outputs = model(**inputs, labels=labels)