import torch
import pandas as pd

data = pd.read_csv("ratings_test.txt", sep='\t', engine='python' ,encoding="utf-8", names=["Index", "data", "label"])

print(data.head(10))