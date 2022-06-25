import io
from lib2to3.pgen2.tokenize import tokenize
import os
import time
from tokenize import Token
import unicodedata
import re
from venv import create

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# kor-eng text path
# English - <Tab> - Korean - <Tab> - ETC information
path_to_file=os.path.dirname(__file__)+"/kor-eng/kor.txt"

total_file = io.open(path_to_file, encoding='UTF-8').read().strip().split('\n')

# Meaningless?
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


# data-processing
# 1. add space between ?, !, (, ), ., ,
# 2. Eliminate except either English or Korean 

def string_processing(str) : 
    str = str.lower().strip()

    str = re.sub(r'([?.!,¿])', r" \1 ", str)
    str = re.sub(r'[" "]+', " ", str)

    # Korean / English / ?.!, regex
    str = re.sub(r'[^\u3131-\u3163\uac00-\ud7a3a-zA-Z0-9?.!,]+'," ", str)

    str = unicode_to_ascii(str.strip())
    str = '<start> ' + str + ' <end>'
    return str

en_test = u"My cat is black."
kr_test = u"내 고양이는 검은색 고양이야."
print(string_processing(en_test))
print(string_processing(kr_test))

def create_dataset(path, num_exam=None) :
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    
    word_pairs = []
    for l in lines[:num_exam] :
        word_pairs.append([string_processing(str) for str in l.split('\t')[:2]])

    return zip(*word_pairs)

###-------------------------------------------------Part II. Tokonizing & split dataset--------------------------------------###

# Use keras.preprocessing.text.tokenizer

def token(lang) :
    lang_tokenize = Tokenizer(filters=' ')
    lang_tokenize.fit_on_texts(lang)
    # tensor : text => sequence
    tensor = lang_tokenize.texts_to_sequences(lang)

    # padding
    tensor = pad_sequences(tensor, padding='post')
    
    return lang_tokenize, tensor

# Load dataset with tensor and tokenize
# With function create_dataset, return (input tensor, target tensor, input tokenize, target tokenize)
def load_dataset(path, num_exam=None) :
    # input, target = list
    # target : korean, input : english
    input, target = create_dataset(path, num_exam)

    input_token, input_tensor = token(input)
    target_token, target_tensor = token(target)

    return input_token, target_token, input_tensor, target_tensor

# Hyper_parameter
# Dataset을 제한하는 부분, 있어도 없어도 상관없을 듯
num_examples = 15
en_token, kr_token, en_tensor, kr_tensor = load_dataset(path_to_file, num_examples)
print(en_tensor)