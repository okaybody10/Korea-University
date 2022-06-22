import io
import os
import time
import unicodedata
import re
from venv import create

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

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

en, kr = create_dataset(path_to_file)

###-------------------------------------------------Part II. Tokonizing--------------------------------------###