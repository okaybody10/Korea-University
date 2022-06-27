from cmath import tanh
import io
from lib2to3.pgen2.tokenize import tokenize
import os
import time
from tokenize import Token
import unicodedata
import re
from venv import create
from matplotlib.style import context

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from torch import embedding

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
# input: en, output: kr
num_examples = 15000
en_token, kr_token, en_tensor, kr_tensor = load_dataset(path_to_file, num_examples)

# Dataset을 split합니다.
# input: en, output: kr
# K-fold?
input_tensor_train, input_tensor_test, target_tensor_train, target_tensor_test = train_test_split(en_tensor, kr_tensor, test_size=0.2)

# data_set 출력
print(len(input_tensor_train), print(len(target_tensor_test)))

###-------------------------------------------------Part III. Hyper-parameter--------------------------------------###
# 1. 훈련하는 것들의 크기가 얼마인가? 
# 2. Batch_size는 얼마인가? + Epoch는?
# 3. Word-embedding은 어느정도 사이즈로 할 것인가?
# 4. 최대 input, target word size => word_index + 1 (word_index는 tokenize 객체로, len이나 size 함수를 가지고 있지 않음)
BUFFER_SIZE = len(input_tensor_test) + 1
BATCH_SIZE = 64
epoch = len(input_tensor_train) // BATCH_SIZE
embedding_size = 256 
units = 1024 # GRU-output size(d_model size, h=16 => 1024/16= 64 <= multi-head attension size)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape[1], example_target_batch.shape[1])

###-------------------------------------------------Part IV. Encoder--------------------------------------###
# vocab_length <-- embedding
vocab_input_len = len(en_token.word_index) + 1
vocab_target_len = len(kr_token.word_index) + 1
# Use GRU, hidden_state initialize : 0
# LSTM: Several Hyperparameter, will use GRU
# Encoder structure: 1. embedding vector(input/output,...) 2. GRU 3. GRU output dimension(enc_units) 4. batch_sz
class Encoder(tf.keras.Model) :
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz) :
        # enc_units: encoder output dimension, 1024(Optimiality hypermeter condition in paper)
        super(Encoder, self).__init__()
        self.batch_size = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # We will use attention, return_sequence: true, return_state= true
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences = True, return_state = True, recurrent_initializer='glorot_uniform')
    
    # GRU call
    def call(self, x, hidden) :
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def hidden_initialization(self) :
        return tf.zeros((self.batch_size, self.enc_units))
    
# encoder: input word
encoder = Encoder(vocab_input_len, embedding_size, units, BATCH_SIZE)

example_hidden = encoder.hidden_initialization()
example_output, example_hidden = encoder(example_input_batch, example_hidden)
# shape print(output)
# gru return sequence=true, so sequence also print
# shape: (batch_size, sequence(total hidden state count), embedding_size(gru_units))
print('Encoder output state shape {}'.format(example_output))
# shart print(hidden)
# shape: (batch_size, embedding_size)
print('Encoder hidden state shape {}'.format(example_hidden))

print('Encoder output {}'.format(example_hidden))

# Bahdanuau attention
class Bahdanuau(tf.keras.layers.Layer) :
    def __init__(self, units) :
        super(Bahdanuau, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.W3 = tf.keras.layers.Dense(1)

    # To utilize the attention technique, we have to define Query, Key, and Value.
    # Query: Hidden state at each timestep in decoder
    # Key, Value: Hidden state(s) all timesteps in encoder
    # Number of Key(Value)s are #{Encoder states}
    # Number of Query: 1
    # Note that Sequence shape: (batch_size, "sequence_size", embedding_size)
    # hidden state return shape: (batch_size, embedding_size (line 153)) => Expand dimension to (batch_size, 1, embedding_size)

    def call(self, query, values) :
        query_expand = tf.expand_dims(query, 1)

        # Attention formula: score(query, hidden i) = W_1^T * tanh(W*query + W'*h_i)
        # Attention score shape : (batch_size, max_len, 1)
        attention_score = self.W3(tf.nn.tanh(self.W1(query_expand) + self.W2(values)))

        # attention weight => softmax(attention_score), shape: same as attention_score i.e. (batch_size, max_len, 1)
        attention_weight = tf.nn.softmax(attention_score, axis=1)

        # Final, dot product of attention_weight and values is context vector
        # shape: (batch_size, hidden_size)
        # But attention_weight shape: (batch_size, max_len, 1) / values shape: (batch_size, max_len, hidden_size) => dimension reduction
        context_vector = attention_weight * values
        context_vector = tf.reduce_sum(context_vector, axis = 1)

        return context_vector, attention_weight

attention_layer = Bahdanuau(10)
attention_res, attention_weight = attention_layer(example_hidden, example_output)