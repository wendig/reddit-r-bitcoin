# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:42:48 2017

@author: Lorand
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import LSTM


print('reading csv files...')
df_sub = pd.read_csv('submission.csv')
df_com = pd.read_csv('comment.csv')
print('done reading')
#concat two dataframes together
frames = [df_sub,df_com]
df = pd.concat(frames)

# transform score between 0..1
max_value = df['score'].max()
min_value = df['score'].min()
df['score'] = (df['score'] - min_value) / (max_value - min_value)


# remove stopwords
# nltk.download() -> install 'punkt'
stop_words = set(stopwords.words('english'))

for index, row in df.iterrows():
    # create list of words
    word_tokens = word_tokenize(row['text'])
    # keep only the 'relevant' words
    filtered_sentence = [w.lower() for w in word_tokens if not w.lower() in stop_words]
    row['text'] = " ".join(filtered_sentence)

#drop rows containing very long and short text  
df = df[df['text'].map(len) < 1000 ]
df = df[df['text'].map(len) > 10 ]

#calculate min,max,average -> the length of text
avg = 0
Max = 0
Min = 1000

len_list = []

for index, row in df.iterrows():
    length = len(row['text'])   
    len_list.append(length)
    avg = avg + length
    if Max < length:
        Max = length
    if Min > length:
        Min = length    
print('shape: ' + str(df.shape))
avg = avg / df.shape[0]
print('average: ' + str(avg))
print('max: ' + str(Max))
print('min: ' + str(Min))


#plotting length of text
len_ar = np.asarray(len_list)
plt.plot(len_ar)
plt.ylabel('lenght of text')
plt.show()

#todo ezzel folytat
#https://medium.com/@thoszymkowiak/how-to-implement-sentiment-analysis-using-word-embedding-and-convolutional-neural-networks-on-keras-163197aef623
#create a list of words

#keras tutorial on werd embeddings:
#https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

embeddings_index = {}
f = open('glove.6B.100d.txt',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
#I follow keras official blog
#https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
texts = []  # list of text samples
labels = []  # list of label ids

for index, row in df.iterrows():
    texts.append( row['text'] )
    labels.append( row['score'] )
print('-----------')
print('example')
print(texts[10])
print(labels[10])
print('-----------')

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

#if score would be categorical
#labels = to_categorical(np.asarray(labels))
labels = np.asarray(labels)


print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')
# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70


# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1)(x)
x = MaxPooling1D(pool_size=pool_size)(x)
x = LSTM(64)(x)
x = Dense(128,activation='relu')(x)
preds = Dense(1)(x)

model = Model(sequence_input, preds)
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=1000,
          validation_data=(x_val, y_val))
