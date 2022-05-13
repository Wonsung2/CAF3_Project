from django.shortcuts import render
from django.http import JsonResponse,HttpResponse
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import re
import urllib.request
import time
import json
# import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model, Model, load_model
from tensorflow.keras.layers import Dense, Activation,InputLayer, Flatten, Input, BatchNormalization,Dropout,Embedding  # Dense란 하나의 뉴럴 층
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import pickle


# CNN
from tensorflow.keras.layers import Conv2D,MaxPooling2D,AveragePooling2D, GlobalAveragePooling1D

# RNN
from tensorflow.keras.layers import SimpleRNN, LSTM

from tensorflow.keras.datasets import boston_housing, mnist, fashion_mnist, reuters
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD,Adam,RMSprop
from tensorflow.keras.utils import to_categorical  # onehotencoding해주는 거

# 이미지 로드
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # image에 대한 전체적인 전처리를 진행시켜준다.

# 자연어 처리
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from konlpy.tag import Okt, Mecab


# import matplotlib.pyplot as plt




# Create your views here.

def index(request):
    print('>>>>>>>>> chatpage')
    return render(request, 'main/index.html')

@csrf_exempt


def test(request):
    print('>>>>>>>>> test page')
    return render(request, 'main/test.html')

def example(request):
    print('>>>>>>>>> example page')
    return render(request, 'main/example.html')

def example2(request):
    print('>>>>>>>>> example page')
    return render(request, 'main/example2.html')

def chattrain(request):
    context = {}

    okt = Okt()
    mecab = Mecab()

    with open('./static/In.json') as file:
        data = json.load(file)

        data_frm = pd.DataFrame(data['intents'])
        data_frm.info()
        intents = data['intents']

        training_sentences = []  # 158
        training_labels = []  # 158
        labels = []
        responses = []

        for li in intents:
            for pattern in li['patterns']:
                training_sentences.append(okt.morphs(pattern))  # stem=True 제외.
                training_labels.append(li['tag'])
            responses.append(li['responses'])
            if li['tag'] not in labels:
                labels.append(li['tag'])
        num_classes = len(labels)

        lbl_encoder = LabelEncoder()
        lbl_encoder.fit(training_labels)
        training_labels = lbl_encoder.transform(training_labels)
        print(len(training_labels))
        training_labels

        vocab_size = 10000
        embedding_dim = 50
        max_len = max(len(i) for i in training_sentences)
        oov_token = "<OOV>"

        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
        tokenizer.fit_on_texts(training_sentences)

        word_index = tokenizer.word_index

        sequences = tokenizer.texts_to_sequences(training_sentences)
        padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_len)

        n_labels = np.array(training_labels)
        n_labels = to_categorical(n_labels)

        nlp_model = Sequential()

        # layer
        nlp_model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
        nlp_model.add(GlobalAveragePooling1D())
        nlp_model.add(Dropout(0.2))

        nlp_model.add(Dense(512))
        nlp_model.add(Activation('relu'))
        nlp_model.add(Dropout(0.2))

        nlp_model.add(Dense(256))
        nlp_model.add(Activation('relu'))
        nlp_model.add(Dropout(0.2))

        nlp_model.add(Dense(128))
        nlp_model.add(Activation('relu'))
        nlp_model.add(Dropout(0.2))

        nlp_model.add(Dense(64))
        nlp_model.add(Activation('relu'))
        nlp_model.add(Dropout(0.2))

        nlp_model.add(Dense(num_classes))
        nlp_model.add(Activation('softmax'))

        # compile
        nlp_model.compile(optimizer=Adam(lr=0.0014), loss='categorical_crossentropy', metrics=['accuracy'])

        nlp_model.summary()

        history = nlp_model.fit(padded_sequences, n_labels, batch_size=50, epochs=500, verbose=1, callbacks=[callbacks])

        nlp_model.save("static/chat_model")

        with open('static/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('static/label_encoder.pickle', 'wb') as ecn_file:
            pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

        context["result"] = 'Success......'
        return JsonResponse(context, content_type= "application/json")
