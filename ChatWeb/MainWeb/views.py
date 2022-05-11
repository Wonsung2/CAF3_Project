from django.shortcuts import render
from django.http import JsonResponse,HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
# from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama
from colorama import Fore, Style, Back
import random
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import re
import urllib.request
import time
import json
from tensorflow import keras

from tensorflow.keras.models import Sequential, clone_model, Model, load_model
from tensorflow.keras.layers import Dense, Activation,InputLayer, Flatten, Input, BatchNormalization,Dropout,Embedding  # Dense란 하나의 뉴럴 층
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# CNN
from tensorflow.keras.layers import Conv2D,MaxPooling2D,AveragePooling2D, GlobalAveragePooling1D

# RNN
from tensorflow.keras.layers import SimpleRNN, LSTM

from tensorflow.keras.datasets import boston_housing, mnist, fashion_mnist, reuters
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD,Adam,RMSprop
from keras.utils.np_utils import to_categorical

# 이미지 로드
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 자연어 처리
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt




# Create your views here.

def index(request):
    print('>>>>>>>>> chatpage')
    return render(request, 'main/index.html')

@csrf_exempt


def test(request):
    print('>>>>>>>>> test page')
    return render(request, 'main/test.html')

def chatanswer(request):
    context = {}


    ctext = request.GET['ctext']
    colorama.init()

    file = open(f"./static/intent.json", encoding="UTF-8")
    data = json.load(file)


    model = keras.models.load_model('static/chat_model')

    with open('static/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('static/label_encoder.pickle','rb') as enc:
        lbl_encoder = pickle.load(enc)
    max_len = 20

    print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end= "")
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([ctext]),
                                                                      truncating= 'post', maxlen= max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intent']:
        if i['tag'] == tag:
            txt1 = np.random.choice(i['responses'])
            print(Fore.GREEN + "Chatbot" + Style.RESET_ALL, txt1)


    context["answer"] = txt1

    return JsonResponse(context, content_type = "application/json")