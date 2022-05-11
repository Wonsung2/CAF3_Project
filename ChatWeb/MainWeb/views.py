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