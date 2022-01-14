import json
import random
import string
import nltk
import numpy as np
import tensorflow as tf

from nltk.stem import WordNetLemmatizer, wordnet
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

nltk.download("punkt")
nltk.download("wordnet")

data = {"intents": [
    {"tag": "greeting",
     "patterns": ["Hello", "How are you?", "Hi there", "Hi", "Whats up"],
     "responses": ["Howdy Partner!", "Hello", "How are you doing?", "Greetings!", "How do you do?"],
    },
    {"tag": "age",
     "patterns": ["how old are you?", "when is your birthday?", "when was you born?"],
     "responses": ["I am 24 years old", "I was born in 1996", "My birthday is July 3rd and I was born in 1996", "03/07/1996"]
    },
    {"tag": "date",
     "patterns": ["what are you doing this weekend?","do you want to hang out some time?", "what are your plans for this week"],
     "responses": ["I am available all week", "I don't have any plans", "I am not busy"]
    },
    {"tag": "name",
     "patterns": ["what's your name?", "what are you called?", "who are you?"],
     "responses": ["My name is Kippi", "I'm Kippi", "Kippi"]
    },
    {"tag": "goodbye",
     "patterns": [ "bye", "g2g", "see ya", "adios", "cya"],
     "responses": ["It was nice speaking to you", "See you later", "Speak soon!"]
    }
]}

#penggalan kata
lemmatizer = WordNetLemmatizer()

#tampung penggalan kata
words = []
classes = []
doc_x = []
doc_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        token = nltk.word_tokenize(pattern)
        words.extend(token)
        doc_x.append(pattern)
        doc_y.append(intent["tag"])

#tambah tag yg belum ada
    if intent ["tag"] not in "classes":
        classes.append(intent["tag"])


words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))

print (words)
print (classes)
print (doc_y)
print (doc_x)

#list training data
training = []
out_empty = [0]*len(classes)

#create model
for idx, doc in enumerate (doc_x):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)