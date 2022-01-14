
#Meet Robo: your friend

#import necessary libraries
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages

# uncomment the following only the first time
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only


#Reading in the corpus
with open('chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("test", "he", "salam", "yo", "hola","hey","halo","test_2")
GREETING_RESPONSES = ["hi", "hey", "*angguk*", "hallo there", "yoi", "nice to hear from you"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split(): #aslinya
    #for word in sentence:
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

QUESTION_INPUTS = ("kapan", "dimana", "berapa", "siapa")
QUESTION_RESPONSES = ["info lengkap DM aja gan"]

def question(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split(): #aslinya
        if word.lower() in QUESTION_INPUTS:
            return random.choice(QUESTION_RESPONSES)

# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"Maaf Robo belum mengerti, bisa ditanyakan langsung ke CS team"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


flag=True
#print("ROBO: Welcome bro, Selamat Datang Di EnigmaBot, Saya akan membantu menjawab pertanyaan mengenai Enigmacamp. Jika ingin keluar, silahkan ketikan 'cukup'")
print("ROBO the bot: Selamat datang di fasilitas chatbot. Jika ingin mengakhiri chat, silahkan ketikan 'cukup'")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='cukup'):
        if(user_response=='thx' or user_response=='tx' ):
            flag=False
            print("ROBO the bot: Sami sami..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO the bot: "+greeting(user_response))
            elif(question(user_response)!=None):
                print("ROBO the bot: "+question(user_response))
            else:
                print("ROBO the bot: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)

    else:
        flag=False
        print("ROBO the bot: Dadah! stay safe..")    
        
        

