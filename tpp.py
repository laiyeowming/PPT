#TPP tryout V0.1 - 30 AUG

#checking numbers of entries
rawCSV = open('text.csv').read()
splitCSV = rawCSV.split('"\n"')
print(len(splitCSV))



#DATAFRAME
import pandas as pd
pd.set_option('display.max_colwidth', 100)

data = pd.read_csv('text.csv')
data.columns = ['text']
print(data.head())



#REMOVE PUNCTUATION
import string

def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

data['text_clean'] = data['text'].apply(lambda x: remove_punct(x))
print(data.head())



#TOKENISE
import re

def tokenise(text):
    tokens = re.split('\W+', text)
    return tokens

data['text_token'] = data['text_clean'].apply(lambda x: tokenise(x.lower()))
print(data.head())



#REMOVE STOPWORDS
# import requests
# stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
# stopwords = set(stopwords_list.decode().splitlines()) 

from nltk.corpus import stopwords
stopword = stopwords.words('english')

def remove_stopwords(tokenised_list):
    text = [word for word in tokenised_list if word not in stopword]
    return text

data['text_nostop'] = data['text_token'].apply(lambda x: remove_stopwords(x))
print(data.head())



#STEMMING VS LEMMATIZING
import nltk

ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()

def stemming(tokenized_text):
    text_stemming = [ps.stem(word) for word in tokenized_text]
    return text_stemming

data['text_stemmed'] = data['text_nostop'].apply(lambda x: stemming(x))

def lemmatizing(tokenized_text):
    text_lemmatizing = [wn.lemmatize(word) for word in tokenized_text]
    return text_lemmatizing

data['text_lemmatized'] = data['text_nostop'].apply(lambda x: lemmatizing(x))

data.head()



#Output a csv file
df = pd.DataFrame(data)
df.to_csv('texted.csv')