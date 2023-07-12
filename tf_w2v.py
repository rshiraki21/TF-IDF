import copy
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# path_to_json = "Internship Data ArmyAPI Pull_06222023"
path_to_json = "TestJSON"

# * Function to process text
def process_text(raw_text):
    tokens = nltk.word_tokenize(raw_text)
    tokens = list(filter(lambda token: nltk.tokenize.punkt.PunktToken(token).is_non_punct, tokens))
    return tokens

# * Creates a list consisting of processed text
corpus = []
for file in os.listdir(path_to_json):
    filename = "%s/%s" % (path_to_json, file)
    with open (filename, "r") as f:
        article = json.load(f)
        raw_text = article["text"]
        corpus.append(process_text(raw_text))
print(corpus)
