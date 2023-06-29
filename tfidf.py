import json
import nltk
import os
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords = nltk.corpus.stopwords.words("english")

path_to_json = "Internship\Internship Data ArmyAPI Pull_06222023"
# path_to_json = "Internship\TestJSON"

# * Function to process text
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation]) # remove punctuation and makes all text lowercase
    tokens = re.split('\W+', text) # causes text to split when it sees more than one non-word characters (whitespace, special characters, etc.)
    text = [word for word in tokens if word not in stopwords] # remove stopwords
    for k in range(len(text)): # tokenizes numbers
        if text[k].isnumeric(): 
            text[k] = "<NUM>"
    return text

# * Creates a list consisting of all the text for processing
unprocessed_texts = []
for file in os.listdir(path_to_json):
    filename = "%s/%s" % (path_to_json, file)
    with open (filename, "r") as f:
        article = json.load(f)
        unprocessed_texts.append(article["text"])

# * Initializing TF-IDF
vectorizer = TfidfVectorizer(analyzer=clean_text)
tfidf = vectorizer.fit_transform(unprocessed_texts)

# * Exporting findings to csv
pd.DataFrame(vectorizer.get_feature_names_out()).to_csv("Internship/feature_names") # export feature names to a csv
pd.DataFrame(tfidf.toarray()).to_csv("Internship/tfidf_matrix") # export tfidf matrix to a csv



