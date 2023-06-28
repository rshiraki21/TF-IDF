import json
import nltk
import os
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# path_to_json_files = "Internship\Internship Data ArmyAPI Pull_06222023"
path_to_json_files = "Internship\TestJSON"

# stopwords
stopwords = nltk.corpus.stopwords.words('english')

def process_text(unprocessed_text):
    unprocessed_text = "".join([word.lower() for word in unprocessed_text if word not in string.punctuation]) # removes punctuation
    tokens = re.split('\W+', unprocessed_text) # splits text into a list of words
    unprocessed_text = [word for word in tokens if word not in stopwords] # removes stopwords ()
    for word in unprocessed_text:
        if word.isnumeric():
            word = "<NUM>"
    return unprocessed_text

# * List of all the text values from json files
corpus = []
for file in os.listdir(path_to_json_files):
    filename = "%s/%s" % (path_to_json_files, file)
    with open(filename, "r") as f:
        article_json = json.load(f)
        corpus.append(process_text(article_json["text"]))
print(corpus)

vectorizer = TfidfVectorizer(stop_words="english", encoding="utf-8") # initialize tfidf

tfidf = vectorizer.fit_transform(corpus) # apply tfidf to corpus

print("Tokens used as features are : ")
print(vectorizer.get_feature_names_out())

print("\n Size of array. Each row represents a document. Each column represents a feature/token")
print(tfidf.shape)

print("\n Actual TF-IDF array")
print(tfidf.toarray())

for ele1, ele2 in zip(vectorizer.get_feature_names_out(), vectorizer.idf_): # print idf values for common words
    print(ele1, ':', ele2)

pd.DataFrame(tfidf.toarray()).to_csv("Internship/ArticleTFIDF") # export tfidf matrix to a csv

pd.DataFrame(vectorizer.get_feature_names_out()).to_csv("Internship/feature_names") # export feature names to a csv
