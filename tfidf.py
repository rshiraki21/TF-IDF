import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

path_to_json_files = "Internship Data ArmyAPI Pull_06222023" 

# * List of all the text values from json files
corpus = []
for file in os.listdir(path_to_json_files):
    filename = "%s/%s" % (path_to_json_files, file)
    with open(filename, "r") as f:
        key_value = json.load(f)
        text = key_value["text"]
        corpus.append(text)

vectorizer = TfidfVectorizer(stop_words="english") # initialize tfidf

tfidf = vectorizer.fit_transform(corpus) # apply tfidf to corpus

# for ele1, ele2 in zip(vectorizer.get_feature_names_out(), vectorizer.idf_): # print idf values for common words
#     print(ele1, ':', ele2)

pd.DataFrame(tfidf.toarray()).to_csv("ArticleTFIDF") # export tfidf matrix to a csv

pd.DataFrame(vectorizer.get_feature_names_out()).to_csv("feature_names") # export feature names to a csv
