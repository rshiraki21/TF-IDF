import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

path_to_json = "Internship\Internship Data ArmyAPI Pull_06222023"
articles = [article for article in os.listdir(path_to_json)] # list of all the articles

# * Need to read articles in json_files, read the 'text' values, put each of those into a corpus, THEN apply TF-IDF
corpus = []

for article in articles:
    with open(article) as json_file:
        text = json.load(json_file)
        corpus.append(text)
print(corpus)


# print(json_files) # ! TODO: need to extract text from json files, as TF-IDF is only applied to the NAMES of the files, rather than the content inside

# vectorizer = TfidfVectorizer(stop_words="english") # initialize tfidf

# tfidf = vectorizer.fit_transform(json_files) # apply tfidf to articles

# for ele1, ele2 in zip(vectorizer.get_feature_names_out(), vectorizer.idf_): # print idf values for common words
#     print(ele1, ':', ele2)

# pd.DataFrame(tfidf.toarray()).to_csv("Internship\ArticleTFIDF") # export tfidf matrix to a csv

# pd.DataFrame(vectorizer.get_feature_names_out()).to_csv("Internship/feature_names") # export feature names to a csv
