import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

path_to_json = "Internship\Internship Data ArmyAPI Pull_06222023"
json_files = [pos_json for pos_json in os.listdir(path_to_json)] # corpus of all the articles

print(json_files) # ! TODO: need to extract text from json files, as TF-IDF is only applied to the NAMES of the files, rather than the content inside

vectorizer = TfidfVectorizer(stop_words="english") # initialize tfidf

tfidf = vectorizer.fit_transform(json_files) # apply tfidf to articles

for ele1, ele2 in zip(vectorizer.get_feature_names_out(), vectorizer.idf_): # print idf values for common words
    print(ele1, ':', ele2)

pd.DataFrame(tfidf.toarray()).to_csv("Internship\ArticleTFIDF") # export tfidf matrix to a csv

pd.DataFrame(vectorizer.get_feature_names_out()).to_csv("Internship/feature_names") # export feature names to a csv
