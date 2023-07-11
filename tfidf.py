import copy
import json
import nltk
import numpy as np
import os
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

# nltk.download('stopwords')
# nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words("english")

tokenizer = nltk.tokenize.WordPunctTokenizer()

path_to_json = "Internship Data ArmyAPI Pull_06222023"
# path_to_json = "TestJSON"

# * Function to process text
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])  # remove punctuation and make all text lowercase
    tokens = word_tokenize(text)  # tokenize the text
    text = [word for word in tokens if word not in stopwords]  # remove stopwords
    text = ["<NUM>" if word.isnumeric() else word for word in text] # replace numbers with <NUM>
    return text

# * Creates a list consisting of all the pre-processed text
corpus = []
for file in os.listdir(path_to_json):
    filename = "%s/%s" % (path_to_json, file)
    with open (filename, "r") as f:
        article = json.load(f)
        corpus.append(article["text"])

# * Initializing TF-IDF
vectorizer = TfidfVectorizer(analyzer=clean_text)
tfidf = vectorizer.fit_transform(corpus)
tfidf_array = tfidf.toarray() # numpy.ndarray
tfidf_words = vectorizer.get_feature_names_out()

# * Exporting findings to csv
pd.DataFrame(vectorizer.get_feature_names_out()).to_csv("feature_names")
pd.DataFrame(tfidf_array).to_excel("tfidf_matrix.xlsx")

# * Export top 5 words from each article
def partition_matrix(matrix):
    nonzero_row = matrix[0]
    nonzero_col = matrix[1]  # end = index 15
    current_row = nonzero_row[0]
    current_row_index = 0
    all_tokens_and_scores = [] # list of lists, where each list is all words and their scores for each row
    current_row_token_and_score = []
    col_count = -1
    for row, col in zip(nonzero_row, nonzero_col):
        col_count += 1
        if current_row != row:
            all_tokens_and_scores.append(copy.deepcopy(current_row_token_and_score))
            current_row_token_and_score.clear()
            current_row = row
        current_row_token_and_score.append((tfidf_array[row, col], col)) # structure: (word value, (row, col))
        if col_count == len(nonzero_col) - 1:
            all_tokens_and_scores.append(copy.deepcopy(current_row_token_and_score))
    return all_tokens_and_scores

def sort_partitions(partitions):
    top5_list = []
    for p in partitions:
        top5_words = []
        sorted_row = list(sorted(p, reverse=True)) # sorts nonzero elements in row by descending order of word uniqueness
        count = 0
        for cell in sorted_row:
            top5_words.append(tfidf_words[cell[1]])
            count += 1
            if count == 5:
                break
        top5_list.append(top5_words)
    return top5_list

partitions = partition_matrix(tfidf_array.nonzero()) # partition all nonzero elements by row
top5_words = sort_partitions(partitions)
pd.DataFrame(sort_partitions(partitions)).to_csv("results")