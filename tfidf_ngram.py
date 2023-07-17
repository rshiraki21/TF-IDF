import copy
import json
import nltk
import os
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words("english")

tokenizer = nltk.tokenize.WordPunctTokenizer()

path_to_json = "Internship Data ArmyAPI Pull_06222023"
# path_to_json = "TestJSON"

# * Function to process text
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])  # remove punctuation and make all text lowercase
    tokens = word_tokenize(text)  # tokenize the text
    text = [word for word in tokens if word not in stopwords]  # remove stopwords
    text = ["<NUM>" if word.isnumeric() else word for word in text] # replace numbers with <NUM>, comment this out if results should include numbers
    return text

# * Creates a list consisting of all the post-processed text
corpus = []
for file in os.listdir(path_to_json):
    filename = "%s/%s" % (path_to_json, file)
    with open (filename, "r") as f:
        article = json.load(f)
        corpus.append(list(clean_text(article["text"])))

# * Overrides string tokenizing step to preserve text processing and n-grams
def identity_tokenizer(text):
    return text

# * Initializing TF-IDF and related variables
vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, ngram_range=(1,3), lowercase=False) # can't use callable analyzer and ngram_range
tfidf = vectorizer.fit_transform(corpus)
tfidf_array = tfidf.toarray() # numpy.ndarray
tfidf_words = vectorizer.get_feature_names_out()

# * Returns a list of tuples containing the word value and col for each row in the TF-IDF matrix. Only nonzero values are included
def partition_matrix(matrix):
    nonzero_row = matrix[0]
    nonzero_col = matrix[1]  # end = index 15
    current_row = nonzero_row[0]
    all_tokens_and_scores = [] # list of lists, where each list is all words and their scores for each row
    current_row_token_and_score = []
    col_count = -1
    for row, col in zip(nonzero_row, nonzero_col):
        col_count += 1
        if current_row != row:
            all_tokens_and_scores.append(copy.deepcopy(current_row_token_and_score))
            current_row_token_and_score.clear()
            current_row = row
        current_row_token_and_score.append((tfidf_array[row, col], col)) # structure: (word value, col)
        if col_count == len(nonzero_col) - 1:
            all_tokens_and_scores.append(copy.deepcopy(current_row_token_and_score))
    return all_tokens_and_scores

# * Gets top 5 words per document
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

# * Exporting feature names, TF-IDF matrix, and top 5 words per document
pd.DataFrame(vectorizer.get_feature_names_out()).to_json("TFIDF_ngram_results_numbers/ngram_feature_names.json", orient="values")
pd.DataFrame(tfidf_array).to_csv("TFIDF_ngram_results_numbers/ngram_tfidf_matrix.csv")
pd.DataFrame(sort_partitions(partitions)).to_json("TFIDF_ngram_results_numbers/ngram_results", orient="records")
# pd.DataFrame(sort_partitions(partitions)).to_csv("TFIDF_ngram_results_no_numbers/results.csv") # option to export to csv for improved readability