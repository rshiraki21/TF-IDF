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

# * Exporting findings
pd.DataFrame(vectorizer.get_feature_names_out()).to_csv("feature_names")
pd.DataFrame(tfidf.toarray()).to_excel("tfidf_matrix.xlsx")

# ! TODO: Exporting top 5 words from each article
