import gensim.downloader as pretrained_models
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from tfidf import clean_text, generate_corpus, partition_matrix, sort_partitions

path_to_json = "Internship Data ArmyAPI Pull_06222023"
# path_to_json = "TestJSON"

google_embeddings = pretrained_models.load(name="word2vec-google-news-300")

corpus = generate_corpus(folder_path=path_to_json)

vectorizer = TfidfVectorizer(analyzer=clean_text) # applies clean_text() as an analyzer
tfidf = vectorizer.fit_transform(corpus)
tfidf_array = tfidf.toarray()
tfidf_words = vectorizer.get_feature_names_out()

def generate_phrase_embeddings(embeddings):
    phrase_embeddings = []
    for text in corpus:
        phrase_embedding = []
        for word in text:
            if word in embeddings:
                phrase_embedding.append(embeddings[word])
        if phrase_embedding:
            phrase_embeddings.append(sum(phrase_embedding) / len(phrase_embedding))
    return phrase_embeddings

def generate_combined_embeddings(phrase_embeddings, tfidf_matrix):
    combined_embeddings = []
    for i, embedding in enumerate(phrase_embeddings):
        combined_embedding = embedding * tfidf_matrix[i].flatten() # ! operands could not be broadcast together with shapes (300,) (7418,)
        combined_embeddings.append(combined_embedding)
    return combined_embeddings

phrase_embeddings = generate_phrase_embeddings(embeddings=google_embeddings)
combined_embeddings = generate_combined_embeddings(phrase_embeddings=phrase_embeddings, tfidf_matrix=tfidf_array)
print(type(combined_embeddings))
print(combined_embeddings)    
   
# partitions = partition_matrix(matrix=tfidf_array.nonzero()) # gets all nonzero elements on each row
# top5_words = sort_partitions(partitions=partitions) # returns the top 5 words in descending order