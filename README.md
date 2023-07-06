# Internship
Implementation of TF-IDF, word2vec, and RNN for API data in SMX Summer 2023 internship.

Relevant definitions: 
Term Frequency - Inverse Document Frequency (TF-IDF): Algorithm to determine how relevant words are in a given document or corpus. In this case, the output is in the form of a matrix. However, I will output the top 5-10 most relevant words in a document instead due to the corpus size.

word2vec: A natural language processing algorithm that uses a neural network to learn word associations. This method is especially well-suited for this dataset because of its large size. Each word is numerically represented as a vector, and cosine similarity is used to determine the level of similarity.

Recurrent Neural Network (RNN): A machine learning algorithm ideal for text analysis. It's often used for language translation, natural language processing, speech recognition, and image captioning. What makes RNN different from other neural networks is its "memory" feature: it uses information from prior inputs to influence the current input and output.

Notes:
The TestJSON folder consists of 3 JSON files pulled from the Army API folder. This is used better to visualize the TF-IDF matrix before the API data.
By default, tfidf.py will use the actual ArmyAPI data. You may also need to change path_to_json to be the relative path of your directory's Army API or TestJSON folders.
feature_names will list all words TF-IDF detected in alphabetical order. All numbers were replaced with the phrase "<NUM>."
tfidf_matrix is the TF-IDF matrix that is often used to visualize the output. Due to the size of the corpus, I do not recommend trying to open it, as its size will make it very hard to understand the results meaningfully. I am working on listing the top 5-10 most relevant words for each document, which will better reflect essential findings. 
