import os, json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

path_to_json = 'Internship\Internship Data ArmyAPI Pull_06222023'
json_files = [pos_json for pos_json in os.listdir(path_to_json)]

jsons_data = pd.DataFrame(columns=["text"])

for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)

        jsons_data.loc[index] = json_text["text"]

vectorizer = TfidfVectorizer(stop_words="english")

tfidf = vectorizer.fit_transform(jsons_data["text"])

print(vectorizer.get_feature_names_out())
print(tfidf.toarray())
