#analysis and clusterin
import duckdb
import pandas as pd
from gensim.models.doc2vec import Doc2Vec,\
    TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


conn = duckdb.connect('Data/mydb.duckdb')
#load into a dataframe
results = conn.execute("SELECT * FROM REDDIT_DATA").df()

#print(results.columns)

results["Full_Text"] = results["title"] + " " + results["selftext"]

#remove punctuation and numbers
results["Full_Text"] = [re.sub(r'[^\w\s]', '', str(x)) for x in results["Full_Text"]]
#normalize lower case
results["Full_Text"] = [str(x).lower() for x in results["Full_Text"]]



lemmatizer = WordNetLemmatizer()
results["Full_Text"] = [lemmatizer.lemmatize(word) for word in results["Full_Text"]]

stop_words = set(stopwords.words('english'))
results["Full_Text"] = [word for word in results["Full_Text"] if word not in stop_words]
#print(results["Full_Text"])

# preproces the documents, and create TaggedDocuments
tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                              tags=[str(i)]) for i,
               doc in enumerate(results["Full_Text"])]

# train the Doc2vec model
model = Doc2Vec(vector_size=30,
                min_count=2, epochs=50)
model.build_vocab(tagged_data)
model.train(tagged_data,
            total_examples=model.corpus_count,
            epochs=model.epochs)

# get the document vectors
results["Vector"] = [model.infer_vector(
    word_tokenize(doc)) for doc in results["Full_Text"]]

#  print the document vectors
'''
for i, doc in enumerate(results["Full_Text"]):
    print("Document", i+1, ":", doc)
    print("Vector:", results["Vector"].iloc[i])
    print()

'''

#now using dataset, going to cluster via kmeans
embedding_df = results['Vector'].apply(pd.Series)
min_cluster = 2
def cluster_count(embedding_df, max):
    sil_scores = []
    for k in range(min_cluster,max+1):
        kmeans = KMeans(n_clusters=k,random_state=0)
        cluster_labels = kmeans.fit_predict(embedding_df)
        sil_avg = silhouette_score(embedding_df,cluster_labels)
        sil_scores.append(sil_avg)
    return sil_scores

max_c_count = min(len(embedding_df), 100)

silhouette_scores = cluster_count(embedding_df, max_c_count)

optimal_clusters = min(range(len(silhouette_scores)), key=lambda i: abs(silhouette_scores[i] - 1))

#performing k means

kmeans = KMeans(n_clusters=optimal_clusters+min_cluster, random_state = 0)
cluster_labels = kmeans.fit_predict(embedding_df)

output_df = pd.DataFrame({'Original Text': results["Full_Text"],'Cluster': cluster_labels})

print(silhouette_scores)
print(optimal_clusters+min_cluster)
print(output_df.head())