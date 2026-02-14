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
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

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
embedding_df["Cluster"] = cluster_labels
embedding_df["text"] = results["Full_Text"]
#print(silhouette_scores)
print(optimal_clusters+min_cluster)
#print(output_df.head())

#----------------------------------------------------
labels = embedding_df["Cluster"].values
vectors = embedding_df.drop(columns={"Cluster","text"}).values.astype(float)

#now using pca so that we can actually plot this on a scatter plot
pca = PCA(n_components=2, random_state=0)
X_2d = pca.fit_transform(vectors)

#using tfidf to figure out which terms are most frequent later
vectorizer = TfidfVectorizer(stop_words='english', max_features=20000) 
tfidf = vectorizer.fit_transform(embedding_df["text"]) 
feature_names = vectorizer.get_feature_names_out()
labels_unique = np.unique(labels)


label_to_idx = {lab:i for i, lab in enumerate(labels_unique)}
cluster_sums = [None] * len(labels_unique)
cluster_counts = np.zeros(len(labels_unique), dtype=int)

#cycling through all the labels
for i, lab in enumerate(labels):
    idx = label_to_idx[lab]

    v_dense = np.asarray(tfidf[i].toarray()).ravel() 
    if cluster_sums[idx] is None:
        cluster_sums[idx] = v_dense.copy()
    else:
        cluster_sums[idx] += v_dense
    cluster_counts[idx] += 1

#finding top keyword
top_keyword = {}
for lab, i_idx in label_to_idx.items():
    if cluster_counts[i_idx] == 0:
        top_keyword[lab] = ""
        continue
    mean_arr = cluster_sums[i_idx] / cluster_counts[i_idx]   
    top_idx = int(mean_arr.argmax())
    top_keyword[lab] = feature_names[top_idx]
centroids = {}
for lab in labels_unique:
    mask = (labels == lab)
    if mask.sum() == 0:
        centroids[lab] = (0, 0)
    else:
        centroids[lab] = (X_2d[mask, 0].mean(), X_2d[mask, 1].mean())


#plot
plt.figure(figsize=(10, 7))

cmap = plt.get_cmap("tab10")

label_indices = np.array([label_to_idx[lab] for lab in labels])
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                      c=label_indices, cmap=cmap, s=10, alpha=0.8)

# labeling with keywords
for lab in labels_unique:
    cx, cy = centroids[lab]
    kw = top_keyword.get(lab, "")
    plt.text(cx, cy, kw, fontsize=12, weight='bold', 
             horizontalalignment='center', verticalalignment='center',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2))

plt.title("Collapsed PCA Scatter Plot")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(False)
plt.tight_layout()
'''
from matplotlib.lines import Line2D
legend_elems = []
for lab, idx in label_to_idx.items():
    legend_elems.append(Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=cmap(idx % 10), markersize=8, label=f"{lab}: {top_keyword.get(lab,'')}"))
plt.legend(handles=legend_elems, bbox_to_anchor=(1.05, 1), loc='upper left')
'''

plt.show()
'''
for c in labels.unique():
    print(embedding_df[embedding_df["Cluster"]==c].head())
    '''

print_df = embedding_df[["Cluster","text"]]
print_df.to_csv("Data/output.csv")