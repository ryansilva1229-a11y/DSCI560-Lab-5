import sys
import time
import subprocess
import duckdb
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

DB_PATH = "../Data/mydb.duckdb"
MODEL_PATH = "doc2vec.model"

def load_database():
    conn = duckdb.connect(DB_PATH)
    df = conn.execute("SELECT title, selftext FROM REDDIT_DATA").df()
    df["Full_Text"] = df["title"].fillna('') + " " + df["selftext"].fillna('')
    return df, conn

def preprocess_text(df):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned_texts = []
    for doc in df["Full_Text"]:
        doc = str(doc).lower()
        doc = re.sub(r'[^\w\s]', '', doc)
        tokens = word_tokenize(doc)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
        cleaned_texts.append(' '.join(tokens))
    df["Full_Text"] = cleaned_texts
    return df

def load_or_train_doc2vec(df):
    if os.path.exists(MODEL_PATH):
        print("[INFO] Loading existing Doc2Vec model...")
        model = Doc2Vec.load(MODEL_PATH)
    else:
        print("[INFO] Training new Doc2Vec model...")
        tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[str(i)]) 
                       for i, doc in enumerate(df["Full_Text"])]
        model = Doc2Vec(vector_size=20, min_count=2, epochs=50)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(MODEL_PATH)
        print("[INFO] Doc2Vec model trained and saved.")
    return model

def vectorize_documents(df, model):
    vectors = [model.infer_vector(word_tokenize(doc)) for doc in df["Full_Text"]]
    df["Vector"] = vectors
    return df

def cluster_documents(df, max_clusters=5):
    embedding_df = df['Vector'].apply(pd.Series)
    silhouette_scores = []
    max_c = min(max_clusters, len(df))
    for k in range(2, max_c+1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(embedding_df)
        sil = silhouette_score(embedding_df, labels)
        silhouette_scores.append(sil)
    optimal_clusters = min(range(len(silhouette_scores)), key=lambda i: abs(silhouette_scores[i]-1)) + 2
    print(f"[INFO] Optimal clusters determined: {optimal_clusters}")
    kmeans_final = KMeans(n_clusters=optimal_clusters, random_state=0)
    cluster_labels = kmeans_final.fit_predict(embedding_df)
    df["Cluster"] = cluster_labels
    return df, silhouette_scores, kmeans_final

def save_to_db(df, conn):
    df["Vector"] = df["Vector"].apply(lambda x: list(x))
    conn.execute("CREATE OR REPLACE TABLE REDDIT_DATA AS SELECT * FROM df")
    print("[INFO] Database updated with Full_Text, Vector, and Cluster columns.")

def plot_clusters(df):
    vectors = np.array(df["Vector"].tolist())
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)
    plt.figure(figsize=(8,6))
    plt.scatter(reduced[:,0], reduced[:,1], c=df["Cluster"], cmap='tab10')
    plt.title("Document Clusters (2D PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

def interactive_query(df, model):
    vectors = np.array(df["Vector"].tolist())
    while True:
        user_input = input("Enter keywords/message (or 'exit'): ")
        if user_input.lower() == "exit":
            break
        user_vec = model.infer_vector(word_tokenize(user_input.lower()))
        sims = cosine_similarity([user_vec], vectors)[0]
        top_idx = np.argsort(sims)[-5:]
        print("\nTop Matches:")
        for idx in reversed(top_idx):
            print(f"- Cluster {df['Cluster'].iloc[idx]} | {df['Full_Text'].iloc[idx][:200]}...")
        print()
        plot_clusters(df)

def automation_loop(interval, subreddit, num_posts):
    while True:
        print("[INFO] Fetching new data...")
        subprocess.run(["python3", "scrape_reddit.py", subreddit, str(num_posts)], check=True)
        df, conn = load_database()
        df = preprocess_text(df)
        model = load_or_train_doc2vec(df)
        df = vectorize_documents(df, model)
        df, _, _ = cluster_documents(df)
        save_to_db(df, conn)
        print(f"[INFO] Sleeping for {interval} minutes...\n")
        time.sleep(interval*60)

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 automation_lab5.py <interval_minutes> <subreddit> <num_posts>")
        return
    interval = int(sys.argv[1])
    subreddit = sys.argv[2]
    num_posts = int(sys.argv[3])
    df, conn = load_database()
    df = preprocess_text(df)
    model = load_or_train_doc2vec(df)
    df = vectorize_documents(df, model)
    df, _, _ = cluster_documents(df)
    save_to_db(df, conn)
    interactive_query(df, model)
    automation_loop(interval, subreddit, num_posts)

if __name__ == "__main__":
    main()
