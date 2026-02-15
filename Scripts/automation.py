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

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

DB_PATH = "../Data/mydb.duckdb"
MODEL_PATH = "doc2vec.model"
ANALYSIS_SCRIPT = "AnalysisCluster.py"

def run_analysis_clustering():
    print(f"[INFO] Running {ANALYSIS_SCRIPT}...")
    try:
        result = subprocess.run(
            ["python3", ANALYSIS_SCRIPT],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print("[SUCCESS] Analysis complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {ANALYSIS_SCRIPT} failed: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False

def load_database():
    conn = duckdb.connect(DB_PATH, read_only=True)
    df = conn.execute("SELECT * FROM REDDIT_DATA").df()
    conn.close()
    
    if "Full_Text" not in df.columns:
        df["Full_Text"] = df["title"].fillna('') + " " + df["selftext"].fillna('')
    
    return df

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = re.sub(r'[^\w\s]', '', str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return tokens

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file {MODEL_PATH} not found. Run analysis first.")
        return None
    
    print("[INFO] Loading Doc2Vec model...")
    model = Doc2Vec.load(MODEL_PATH)
    return model

def vectorize_documents(df, model):
    print("[INFO] Vectorizing documents...")
    df["Tokens"] = df["Full_Text"].apply(preprocess_text)
    df["Vector"] = df["Tokens"].apply(lambda tokens: model.infer_vector(tokens))
    return df

def cluster_documents(df, max_clusters=5):
    print("[INFO] Clustering documents...")
    
    embedding_df = df['Vector'].apply(pd.Series)
    
    silhouette_scores = []
    max_c = min(max_clusters, len(df))
    for k in range(2, max_c+1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(embedding_df)
        sil = silhouette_score(embedding_df, labels)
        silhouette_scores.append(sil)
    
    optimal_clusters = np.argmax(silhouette_scores) + 2
    print(f"[INFO] Optimal clusters determined: {optimal_clusters}")
    
    kmeans_final = KMeans(n_clusters=optimal_clusters, random_state=0)
    cluster_labels = kmeans_final.fit_predict(embedding_df)
    df["Cluster"] = cluster_labels
    
    return df, optimal_clusters

def plot_clusters(df):
    print("[INFO] Generating cluster visualization...")
    
    vectors = np.array(df["Vector"].tolist())
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=df["Cluster"], cmap='tab10', s=30, alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title("Document Clusters (2D PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

def interactive_query(df, model):
    print("\n" + "="*60)
    print("INTERACTIVE QUERY MODE")
    print("="*60)
    print("Enter keywords/message to find matching cluster")
    print("Commands: 'exit' to quit | 'plot' to visualize | 'stats' for cluster info\n")
    
    vectors = np.array(df["Vector"].tolist())
    
    while True:
        user_input = input(">> Enter query: ").strip()
        
        if user_input.lower() == "exit":
            print("[INFO] Exiting interactive mode...")
            break
            
        if user_input.lower() == "plot":
            plot_clusters(df)
            continue
        
        if user_input.lower() == "stats":
            print("\n[CLUSTER STATISTICS]")
            for cluster_id in sorted(df["Cluster"].unique()):
                count = len(df[df["Cluster"] == cluster_id])
                print(f"  Cluster {cluster_id}: {count} documents")
            print(f"  Total: {len(df)} documents\n")
            continue
            
        if not user_input:
            continue
        
        user_tokens = preprocess_text(user_input)
        user_vec = model.infer_vector(user_tokens)
        
        sims = cosine_similarity([user_vec], vectors)[0]
        top_idx = np.argsort(sims)[-5:]
        
        print("\n[TOP 5 MATCHES]")
        for i, idx in enumerate(reversed(top_idx), 1):
            cluster = df['Cluster'].iloc[idx]
            text = df['Full_Text'].iloc[idx][:200]
            score = sims[idx]
            print(f"{i}. Cluster {cluster} | Similarity: {score:.4f}")
            print(f"   {text}...\n")

def automation_loop(interval, subreddit, num_posts):
    print("\n" + "="*60)
    print(f"AUTOMATION MODE: Updating every {interval} minutes")
    print(f"Subreddit: r/{subreddit} | Posts per fetch: {num_posts}")
    print("="*60)
    
    while True:
        try:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting update cycle...")
            
            print("[INFO] Fetching new data from Reddit...")
            result = subprocess.run(
                ["python3", "scrape_reddit.py", subreddit, str(num_posts)],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            
            success = run_analysis_clustering()
            
            if success:
                print(f"[SUCCESS] Data fetched and analyzed successfully")
            else:
                print("[WARNING] Analysis failed, will retry next cycle")
            
            print(f"[INFO] Next update in {interval} minutes")
            print(f"[INFO] Sleeping...\n")
            time.sleep(interval * 60)
            
        except KeyboardInterrupt:
            print("\n[INFO] Automation stopped by user.")
            break
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Process error: {e}")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)
            print(f"[INFO] Retrying in {interval} minutes...")
            time.sleep(interval * 60)
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            print(f"[INFO] Retrying in {interval} minutes...")
            time.sleep(interval * 60)

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 automation.py <interval_minutes> <subreddit> <num_posts>")
        print("\nExample: python3 automation.py 5 health 500")
        print("\nThe script will:")
        print("  1. Scrape data from Reddit")
        print("  2. Run analysis and clustering")
        print("  3. Enter interactive query mode")
        print("  4. Repeat steps 1-2 every <interval_minutes>")
        return
    
    interval = int(sys.argv[1])
    subreddit = sys.argv[2]
    num_posts = int(sys.argv[3])
    
    if not os.path.exists(DB_PATH):
        print("[INFO] No database found. Running initial scrape...")
        subprocess.run(["python3", "scrape_reddit.py", subreddit, str(num_posts)], check=True)
    
    print("[INFO] Running initial analysis...")
    success = run_analysis_clustering()
    
    if not success:
        print("[ERROR] Initial analysis failed. Please check your data.")
        return
    
    print("[INFO] Loading processed data for interactive mode...")
    df = load_database()
    model = load_model()
    
    if model is None:
        print("[ERROR] Could not load model. Exiting.")
        return
    
    df = vectorize_documents(df, model)
    df, optimal_clusters = cluster_documents(df)
    
    print(f"\n[SUCCESS] Initial setup complete!")
    print(f"  - Total documents: {len(df)}")
    print(f"  - Number of clusters: {optimal_clusters}")
    
    interactive_query(df, model)
    
    automation_loop(interval, subreddit, num_posts)

if __name__ == "__main__":
    main()