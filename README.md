# DSCI-560 Lab 5

**Team Members:** Elise Hadidi (1137648541), Jordan Davies (1857892197), Ryan Silva (6463166471) 
**Team Number:**  17

## Overview
This project scrapes Reddit posts, preprocesses and clusters them using Doc2Vec and KMeans, and has an interactive query mode to find the closest matching cluster to a user's input.

## Folder Structure
```
DSCI560-Lab-5/
├── Data/
│   └── mydb.duckdb
├── Scripts/
│   ├── scrapeReddit.py
│   ├── analysisCluster.py
│   └── automation.py
└── requirements.txt
```

## Requirements
### Install dependencies with:
```
pip install -r requirements.txt
```

## Scripts
- Scripts/automation.py —  encompassing script, runs the full pipeline.
- Scripts/scrapeReddit.py — scrapes Reddit and stores posts in DuckDB. 
- Scripts/analysisCluster.py — preprocesses text, trains Doc2Vec, clusters with KMeans, and plots results.

## How to Run
Run from the project root (DSCI560-Lab-5/):
```
python3 Scripts/automation.py <interval_minutes> <subreddit> <num_posts>
```
Example:
```
python3 Scripts/automation.py 5 cybersecurity 1000
```
- interval_minutes: how often to scrape and re-cluster
- subreddit: the subreddit to scrape 
- num_posts: number of posts to fetch per cycle 

## Interactive Query Mode
While the script is waiting between updates, you can type a keyword or message to find the closest matching cluster. The matching cluster's messages will be printed and a plot will be displayed highlighting that cluster.

Commands:
- Type any keyword or message to search
- skip: wait for the next update cycle
- exit: stop the script

## What It Does
1. Scrapes posts from the specified subreddit and stores them in a DuckDB database
2. Preprocesses text (removes punctuation, URLs, stopwords, lemmatizes)
3. Trains a Doc2Vec model and generates document embeddings
4. Clusters documents using KMeans with optimal cluster count determined by silhouette score
5. Displays a PCA scatter plot with cluster keywords labeled
6. Enters the interactive query mode between update cycles

