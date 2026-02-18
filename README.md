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
│   ├── duckdbInterface.py
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

## Building the Database
Before running the automation script, you need to populate the database by scraping subreddits. Run from the project root (DSCI560-Lab-5/):
```
python3 Scripts/scrapeReddit.py <subreddit> <num_posts>
```
Example:
```
python3 Scripts/scrapeReddit.py cybersecurity 500
```
You can run this multiple times with different subreddits to combine data in the same database. The database file will be created at Data/mydb.duckdb. 

## How to Run
Once the database has been populated, run the automation script from the project root (DSCI560-Lab-5/):
```
python3 Scripts/automation.py <interval_minutes> 
```
Example:
```
python3 Scripts/automation.py 5 
```
- interval_minutes: how often to scrape and re-cluster

## Interactive Query Mode
While the script is waiting between updates, you can type a keyword or message to find the closest matching cluster. The matching cluster's messages will be printed and a plot will be displayed highlighting that cluster.

Commands:
- Type any keyword or message to search
- skip: wait for the next update cycle
- exit: stop the script

## What It Does
1. Preprocesses text (removes punctuation, URLs, stopwords, lemmatizes)
2. Trains a Doc2Vec model and generates document embeddings
3. Clusters documents using KMeans with optimal cluster count determined by silhouette score
4. Displays a PCA scatter plot with cluster keywords labeled
5. Enters the interactive query mode between update cycles

