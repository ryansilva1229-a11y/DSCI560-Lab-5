import requests
import time
import sys
import duckdb
import pandas as pd

HEADERS = {
    "User-Agent": "classification_bot/1.0 (by u/team17)"
}

PII_FIELDS = {
    "author",
    "author_fullname"
}

max = 5000  # assignement max possible
def remove_pii(post_data):
    return {k: v for k, v in post_data.items() if k not in PII_FIELDS}

def fetch_reddit_json(subreddit=None, query=None, max_results=5000):
    results = []
    after = None
    retries = 0
    MAX_RETRIES = 5

    while True:
        params = {
            "limit": 100,
            "after": after
        }

        if subreddit:
            url = f"https://www.reddit.com/r/{subreddit}/new.json"
        else:
            url = "https://www.reddit.com/search.json"
            params["q"] = query

        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)

            if r.status_code == 429:
                time.sleep(5)
                continue

            r.raise_for_status()
            data = r.json()["data"]["children"]

            if not data:
                break

            for item in data:
                clean_post = remove_pii(item["data"])
                results.append(clean_post)

                if len(results) >= max_results:
                    return results

            after = r.json()["data"]["after"]

            if after is None:
                break

            retries = 0
            time.sleep(1.2)

        except Exception as e:
            retries += 1
            if retries > MAX_RETRIES:
                print("Failed after retries:", e)
                break
            time.sleep(2 ** retries)

    return results

def map_post_to_table(post):
    post = dict(post)
    post["post_id"] = post.get("id")
    post["num_awards"] = post.get("total_awards_received", 0)
    post["media"] = str(post.get("media"))

    columns = [
        "post_id", "subreddit", "title", "selftext", "created_utc",
        "url", "upvote_ratio", "score", "num_comments", "over_18",
        "is_self", "link_flair_text", "subreddit_subscribers",
        "num_crossposts", "num_awards", "media"
    ]
    return {k: post.get(k, None) for k in columns}


def prompt_num_posts(initial_value=None):

    while True:
        if initial_value is not None:
            val = initial_value
            initial_value = None
        else:
            val = input(f"Enter number of posts to scrape (1-{max}): ").strip()
 



        n = int(val)
        if n <= 0:
            print("Please enter a valid count.")
            continue
        if n > max:
            print(f"Requested count ({n}) is greater than the max required ({max}).")
            continue
        return n

        


def main():
    if len(sys.argv) < 2:
        print("Usage: python scrape_reddit.py <subreddit> [num_posts]")
        print("Enter a subreddit name as either r/subreddit or subreddit")
        return

    subreddit = sys.argv[1].strip()
    if subreddit.startswith("r/"):

        subreddit = subreddit[2:]

    initial_num = None
    if len(sys.argv) >= 3:
        initial_num = sys.argv[2].strip()

    num_posts = prompt_num_posts(initial_value=initial_num)

    print(f"Scraping up to {num_posts} posts from r/{subreddit}...")

    conn = duckdb.connect('Data/mydb.duckdb')

    posts = fetch_reddit_json(subreddit=subreddit, max_results=num_posts)
    mapped_posts = [map_post_to_table(p) for p in posts]
    df = pd.DataFrame(mapped_posts)

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS REDDIT_DATA (
            post_id TEXT PRIMARY KEY,
            subreddit TEXT,
            title TEXT,
            selftext TEXT,
            created_utc BIGINT,
            url TEXT,
            upvote_ratio DOUBLE,
            score INTEGER,
            num_comments INTEGER,
            over_18 BOOLEAN,
            is_self BOOLEAN,
            link_flair_text TEXT,
            subreddit_subscribers INTEGER,
            num_crossposts INTEGER,
            num_awards INTEGER,
            media TEXT
        );
    """)

    try:
        conn.register("tmp_df", df) 
        conn.execute("""
            INSERT INTO REDDIT_DATA
            SELECT * FROM tmp_df
            ON CONFLICT (post_id) DO NOTHING
        """)
        conn.unregister("tmp_df")
    except Exception as e:
        print("Warning: insertion railed, error = ", e)
        for row in mapped_posts:
            try:
                conn.execute("""
                    INSERT INTO REDDIT_DATA VALUES (
                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                    )
                    ON CONFLICT (post_id) DO NOTHING
                """, list(row.values()))
            except Exception as ex:
                print("Insert row error:", ex)

    count = conn.execute("SELECT COUNT(*) FROM REDDIT_DATA").fetchone()[0]
    print(f"[r/{subreddit}] Total rows in REDDIT_DATA: {count}")

    return


if __name__ == "__main__":
    main()
