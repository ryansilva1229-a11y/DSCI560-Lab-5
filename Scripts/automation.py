import sys
import time
import subprocess
import os

DB_PATH = "Data/mydb.duckdb"

def run_scraper(subreddit, num_posts):
    print("Fetching data from Reddit...")
    subprocess.run(
        ["python3", "Scripts/scrapeReddit.py", subreddit, str(num_posts)],
        check=True
    )
    print("Data fetched and stored successfully.")

def run_analysis():
    print("Processing data and updating clusters...")
    subprocess.run(
        ["python3", "Scripts/analysisCluster.py"],
        check=True
    )
    print("Database updated successfully.")

def run_query(user_input):
    subprocess.run(
        ["python3", "Scripts/analysisCluster.py", "--query", user_input],
        check=True
    )

def automation_loop(interval, subreddit, num_posts):
    while True:
        try:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting update cycle...")
            run_scraper(subreddit, num_posts)
            run_analysis()
            print(f"\nNext update in {interval} minutes. Entering interactive query mode...")
            print("Type a keyword or message to search clusters, 'skip' to wait for the next update cyle, or 'exit' to quit.\n")

            deadline = time.time() + (interval * 60)
            while time.time() < deadline:
                user_input = input(">> Enter query: ").strip()

                if user_input.lower() == 'exit':
                    print("Exiting.")
                    return
                if user_input.lower() == 'skip':
                    remaining = deadline - time.time()
                    if remaining > 0:
                        time.sleep(remaining)
                    break
                if user_input:
                    run_query(user_input)

                remaining = int((deadline - time.time()) / 60)
                print(f"{remaining} minutes until next update.\n")

        except KeyboardInterrupt:
            print("\nAutomation stopped by user.")
            break
        except subprocess.CalledProcessError as e:
            print(f"A script failed: {e}")
            print(f"Retrying in {interval} minutes...")
            time.sleep(interval * 60)
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(interval * 60)

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 automation.py <interval_minutes> <subreddit> <num_posts>")
        print("Example: python3 automation.py 5 cybersecurity 500")
        return

    interval = int(sys.argv[1])
    subreddit = sys.argv[2]
    num_posts = int(sys.argv[3])

    automation_loop(interval, subreddit, num_posts)

if __name__ == "__main__":
    main()