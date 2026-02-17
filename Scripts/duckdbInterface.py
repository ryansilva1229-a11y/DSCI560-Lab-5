import duckdb

conn = duckdb.connect('Data/mydb.duckdb')


#print(conn.execute("SHOW TABLES").fetchall())

print(conn.execute("SELECT * FROM REDDIT_DATA LIMIT 5").fetchall())