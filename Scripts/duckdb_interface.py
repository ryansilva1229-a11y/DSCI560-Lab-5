import duckdb

conn = duckdb.connect('Data/mydb.duckdb')


#print(conn.execute("SHOW TABLES").fetchall())

print(conn.execute("SELECT COUNT(*) FROM REDDIT_DATA").fetchall())