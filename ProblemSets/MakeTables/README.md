# Creating tables using SQL

This directory has some files that you can use when creating tables using SQL. The commands are described in the lecture notes of Lecture 1.


- The data file for the Star database for sqlite ingest -[YAEPS.stars-table-sqlite.dat](YAEPS.stars-table-sqlite.dat). If you want a version with header suitable for MySQL try [YAEPS.stars-table.dat](YAEPS.stars-table.dat)
- The data file for the Observations database for sqlite ingest - [YAEPS.observations-table-sqlite.dat](YAEPS.observations-table-sqlite.dat). If you want a version with header suitable for MySQL try [YAEPS.observations-table.dat](YAEPS.observations-table.dat).
- The SQL code to create the Star table in sqlite3 - [sqlite3-make-stars-table.sql](sqlite3-make-stars-table.sql). For MySQL you would use [myqsl-make-stars-table.sql](myqsl-make-stars-table.sql).
- The SQL code to create the Observations table in sqlite3 - [sqlite3-make-observations-table.sql](sqlite3-make-observations-table.sql).  For MySQL you would use [myqsl-make-observations-table.sql](myqsl-make-observations-table.sql).


If instead you want to create these tables using python, you would want to use Python, you can look at [make_tables_python.py](make_tables_python.py) which will require some work on your part to get the queries right. If you get stuck the solution is also in this directory. 
