#!/usr/bin/env python

"""
Query the simple tables created by make_simple_tables.py

As in that routine we can do this using standard Python and Pandas.
Here I will show both for the first question - then only Pandas since it
is shorter (although to be fair one could wrap the stnadard Python in
a function and then it would be just as short)
"""

from __future__ import print_function
import sqlite3 as lite
from astropy.table import Table
import pandas as pd

# It does not matter which database I connect to - the two are identical. 
con = lite.connect('SimpleTables-default.db')

#----------------
# Question a)
#  Find the Ra & Dec of all objects with B > 16.
#----------------

# Standard Python
rows = con.execute("Select Name, Ra, Decl From MagTable Where B > 16")
print("Standard Python:\n-------------------------\n")
print("   Name    Ra      Dec")
for row in rows:
    print("{0:8s}:  {1}  {2}".format(row[0], row[1], row[2]))


# Pandas
print("\nUsing Pandas:\n-------------------------\n")
t = pd.read_sql_query("Select Name, Ra, Decl From MagTable Where B > 16", con)
print(t)


#----------------
# Question b)
#  Output B, R, Teff and FeH for all stars (is this question well-defined?).
#----------------
print("\n\nQuestion b)")
query = """
SELECT m.B, m.R, p.Teff, p.FeH
FROM MagTable as m OUTER LEFT JOIN PhysTable as p
ON m.Name = p.Name
"""
t = pd.read_sql_query(query, con)

#-------------
# Question c)
#  Output the same as in b) for all objects with FeH > 0
#-------------
print("\n\nQuestion c)")
query = """
SELECT m.B, m.R, p.Teff, p.FeH
FROM MagTable as m OUTER LEFT JOIN PhysTable as p
ON m.Name = p.Name 
WHERE FeH > 0
"""
t = pd.read_sql_query(query, con)

#-------------
# Question d)
#  Create a table with the B-R colour
#-------------

# First get one using a query
col_t = pd.read_sql_query("SELECT Name, B-R as BR FROM MagTable", con)

# Then save it
col_t.to_sql("BRTable", con, if_exists="replace")
