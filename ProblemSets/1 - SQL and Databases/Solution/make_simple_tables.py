#!/usr/bin/env python

# Create the simple tables requested and run the necessary queries.
#
# I will actually create the tables in two different ways - one using
# the standard python approach we have seen before, and one using Pandas.
#

# I import the print function here so that the code works
# with both python2 and python3
from __future__ import print_function
import sqlite3 as lite
from astropy.table import Table
import pandas as pd

def is_number(s):
    # I need a function like this below to decide whether
    # to insert apostrophes around the arguments in SQL.
    try:
        float(s)
        return True
    except ValueError:
        return False

#
# Standard Python creation
#

# I will simplify the creation a bit - for readability I define the schemas here
MagSchema = """CREATE TABLE IF NOT EXISTS MagTable (Name varchar(6),
       Ra varchar(12),
       Decl varchar(12),
       B Float,
       R Float,
       UNIQUE(Name));
"""
# Note that I decided to modify the table somewhat - that is ok, and
# split off the unit as a separate quantity.
PhysSchema = """
CREATE TABLE IF NOT EXISTS PhysTable (Name varchar(6),
       Teff Float,
       Unit varchar(1),
       FeH Float,
       UNIQUE(Name));
"""


# I now define my tables through a dict. Each element in the dict
# contains the name of the file to read the data from and the
# schema that we want to use.
tables = {'MagTable': ['MagTable.csv', MagSchema],
            'PhysTable': ['PhysTable.csv', PhysSchema]}

con = lite.connect('SimpleTables-default.db')
with con:
    for name in tables.keys():
        file_name, schema = tables[name]
        print("I will read from {0}".format(file_name))
        t = Table().read(file_name, format='csv')

        con.execute(schema)
        for row in t:
            command = "INSERT INTO {0} VALUES(".format(name)
            n_columns = len(row)
            for i, col in enumerate(row):
                # Now the trick here is how to handle strings. Numbers do
                # not need to be enclosed in apostrophes so I'll just do
                # the simple check
                if is_number(col):
                    arg = str(col)
                else:
                    arg = "'"+str(col)+"'"
                    
                command = command+arg
                if i < n_columns-1:
                    command = command+','

            command = command+')'
            
            try:
                print("Command = {0}".format(command))
                con.execute(command)
            except:
                pass
            


#
# Create the tables using Pandas.
#
# To do this, we first need to create Pandas data frame. Luckily this is
# very easy from astropy tables
#
MagTable = Table().read('MagTable.csv', format='csv')
PhysTable = Table().read('PhysTable.csv', format='csv')

# Conversion to Pandas data frames.
df_MagTable = MagTable.to_pandas()
df_PhysTable = PhysTable.to_pandas()

# Finally, create the connection
con = lite.connect('SimpleTables-pandas.db')

# And create the tables.
df_MagTable.to_sql("MagTable", con, if_exists='replace')
df_PhysTable.to_sql("PhysTable", con, if_exists='replace')
