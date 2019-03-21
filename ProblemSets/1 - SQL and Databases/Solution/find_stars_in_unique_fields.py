#!/usr/bin/env python

import sqlite3 as lite

# Get this from the MakeTables directory if you haven't
# got this already
con = lite.connect('MLD2019.db')

# Find the unique field IDs
rows = con.execute('Select DISTINCT ID From Observations')
fieldIDs = [row[0] for row in rows]

# Loop over the fieldIDs and find those that have
# stars in the Stars table.
for fid in fieldIDs:
    SQLstatement = """
SELECT Star 
FROM Stars
WHERE FieldID = "{0}"
""".format(fid)

    rows = con.execute(SQLstatement)

    # This gives us the number of stars in field i
    starIDs = [row[0] for row in rows] 
    if len(starIDs)>0: 
        print("Field {0} has {1} stars".format(fid, len(starIDs))) 
        for sid in starIDs: 
            print("    {0}".format(sid)) 

        
