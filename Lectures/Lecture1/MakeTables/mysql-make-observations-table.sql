USE DDM15;
CREATE TABLE IF NOT EXISTS Observations (ID INT,
       	     	    Field varchar(10),
  		    date DOUBLE, 
		    exptime FLOAT,
		    quality FLOAT, 
  		    WhereStored varchar(256),
		    UNIQUE (ID),
		    PRIMARY KEY (ID)
		    );
LOAD DATA INFILE '/home/brinchmann/MySQL/YAEPS.observations-table.dat' INTO TABLE Observations
  FIELDS TERMINATED BY ',' IGNORE 1 LINES;

