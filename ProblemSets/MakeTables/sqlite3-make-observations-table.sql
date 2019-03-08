CREATE TABLE IF NOT EXISTS Observations (ID INT,
       	     	    Field varchar(10),
  		    date DOUBLE, 
		    exptime FLOAT,
		    quality FLOAT, 
  		    WhereStored varchar(256),
		    UNIQUE (ID),
		    PRIMARY KEY (ID)
			);
.separator ","
.import YAEPS.observations-table-sqlite.dat Observations
