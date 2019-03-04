CREATE TABLE IF NOT EXISTS Stars (StarID INT,
			 FieldID INT,
       	     	    	 Star varchar(10),
  		    	 ra DOUBLE,
			 decl DOUBLE,
		   	 g FLOAT, 
			 r FLOAT,
			 UNIQUE(StarID),
			PRIMARY KEY(StarID),
			FOREIGN KEY(FieldID) REFERENCES Observations(ID));
LOAD DATA INFILE '/home/brinchmann/MySQL/YAEPS.stars-table.dat' INTO TABLE Stars
  FIELDS TERMINATED BY ',' IGNORE 1 LINES;

