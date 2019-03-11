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

.separator ,
.import YAEPS.stars-table-sqlite.dat Stars

