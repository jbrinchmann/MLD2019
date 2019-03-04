# Machine learning and Databases at CAUP/IA in 2019

## Background


## Make a copy of the repository that you can edit

In this case you will want to *fork* the repository rather than just clone this. You can follow the instructions below (credit to Alexander Mechev for this) to create a fork of the repository:

-    Make a github account and log in.
-    Click on the 'Fork' at the top right. This will create a 'fork' on your own account. That means that you now have the latest commit of the repo and its history in your control. If you've tried to 'git push' to the DDM2017 repo you'd have noticed that you don't have access to it.
-    Once it's forked, you can go to your github profile and you'll see a DDM2017 repo. Go to it and get the .git link (green button)
-    Somewhere on your machine, git clone git clone https://github.com/[YOUR_GIT_UNAME]/DDM2017.git. You also need to enter the directory
-    Add our repo as an upstream. That way you can get (pull) new updates: git remote add upstream https://github.com/jbrinchmann/DDM2017.git
-    git remote -v should give: origin https://github.com/[YOUR_GIT_UNAME]/DDM2017.git (fetch) origin https://github.com/[YOUR_GIT_UNAME]/DDM2017.git (push) upstream https://github.com/jbrinchmann/DDM2017.git (fetch) upstream https://github.com/jbrinchmann/DDM2017.git (push)
-    Now you're ready to add files and folders to your local fork. Use git add, git commit and git push (origin master) to add your assignments.


## Lecture 1 - links and information

The slides are available in the Lectures directory.


- The data file for the Star database for sqlite ingest - YAEPS.stars-table-sqlite.dat. If you want a version with header suitable for MySQL try this<LINK>
- The data file for the Observations database for sqlite ingest - YAEPS.observations-table-sqlite.dat. If you want a version with header suitable for MySQL try this
- The SQL code to create the Star table in sqlite3 - sqlite3-make-stars-table.sql. For MySQL you would use myqsl-make-stars-table.sql
- The SQL code to create the Observations table in sqlite3 - sqlite3-make-observations-table.sql.  For MySQL you would use myqsl-make-observations-table.sql.
- A small intro to python with some simple tasks can be found here, with a possible solution that uses a driver routine as well as a and a collection of Python functions that this uses.
- A simple introductory problem set to familiarise you with various astronomical data archives/databases and get more experience with SQL.


