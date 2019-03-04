# Machine learning and Databases at CAUP/IA in 2019

## Course overview

This course is an advanced course at CAUP during March and April 2019. Lectures will take place on Mondays at 14:00 while practical classes will take place on Thursdays at 10:00. Both have duration 2 hours with a short break.

The aim of this course is to get a good *practical* grasp of machine learning. I will not spend a lot of time on algorithm details but more on how to use these in python and try to discuss what methods are useful for what type of scientific question/research goal.

<dl>
<dt>March 4 - Managing data and simple regression</dt>
  <dd>
   <ul>
     <li> Covering git and SQL</li>
     <li> Introducing machine learning through regression techniques.</li>
   </ul>
  </dd>


<dt>March 11 - Visualisation and inference methods</dt>
  <dd>
   <ul>
     <li>  Visualisation of data, do's and don't's </li>
     <li>  Classical inference </li>
     <li>  Bayesian inference </li>
     <li>  MCMC </li>
   </ul>
  </dd>

<dt>March 18 - Density estimation and model choice</dt>
  <dd>
   <ul>
     <li>  Estimating densities, parametric & non-parametric </li>
     <li>  Bias-variance trade-off </li>
     <li>  Cross-validation </li>
     <li>  Classification </li>
   </ul>
  </dd>

<dt>March 25 - Dimensional reduction</dt>
  <dd>
   <ul>
     <li>  Standardising data. </li>
     <li>  Principal Component Analysis </li>
     <li>  Manifold learning </li>
   </ul>
  </dd>

<dt>April 8 - Ensemble methods, neural networks, deep learning</dt>
  <dd>
   <ul>
     <li>  Local regression methods </li>
     <li>  Random forests and other boosting methods </li>
     <li>  Neural networks & deep learning </li>
   </ul>
  </dd>
</dl>


## Literature for the course


Below you can find some books of use. The links from the titles get you to the Amazon page. If there are free versions of the books legally available online, I include a link as well.


- I base myself partially on ["Statistics, Data Mining, and Machine Learning in Astronomy" - Ivezic, Connolly, VanderPlas &amp; Gray](http://www.amazon.co.uk/Statistics-Mining-Machine-Learning-Astronomy/dp/0691151687/ref=sr_1_1?ie=UTF8&amp;qid=1444255176&amp;sr=8-1&amp;keywords=Statistics%2C+Data+Mining%2C+and+Machine+Learning+in+Astronomy+-+Ivezic%2C+Connolly%2C+VanderPlas+%26+Gray)

- I have also consulted ["Deep Learning" - Goodfellow, Bengio &amp; Courville](https://www.amazon.co.uk/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=sr_1_1?ie=UTF8&amp;qid=1505297517&amp;sr=8-1&amp;keywords=Deep+Learning)

- ["Pattern Classification" - Duda, Hart &amp; Stork](http://www.amazon.co.uk/Pattern-Classification-Second-Wiley-Interscience-publication/dp/0471056693/ref=sr_1_1?ie=UTF8&amp;qid=1444255264&amp;sr=8-1&amp;keywords=Pattern+Classification), is a classic in the field

- ["Pattern Recognition and Machine Learning" - Bishop](http://www.amazon.co.uk/Pattern-Recognition-Machine-Learning-BISHOP/dp/8132209060/ref=sr_1_1?ie=UTF8&amp;qid=1444255326&amp;sr=8-1&amp;keywords=Pattern+Recognition+and+Machine+Learning+-+Bishop), is a very good and comprehensive book. Personally I really like this one.

- ["Bayesian Data Analysis" - Gelman](http://www.amazon.co.uk/Bayesian-Analysis-Chapman-Statistical-Science/dp/1439840954/ref=sr_1_1?ie=UTF8&amp;qid=1444255416&amp;sr=8-1&amp;keywords=Bayesian+Data+Analysis+-+Gelman), is often the first book you are pointed to if you ask questions about Bayesian analysis.

- ["Information Theory, Inference and Learning Algorithms" - MacKay](http://www.amazon.co.uk/Information-Theory-Inference-Learning-Algorithms/dp/0521642981/ref=sr_1_1?ie=UTF8&amp;qid=1444255466&amp;sr=8-1&amp;keywords=Information+Theory%2C+Inference+and+Learning+Algorithms), is a very readable book on a lot of related topics. The book is also [freely available](http://www.inference.phy.cam.ac.uk/itila/book.html) on the web.

- ["Introduction to Statistical Learning - James et al"](http://www.amazon.co.uk/Introduction-Statistical-Learning-Applications-Statistics/dp/1461471370/ref=sr_1_fkmr0_1?ie=UTF8&amp;qid=1444255565&amp;sr=8-1-fkmr0&amp;keywords=Introduction+to+Statistical+Learning+-+James+et+al) is a readable introduction (fairly basic) to statistical technique of relevance. It is also [freely available](http://www-bcf.usc.edu/~gareth/ISL/) on the web.

-["Elements of Statistical Learning - Hastie et al](http://www.amazon.co.uk/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576/ref=sr_1_1?ie=UTF8&amp;qid=1444255710&amp;sr=8-1&amp;keywords=Elements+of+Statistical+Learning), is a  more advanced version of the Introduction to Statistical Learning with much the same authors. This is also [freely available](http://statweb.stanford.edu/~tibs/ElemStatLearn/) on the web.

- ["Bayesian Models for Astrophysical Data", Hilbe, Souza & Ishida](https://www.amazon.com/Bayesian-Models-Astrophysical-Data-Python/dp/1107133084) is a good reference book for a range of Bayesian techniques and is a good way to learn about different modelling frameworks for Bayesian inference. 


## Making a copy of the repository that you can edit

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


