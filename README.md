# Machine learning and Databases at CAUP/IA in 2019


We have started!

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

I expect that you have read through these two documents:
- [A couple of Python & Topcat pointers](Texts/Python-Topcat/Python-Topcat-MLD2019.pdf). This is a very basic document and might not contain a lot of new stuff. It does have a couple of tasks to try out - the solution for these you can find in the [ProblemSets/0 - Pyton and Topcat](ProblemSets/0 - Pyton and Topcat) directory. 

- [A reminder/intro to relevant math](Texts/MathIntro/math_reminders.pdf) contains a summary of some basic facts from linear algebra and probability theory that are useful for this course.

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
-    Click on the 'Fork' at the top right. This will create a 'fork' on your own account. That means that you now have the latest commit of the repo and its history in your control. If you've tried to 'git push' to the MLD2019 repo you'd have noticed that you don't have access to it.
-    Once it's forked, you can go to your github profile and you'll see a MLD2019 repo. Go to it and get the .git link (green button)
-    Somewhere on your machine, git clone git clone https://github.com/[YOUR_GIT_UNAME]/MLD2019.git. You also need to enter the directory
-    Add our repo as an upstream. That way you can get (pull) new updates: git remote add upstream https://github.com/jbrinchmann/MLD2019.git
-    git remote -v should give: origin https://github.com/[YOUR_GIT_UNAME]/MLD2019.git (fetch) origin https://github.com/[YOUR_GIT_UNAME]/MLD2019.git (push) upstream https://github.com/jbrinchmann/MLD2019.git (fetch) upstream https://github.com/jbrinchmann/MLD2019.git (push)
-    Now you're ready to add files and folders to your local fork. Use git add, git commit and git push (origin master) to add your assignments.

## Software you need for the course

The course will make use of python throughout, and for this you need a recent version of python installed. I use python 3 by default but will try to make all scripts compatible with python 2 and python 3. For python you will need (well, I recommend it at least) at least these libraries installed:

- [numpy](http://www.numpy.org/) - for numerical calculations
- [astropy](http://www.astropy.org/) - because we are astronomers
- [scipy](https://www.scipy.org/) - because we are scientists 
- [sklearn](https://scikit-learn.org/) - Machine learning libraries with full name scikit-learn.
- [matplotlib](https://matplotlib.org/) - plotting (you can use alternatives of course)
- [pandas](https://pandas.pydata.org/) - nice handling of data
- [seaborn](https://seaborn.pydata.org/) - nice plots

(the last two are really "nice to have" but if you can install the others then these are easy).

You should also get `astroML` which has a nice web page at XX and a git repository at [https://github.com/astroML/astroML](https://github.com/astroML/astroML) 

It turns out that the astroML distribution that is often picked up when you install it using a package manager (maybe also pip?) is outdated and does not work with new versions of sklearn. To check whether you have a problem, try:

`from astroML.datasets import fetch_sdss_sspp`

If this crashes with a complaint about a module GMM, you have the old version. To fix this the best way is probably to check out the git version of astroML linked above using e.g.:

`git clone https://github.com/astroML/astroML.git`



To use astroML in Anaconda you need to get it from the astropy channel. For a one-off you can do:

`conda install -c astropy astroML`

If you want to add the astropy channel permanently (which probably is a good idea), you can do:

`conda config --add channels astropy`

## Lecture 1 - links and information

The slides are available in the Lectures directory. You can find some files for creating tables in the ProblemSets/MakeTables directory. 


## Lecture 2-4

The slides are available in the Lectures directory.

# Getting ready for deep learning in python

In the final problem class we will look at using deep learning in python. There are quite a few libraries for this around but we will use the most commonly used one, [TensorFlow](https://www.tensorflow.org/) and we will use the [keras](https://keras.io/) python package for interacting with TensorFlow. Keras is a high-level interface (and can also use other libraries, [Theano](https://github.com/Theano/Theano) and [CNTK](https://github.com/Microsoft/cntk), in addition to TensorFlow). 

There are many pages that detail the installation of these packages and what you need for them. A good one with a bias towards Windows is [this one](https://towardsdatascience.com/setup-an-environment-for-machine-learning-and-deep-learning-with-anaconda-in-windows-5d7134a3db10). I will give a very brief summary here of how I set things up. This is not optimised for Graphical Processing Unit (GPU) work so for serious future work you will need to adjust this.

### Create an environment in anaconda 

I am going to assume you use anaconda for your python environment. If not, you need to change this section a bit - use virtualenv instead of setting up a conda environment.  It is definitely better to keep your TensorFlow/keras etc setup out of your default Python work environment. Most of the packages are also installed with `pip` rather than conda, so what I use is

`conda create -n tensorflow pip python=3.6`

This creates an environment called tensorflow which uses python 3.6 and `pip` for installation. To use this we need to activate it first:

`activate tensorflow`

(assuming you use bash - I do not so I need to do some more tricks. Use bash). Your prompt should not change to include `(tensorflow)`.


### Install tensorflow and keras

I went for the simplest approach here:

`pip install --upgrade tensorflow`

This takes a while - the package is fairly large, 71.6Mb in my installation, and it requires a fair number of additional packages.

`pip install keras`

This is quicker.

`pip install ipython`

because that is not installed by default (you can skip this if you prefer not to use ipython).

`pip install jupyter`

because my example is a jupyter notebook.

You will also need to install some other packages I am sure you will need:

`pip install matplotlib`

`pip install astropy`

`pip install pandas`

`pip install sklearn`

`pip install seaborn`

and you might have others that you want to use but that should set up you fairly well for deep learning. 



###

