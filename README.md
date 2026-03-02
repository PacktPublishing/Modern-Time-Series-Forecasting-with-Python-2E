<p align='center'><a href='https://www.packtpub.com/en-us/unlock?step=1'><img src='https://static.packt-cdn.com/assets/images/packt+events/finalGH_design_redeem.png'/></a></p>

<p align="center"><a href="https://packt.link/mlsumgh"><img src="https://static.packt-cdn.com/assets/images/ML Summit Banner v3 1200x627.png" alt="Machine Learning Summit 2025"/></a></p>

## Machine Learning Summit 2025
**Bridging Theory and Practice: ML Solutions for Today’s Challenges**

3 days, 20+ experts, and 25+ tech sessions and talks covering critical aspects of:
- **Agentic and Generative AI**
- **Applied Machine Learning in the Real World**
- **ML Engineering and Optimization**

👉 [Book your ticket now >>](https://packt.link/mlsumgh)

---

## Join Our Newsletters 📬

### DataPro  
*The future of AI is unfolding. Don’t fall behind.*

<p><a href="https://landing.packtpub.com/subscribe-datapronewsletter/?link_from_packtlink=yes"><img src="https://static.packt-cdn.com/assets/images/DataPro NL QR Code.png" alt="DataPro QR" width="150"/></a></p>

Stay ahead with [**DataPro**](https://landing.packtpub.com/subscribe-datapronewsletter/?link_from_packtlink=yes), the free weekly newsletter for data scientists, AI/ML researchers, and data engineers.  
From trending tools like **PyTorch**, **scikit-learn**, **XGBoost**, and **BentoML** to hands-on insights on **database optimization** and real-world **ML workflows**, you’ll get what matters, fast.

> Stay sharp with [DataPro](https://landing.packtpub.com/subscribe-datapronewsletter/?link_from_packtlink=yes). Join **115K+ data professionals** who never miss a beat.

---

### BIPro  
*Business runs on data. Make sure yours tells the right story.*

<p><a href="https://landing.packtpub.com/subscribe-bipro-newsletter/?link_from_packtlink=yes"><img src="https://static.packt-cdn.com/assets/images/BIPro NL QR Code.png" alt="BIPro QR" width="150"/></a></p>

[**BIPro**](https://landing.packtpub.com/subscribe-bipro-newsletter/?link_from_packtlink=yes) is your free weekly newsletter for BI professionals, analysts, and data leaders.  
Get practical tips on **dashboarding**, **data visualization**, and **analytics strategy** with tools like **Power BI**, **Tableau**, **Looker**, **SQL**, and **dbt**.

> Get smarter with [BIPro](https://landing.packtpub.com/subscribe-bipro-newsletter/?link_from_packtlink=yes). Trusted by **35K+ BI professionals**, see what you’re missing.

# Modern-Time-Series-Forecasting-with-Python-2E
Modern Time Series Forecasting with Python 2E, Published by Packt


# Modern Time Series Forecasting with Python

<a href="https://www.packtpub.com/en-us/product/modern-time-series-forecasting-with-python-9781835883181"><img src="https://content.packt.com/_/image/original/B22389/cover_image_large.jpg" alt="Modern Time Series Forecasting with Python 2nd Edition" height="256px" align="right"></a>

This is the code repository for [Modern Time Series Forecasting with Python](https://www.packtpub.com/en-us/product/modern-time-series-forecasting-with-python-9781835883181), published by Packt.

**Explore industry-ready time series forecasting using modern machine learning and deep learning**

## What is this book about?
We live in a serendipitous era where the explosion in the quantum of data collected and a renewed interest in data-driven techniques such as machine learning (ML), 
has changed the landscape of analytics, and with it, time series forecasting. This book, filled with industry-tested tips and tricks, 
takes you beyond commonly used classical statistical methods such as ARIMA and introduces to you the latest techniques from the world of ML.

This book covers the following exciting features: 
* Find out how to manipulate and visualize time series data like a pro
* Set strong baselines with popular models such as ARIMA
* Discover how time series forecasting can be cast as regression
* Engineer features for machine learning models for forecasting
* Explore the exciting world of ensembling and stacking models
* Get to grips with the global forecasting paradigm
* Understand and apply state-of-the-art DL models such as N-BEATS and Autoformer
* Explore multi-step forecasting and cross-validation strategies

If you feel this book is for you, get your [copy](https://a.co/d/0guEbBsv) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>


## Instructions and Navigations
All of the code is organized into folders.

The code will look like the following:
```
#Does not support missing values, so using imputed ts instead
res = seasonal_decompose(ts, period=7*48, model="additive",
extrapolate_trend="freq")
```

**Following is what you need for this book:**
The book is for data scientists, data analysts, machine learning engineers, and Python developers who want to build industry-ready time series models. Since the book explains most concepts from the ground up, basic proficiency in Python is all you need. Prior understanding of machine learning or forecasting will help speed up your learning. 
For experienced machine learning and forecasting practitioners, this book has a lot to offer in terms of advanced techniques and traversing the latest research frontiers in time series forecasting.	



# Setup the environment
Setting up an environment, preferably a separate one, for the book is highly recommended. There are two main ways we suggest to create the environment – _Anaconda/Mamba_ or _Python Virtual Environment_.

The easiest way to set up an environment is by using _Anaconda_, a distribution of Python for scientific computing. You can use _Minicond_a_, a minimal installer for Conda, as well if you do not want the pre-installed packages that come with Anaconda. And you can also use _Mamba_, a reimplementation of the conda package manager in C++. It is much faster than conda and is a drop-in replacement for conda. _Mamba_ is the recommended way because it has much less chances of getting stuck at the dreaded **“Resolving dependencies…”** screen in _Anaconda_. If you are using Anaconda version 23.10 or above, then you need not worry about Mamba that much because the fast and efficient package resolver is part of anaconda by default.

## Using Anaconda/Mamba
1.	1.	Install Anaconda/Miniconda/Mamba/MicroMamba: Anaconda can be installed from https://www.anaconda.com/products/distribution. Depending on your operating system, choose the corresponding file and follow the instructions. Alternatively, you can install Miniconda from here: https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links. For Mamba and MicroMamba you can install from here: https://mamba.readthedocs.io/en/latest/ . In case you are using Mamba, in all the instructions below replace `conda` with `mamba`.
2.	Open conda prompt: To open Anaconda Prompt (or terminal on Linux or macOS):
    1.	Windows: Open the Anaconda Prompt (Start >> Anaconda Prompt)
    2.	macOS: Open Launchpad and then open Terminal. Type `conda activate`
    3.	Linux: Open Terminal. Type `conda activate`
3. Create a new environment: Use the following command to create a new environment of your choice. For instance, to create an environment named `modern_ts_2E` with Python 3.10 (recommended to use 3.10 or above), use the following command:
`conda create -n modern_ts_2E python=3.10`
4. Activate the environment: Use the following command to activate the environment:
`conda activate modern_ts_2E`
6. Install PyTorch from the official website: PyTorch is best installed from the official website. Go to https://pytorch.org/get-started/locally/ and select the appropriate options for your system. you can replace `conda` with `mamba` if you want to use mamba to install.
7.	Navigate to the downloaded code: Use operating system specific commands to navigate to the folder where you have downloaded the code. For instance, in Windows, use `cd`.
7. Install the required libraries: Use the provided `anaconda_env.yml` file to install all the required libraries. Use the following command:
`mamba env update --file anaconda_env.yml`
This will install all the required libraries in the environment. This can take a while.
8.	Checking the installation: We can check if all the libraries required for the book is installed properly by executing a script in the downloaded code folder
`python test_installation.py`. If the GPU is not showing up, install PyTorch again on top of the environment.
9.	Activating the environment and Running Notebooks: Every time you want to run the notebooks, first activate the environment using the command `conda activate modern_ts_2E` and then use Jupyter Notebook (`jupyter notebook`) or Jupyter Lab (`jupyter lab`) according to your preference.

## Using Pip based environment
1.	Install Python: You can download Python from https://www.python.org/downloads/. Make sure to install Python 3.10 or above.
2. Create a virtual environment: Use the following command to create a virtual environment named `modern_ts_2E`:
`python -m venv modern_ts_2E`
3. Activate the environment: Use the following command to activate the environment:
    1. Windows: `modern_ts_2E\Scripts\activate`
    2. macOS/Linux: `source modern_ts_2E/bin/activate`
4. Install PyTorch: Pytorch is best installed from the official website. Go to https://pytorch.org/get-started/locally/ and select the appropriate options for your system.
5. Navigate to the downloaded code: Use operating system specific commands to navigate to the folder where you have downloaded the code. For instance, in Windows, use `cd`.
6. Install the required libraries: Use the provided `requirements.txt` file to install all the required libraries. Use the following command:
`pip install -r requirements.txt`
This will install all the required libraries in the environment. This can take a while.
7. Checking the installation: We can check if all the libraries required for the book is installed properly by executing a script in the downloaded code folder
`python test_installation.py`
8. Activating the environment and Running Notebooks: Every time you want to run the notebooks, first activate the environment using the command `modern_ts_2E\Scripts\activate` (Windows) or `source modern_ts_2E/bin/activate` (macOS/Linux) and then use Jupyter Notebook (`jupyter notebook`) or Jupyter Lab (`jupyter lab`) according to your preference.

# Download the Data
You are going to be using a single dataset throughout the book. The book uses London Smart Meters Dataset from Kaggle for this purpose. Many of the notebooks from early chapters are dependencies for some of the later chapters.  As such, to remove this dependency if you want to run the notebooks out of order, we have included a data.zip file with all of the required datasets. 
To setup, follow these steps:
1. Download the data from AWS: https://packt-modern-time-series-py.s3.eu-west-1.amazonaws.com/data.zip 
2. Unzip the content
3. Copy over the data folder to the Modern-Time-Series-Forecasting-with-Python-2E folder you pull from github. 

That's it!  You are now ready to start running the code.




# Blocks vs RAM

Number of blocks to select from the dataset is dependent on how much RAM you have in your machine. Although, these are not rules, but rough guidelines on how much blocks to choose based on your RAM is given below. If you still face problems, please experiment with lowering the number of blocks to make it work better for you.

* 1 or <1 Block for 4GB RAM
* 1 or 2 Blocks for 8GB RAM
* 3 Blocks for 16GB RAM
* 5 Blocks for 32GB RAM

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://packt.link/5NVrW).


### Related products <Other books you may enjoy>
* Intelligent Document Processing with AWS AI/ML [[Packt]](https://www.packtpub.com/product/intelligent-document-processing-with-aws-aiml/9781801810562) [[Amazon]](https://www.amazon.com/dp/1801810567)

* Practical Deep Learning at Scale with MLflow [[Packt]](https://www.packtpub.com/product/practical-deep-learning-at-scale-with-mlflow/9781803241333) [[Amazon]](https://www.amazon.com/dp/1803241330)

## Get to Know the Authors
**Manu Joseph**
is a self-made data scientist with more than a decade of experience working with many
Fortune 500 companies, enabling digital and AI transformations, specifically in machine learningbased demand forecasting. He is considered an expert, thought leader, and strong voice in the world
of time series forecasting. Currently, Manu leads applied research at Thoucentric, where he advances
research by bringing cutting-edge AI technologies to the industry. He is also an active open source
contributor and has developed an open source library—PyTorch Tabular—which makes deep learning
for tabular data easy and accessible. Originally from Thiruvananthapuram, India, Manu currently
resides in Bengaluru, India, with his wife and son.

**Jeff Tackes**
is a seasoned data scientist specializing in demand forecasting with over a decade of industry experience. Currently he is at Kraft Heinz, where he leads the research team in charge of demand forecasting.  He has pioneered the development of best-in-class forecasting systems utilized by leading Fortune 500 companies. Jeff’s approach combines a robust data-driven methodology with innovative strategies, enhancing forecasting models and business outcomes significantly. Leading cross-functional teams, Jeff has designed and implemented demand forecasting systems that have markedly improved forecast accuracy, inventory optimization, and customer satisfaction. His proficiency in statistical modeling, machine learning, and advanced analytics has led to the implementation of forecasting methodologies that consistently surpass industry norms. Jeff’s strategic foresight and his capability to align forecasting initiatives with overarching business objectives have established him as a trusted advisor to senior executives and a prominent expert in the data science domain. Additionally, Jeff actively contributes to the open-source community, notably to PyTimeTK, where he develops tools that enhance time series analysis capabilities.  He currently resides in Chicago, IL with his wife and son.

### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781803246802">https://packt.link/free-ebook/9781803246802 </a> </p>



