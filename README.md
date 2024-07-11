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
The easiest way to setup the environment is by using Anaconda, a distribution of Python for scientific computing. You can use Miniconda, a minimal installer for conda as well if you do not want the pre-installed packages that come with Anaconda. 
## Using Anaconda/Mamba
1.	Install Anaconda/Miniconda: Anaconda can be installed from https://www.anaconda.com/products/distribution. Depending on your operating system choose the corresponding file and follow instructions. Or you can install Miniconda from here: https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links.
2.	Open conda prompt: To open Anaconda Prompt (or terminal on Linux or macOS):
    1.	Windows: Open the Anaconda Prompt (Start >> Anaconda Prompt)
    2.	macOS: Open Launchpad and then open Terminal. Type `conda activate`
    3.	Linux: Open Terminal. Type `conda activate`
3. Install Mamba (Recommended as you may face problem resolving dependencies with conda): Mamba is a fast, robust, and cross-platform package manager. It runs on Windows, OS X and Linux (ARM64 and PPC64LE included) and is fully compatible with conda packages and supports most of conda’s commands (just replace `conda` with `mamba`). You can install it using the following command:
`conda install mamba -n base -c conda-forge`
4. Create a new environment: Use the following command to create a new environment of your choice. For instance, to create an environment named `modern_ts_2E` with Python 3.10 (recommended touse 3.10 or above), use the following command:
`mamba create -n modern_ts_2E python=3.10`
5. Activate the environment: Use the following command to activate the environment:
`conda activate modern_ts_2E`
6. Install Pytorch from the official website: Pytorch is best installed from the official website. Go to https://pytorch.org/get-started/locally/ and select the appropriate options for your system. you can replace `conda` with `mamba` if you want to use mamba to install.
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
You are going to be using a single dataset throughout the book. The book uses London Smart Meters Dataset from Kaggle for this purpose. Therefore, if you don’t have an account with Kaggle, please go ahead and make one. https://www.kaggle.com/account/login?phase=startRegisterTab
There are two ways you can download the data- automated and manual. 
For the automated way, we need to download a key from Kaggle. Let’s do that first (if you are going to choose the manual way, you can skip this).
1.	Click on your profile picture on the top right corner of Kaggle
2.	Select "Account”, and find the section for “API”
3.	Click the “Create New API Token” button. A file by the name kaggle.json will be downloaded.
4.	Copy the file and place it in the api_keys folder in the downloaded code folder.
Now that we have the kaggle.json downloaded and placed in the right folder, let’s look at the three methods to download data:
## Method 1: Automated Download
1.	Activate the environment using `conda activate modern_ts_2E`
2.	Run the provided script from the root directory of downloaded code `python scripts/download_data.py`
That’s it. Now just wait for the script to finish downloading, unzipping and organize the files in the expected format.
## Method 2: Manual Download
1.	Go to https://www.kaggle.com/jeanmidev/smart-meters-in-london and download the dataset
2.	Unzip the contents to data/london_smart_meters
3.	Unzip hhblock_dataset to get the raw files we want to work with.
4.	Make sure the unzipped files are in the expected folder structure (next section)
Now that you have downloaded the data, we need to make sure it is arranged in the below folder structure. Automated Download does it automatically, but for Manual Download this structure needs to be created. To avoid ambiguity, the expected folder structure can be found below:
```
data
├── london_smart_meters
│   ├── hhblock_dataset
│   │   ├── hhblock_dataset
│   │       ├── block_0.csv
│   │       ├── block_1.csv
│   │       ├── ...
│   │       ├── block_109.csv
│   │── acorn_details.csv
│   ├── informations_households.csv
│   ├── uk_bank_holidays.csv
│   ├── weather_daily_darksky.csv
│   ├── weather_hourly_darksky.csv
```
There can be additional files as part of the extraction process. You can remove them without impacting anything. There is a helpful script which checks this structure.
python test_data_download.py

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

## Get to Know the Author
**Manu Joseph**
is a self-made data scientist with more than a decade of experience working with many
Fortune 500 companies, enabling digital and AI transformations, specifically in machine learningbased demand forecasting. He is considered an expert, thought leader, and strong voice in the world
of time series forecasting. Currently, Manu leads applied research at Thoucentric, where he advances
research by bringing cutting-edge AI technologies to the industry. He is also an active open source
contributor and has developed an open source library—PyTorch Tabular—which makes deep learning
for tabular data easy and accessible. Originally from Thiruvananthapuram, India, Manu currently
resides in Bengaluru, India, with his wife and son.


### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781803246802">https://packt.link/free-ebook/9781803246802 </a> </p>



