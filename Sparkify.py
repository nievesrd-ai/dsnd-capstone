#!/usr/bin/env python
# coding: utf-8

# # Sparkify Project Workspace
# This workspace contains a tiny subset (128MB) of the full dataset available (12GB). Feel free to use this workspace to build your project, or to explore a smaller subset with Spark before deploying your cluster on the cloud. Instructions for setting up your Spark cluster is included in the last lesson of the Extracurricular Spark Course content.
# 
# You can follow the steps below to guide your data analysis and model building portion of this project.

# In[1]:


# import libraries
import pyspark
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
from pyspark.sql.functions import sum as Fsum

import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark import SparkConf, SparkContext

# %%
# Helper function
def get_unique(df, col_name):
    """Return unique values in a columnn of a df in list format

    Args:
        df (pyspark.DataFrame): Data frame from which to pull uniques
        col_name (str): Name of the column to analyze

    Returns:
        list: Python list of unique values found
    """

    wrapped_unique = df.select(col_name).drop_duplicates().collect()
    uniques = [x.asDict()[col_name] for x in wrapped_unique]
    return uniques


# In[3]:
# conf = pyspark.SparkConf().setAppName("Sparkify")
# conf.set("spark.executor.heartbeatInterval", "3600s")
# %%

conf = pyspark.SparkConf().setAppName("Sparkify")
conf.set("spark.network.timeout","6600s")
conf.set("spark.executor.heartbeatInterval","3600s")
conf.set("spark.rpc.lookupTimeout", "3600s")
# conf.set("spark.driver.cores", 1)

# 
# %%
# create a Spark session
spark_context = pyspark.SparkContext(conf=conf) # uses 4 cores on your local machine
spark = SQLContext(spark_context)
# spark = SparkSession.builder.appName("Sparkify").heartbeatInterval("3600s").getOrCreate()
# spark = pyspark.SparkContext(conf=conf)

# spark = SparkSession.builder.config(spark_context.getConf()).getOrCreate()

# # Load and Clean Dataset
# In this workspace, the mini-dataset file is `mini_sparkify_event_data.json`. Load and clean the dataset, checking for invalid or missing data - for example, records without userids or sessionids. 

# In[2]:

path = "mini_sparkify_event_data.json"
user_log = spark.read.json(path)
# %%
# Dropping columns that might not help with the goals of the project
columns_to_drop = ['userAgent', 'method', 'registration']
user_log = user_log.drop(*columns_to_drop)
number_of_records = user_log.count()
user_log.head()
# %%
# Event types
event_types = get_unique(user_log, 'page')
# Users
users = get_unique(user_log, 'userID')

# %%
# # Exploratory Data Analysis
# When you're working with the full dataset, perform EDA by loading a small subset of the data and doing basic manipulations within Spark. In this workspace, you are already provided a small subset of data you can explore.
# 
# ### Define Churn
# 
# Once you've done some preliminary analysis, create a column `Churn` to use as the label for your model. I suggest using the `Cancellation Confirmation` events to define your churn, which happen for both paid and free users. As a bonus task, you can also look into the `Downgrade` events.
# %%
churn_flagger = udf(lambda x: 1 if x == 'Cancellation Confirmation' else 0, IntegerType())
user_log = user_log.withColumn("churn", churn_flagger(user_log.page)).collect()

# chrun_array = user_log.where(user_log.page=='Cancellation Confirmation').collect()
# user_log = user_log.withColumn('churn', churn_array).collect()
# user_log.head()
# %%
# ### Explore Data
# Once you've defined churn, perform some exploratory data analysis to observe the behavior for users who stayed vs users who churned. You can start by exploring aggregates on these two groups of users, observing how much of a specific action they experienced per a certain time unit or number of songs played.

# In[ ]:





# # Feature Engineering
# Once you've familiarized yourself with the data, build out the features you find promising to train your model on. To work with the full dataset, you can follow the following steps.
# - Write a script to extract the necessary features from the smaller subset of data
# - Ensure that your script is scalable, using the best practices discussed in Lesson 3
# - Try your script on the full data set, debugging your script if necessary
# 
# If you are working in the classroom workspace, you can just extract features based on the small subset of data contained here. Be sure to transfer over this work to the larger dataset when you work on your Spark cluster.

# In[ ]:





# # Modeling
# Split the full dataset into train, test, and validation sets. Test out several of the machine learning methods you learned. Evaluate the accuracy of the various models, tuning parameters as necessary. Determine your winning model based on test accuracy and report results on the validation set. Since the churned users are a fairly small subset, I suggest using F1 score as the metric to optimize.

# In[ ]:





# # Final Steps
# Clean up your code, adding comments and renaming variables to make the code easier to read and maintain. Refer to the Spark Project Overview page and Data Scientist Capstone Project Rubric to make sure you are including all components of the capstone project and meet all expectations. Remember, this includes thorough documentation in a README file in a Github repository, as well as a web app or blog post.

# In[ ]:




