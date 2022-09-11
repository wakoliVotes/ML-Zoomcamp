#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import requests
import io

get_ipython().run_line_magic('matplotlib', 'inline')


# #### Question 1
# What's the version of NumPy that you installed?

# In[6]:


np.__version__


# #### Getting the data

# In[30]:


# Using requests library
url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'
response = requests.get(url).content


# In[71]:


# Reading the dataset
df=  pd.read_csv(io.StringIO(response.decode('utf-8')))


# In[72]:


# An overview of the dataset
df.head()


# #### Question 2
# How many records are in the dataset?
# 
# Here you need to specify the number of rows

# In[29]:


df.describe()


# In[12]:


df.info()


#  - From above, RangeIndex gives **11914**

# ### Question 3
# Who are the most frequent car manufacturers (top-3) according to the dataset?

# In[13]:


df.Make.value_counts()


# - Chevrolet (1123), Ford (881) and Volkswagen (809) are top 3 based on the value_counts

# #### Question 4
# What's the number of unique Audi car models in the dataset?

# In[14]:


# Getting unique items
df.loc[df['Make'] == 'Audi']['Model'].nunique()


# - 34 Audi models are unique

# #### Question 5
# How many columns in the dataset have missing values?

# In[15]:


df.isnull().sum()


# - Answer is  **5**.
# - From above, by manually counting, we have 5 columns have integers other than zero (0), i.e., Engine Fuel Type 3, Engine HP 69, Engine Cylinders 30, Number of Doors 6 and Market Category 3742 have missing values.
# - Hence, answer is **5**

# #### Question 6: Does the median value change after filling missing values?
# 1. Find the median value of "Engine Cylinders" column in the dataset.
# 2. Next, calculate the most frequent value of the same "Engine Cylinders".
# 3. Use the fillna method to fill the missing values in "Engine Cylinders" with the most frequent value from the previous step.
# 4. Now, calculate the median value of "Engine Cylinders" once again.
# 
# - Has it changed?
# 

# In[50]:


median_engine_cylinders = df['Engine Cylinders'].median()
median_engine_cylinders


# In[51]:


mode_engine_cylinders = df['Engine Cylinders'].mode()
mode_engine_cylinders


# In[73]:


# Copyin the dataset
clean_cylinders = df['Engine Cylinders']


# In[90]:


# Checking empty rows in the column Engine Cylinders
clean_cylinders.isna().sum()


# In[97]:


clean_cylinders.fillna(mode_engine_cylinders, inplace=True)


# In[98]:


clean_cylinders.median()


# - No change, the median is the same, i.e.,  **6.0**

# #### Questions 7: Value of the first element of w*
# 
# - Select all the "Lotus" cars from the dataset.
# - Select only columns "Engine HP", "Engine Cylinders".
# - Now drop all duplicated rows using drop_duplicates method (you should get a dataframe with 9 rows).
# - Get the underlying NumPy array. Let's call it X.
# - Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
# - Invert XTX.
# - Create an array y with values [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800].
# - Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
# - What's the value of the first element of w?
# 

# In[100]:


df_lotus = df[df['Make'] == 'Lotus']
df_lotus.sample(5)


# In[101]:


lotus_df_columns = df_lotus[['Engine HP', 'Engine Cylinders']]
lotus_df_columns.head()


# In[102]:


lotus_df_columns.duplicated().sum()


# In[103]:


# Replacing with inplace=True
lotus_df_columns.drop_duplicates(inplace=True)


# In[104]:


# Checking to confirm is changes are effective
lotus_df_columns.duplicated().sum()


# In[105]:


# Checking rows, should give 9 based on instructions
lotus_df_columns.info()


# In[106]:


# Get the underlying NumPy array. Let's call it X.
X = lotus_df_columns.to_numpy()


# In[107]:


# Displaying X
X


# In[108]:


# Compute matrix-matrix multiplication between the transpose of X and X.
# To get the transpose, use X.T. Let's call the result XTX.
XT = X.T
XT


# In[109]:


XTX = np.dot(XT, X)
XTX


# In[115]:


# Inverting XTX
XTX_Inverse = np.linalg.inv(XTX)
XTX_Inverse


# In[116]:


y = np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])


# In[117]:


w = np.dot(np.dot(XTX_Inverse, XT), y)


# In[118]:


# Displaying w
w


# - Hence, from above, the first value in **w** is **4.59494481**

# #### Saving Dataframe to csv for future use

# In[114]:


df.to_csv('data_data.csv')

