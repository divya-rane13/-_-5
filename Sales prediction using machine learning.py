#!/usr/bin/env python
# coding: utf-8

# #  Importing neccesaary python libraries
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing data 

# In[4]:


data = pd.read_csv("Advertising.csv")
data.head()


# In[15]:


newdata=data.drop("Unnamed: 0",axis=1)


# In[6]:


print(data.isnull().sum())


# # correlation between features

# In[20]:


plt.figure(figsize=(8,6))
sns.heatmap(newdata.corr())
plt.show()


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# # use a linear regression algorithm to train a sales prediction model 

# In[18]:


x = np.array(newdata.drop(["Sales"], axis=1))
y = np.array(newdata["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(xtrain, ytrain)
pred_value = model.predict(xtest)


# # Predicted Sales

# In[19]:


data = pd.DataFrame(data={"Predicted Sales": pred_value.flatten()})
print(data)


# In[ ]:




