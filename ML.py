#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


df = pd.read_csv('iris.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.drop('Id',axis=1,inplace=True)


# In[8]:


class_names = ['iris setosa','Iris-versicolor','Iris-virginica']


# In[9]:


iris = load_iris()
X = iris.data
y = iris.target


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


y_pred = clf.predict(X_test)


# In[14]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[15]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[ ]:




