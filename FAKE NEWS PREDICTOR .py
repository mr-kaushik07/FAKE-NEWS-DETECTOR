#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


data=pd.read_csv('C:\Users\hp\Desktop\Fake.csv')


# In[5]:


data.head()


# # importing the Dependencies

# In[6]:


import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[12]:


import nltk
nltk.download('stopwords')


# In[13]:


print(stopwords.words('english'))


# # Data Pre-processing

# In[14]:


data.shape


# In[15]:


data.isnull().sum()


# In[16]:


data=data.fillna('')


# In[17]:


data['content']=data['author']+' '+data['title']


# In[18]:


print(data['content'])


# In[19]:


x=data.drop(columns='label',axis=1)
y=data['label']


# In[20]:


print(x)
print(y)


# In[30]:


port_stem=PorterStemmer()


# In[28]:


def stemming(content):
    stemmed_content=re.sub('[^a-zA-Z]',' ', content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content


# In[29]:


data['content']=data['content'].apply(stemming)


# In[24]:


print(data['content'])


# In[25]:


x=data['content'].values
y=data['label'].values


# In[21]:


print(x)


# In[22]:


print(y)


# In[23]:


y.shape


# In[24]:


x.shape


# In[25]:


vector=TfidfVectorizer()
vector.fit(x)

x=vector.transform(x)


# In[26]:


print(x)


# # spiliting the dataset to training and test data

# In[27]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)


# # Training the Model: Logistic Regression 

# In[28]:


model=LogisticRegression()


# In[29]:


model.fit(xtrain, ytrain)


# # Evaluation accuracy score

# In[30]:


xtrain_predection= model.predict(xtrain)
trainingdata_accuracy= accuracy_score(xtrain_predection, ytrain)


# In[31]:


print("Accuracy score of training data:",trainingdata_accuracy)


# In[32]:


xtest_predection= model.predict(xtest)
testdata_accuracy= accuracy_score(xtest_predection, ytest)


# In[33]:


print("Accuracy score for test data:",testdata_accuracy)


# # Making a Predective System

# In[34]:


xnew= xtest[0]

predict= model.predict(xnew)
print(predict)
if (predict[0]==0):
    print("The news is real")
else:
    print("THE news is fake")


# In[1]:


print(ytest[3000])


# In[ ]:




