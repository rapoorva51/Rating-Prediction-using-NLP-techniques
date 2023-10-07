#!/usr/bin/env python
# coding: utf-8

# # Rating Prediction using NLP techniques. 

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


df = pd.read_csv('IMDB Dataset.csv')


# In[4]:


df


# In[5]:


df.isnull().sum()


# In[6]:


df['sentiment'].value_counts()


# In[7]:


import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string


# cleaning/EDA was performed

# In[8]:


def custom_preprocessor(text):
    '''
    Make text lowercase, remove text in square brackets,remove links,remove special characters
    and remove words containing numbers.
    '''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) # remove special chars
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    return text


# In[9]:


df['review']=df['review'].apply(custom_preprocessor)


# In[10]:


df


# In[11]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[12]:


df['sentiment']=le.fit_transform(df['sentiment'])


# In[13]:


df


# Independent(Sentiment) and dependent feature(review)

# In[14]:


y = df['sentiment']
X = df['review']


# Train and test the data

# In[15]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state = 42,test_size = 0.2)


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(ngram_range=(1,2))


# In[17]:


X_train_trans=cv.fit_transform(X_train)


# In[18]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train_trans,y_train)
pred_y=lr.predict(cv.transform(X_test))


# In[19]:


from sklearn.metrics import accuracy_score
score_1 = accuracy_score(y_test,pred_y)
score_1


# Data Visualization

# In[20]:


import seaborn as sns
sns.pairplot(df)


# In[21]:


sns.set(style = "darkgrid" , font_scale = 1.2)
sns.countplot(df.sentiment)


# In[22]:


df['sentiment'].value_counts().plot(kind='pie',autopct='%.1f')


# The Word Cloud

# In[23]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
plt.figure(figsize = (20,20)) # Positive Review Text
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.sentiment == 1].review))
plt.imshow(wc , interpolation = 'bilinear')


# In[24]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
word=df[df['sentiment']==1]['review'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red')
ax1.set_title('Text with Good Reviews')
word=df[df['sentiment']==0]['review'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='green')
ax2.set_title('Text with Bad Reviews')
fig.suptitle('Average word length in each text')


# The Confusion Matrix and Heatmap

# In[25]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred_y)
cm


# In[26]:


plt.figure(figsize = (10,10))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['Bad Reviews','Good Reviews'] , yticklabels = ['Bad Reviews','Good Reviews'])
plt.xlabel("Predicted")
plt.ylabel("Actual")

