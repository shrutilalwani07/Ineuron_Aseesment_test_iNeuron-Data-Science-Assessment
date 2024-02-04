#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[5]:


import pandas as pd

df = pd.read_csv("C:\\Users\\shrut\\Downloads\\archive (2)\\instagram_reach.csv")

#df = pd.read_csv(r"C:\Users\shrut\Downloads\archive (2)\instagram_reach.csv")


# In[6]:


df


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[9]:


X = df[['USERNAME', 'Caption', 'Hashtags', 'Followers']]


# In[12]:


Y_likes = df['Likes']
Y_time_since_posted = df['Time since posted']


# In[13]:


X_train, X_test, Y_likes_train, Y_likes_test, Y_time_train, Y_time_test = train_test_split(
    X, Y_likes, Y_time_since_posted, test_size=0.2, random_state=42
)


# In[14]:


model_likes = RandomForestRegressor()
model_time_since_posted = RandomForestRegressor()



# In[ ]:


y_likes_pred = model_likes.predict(X_test)
y_time_pred = model_time_since_posted.predict(X_test)


# In[ ]:


mse_likes = mean_squared_error(y_likes_test, y_likes_pred)
mse_time = mean_squared_error(y_time_test, y_time_pred)


# In[ ]:


print(f'Mean Squared Error (Likes): {mse_likes}')
print(f'Mean Squared Error (Time Since Posted): {mse_time}')

