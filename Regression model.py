#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')


# In[2]:


df


# data preprocessing 

# In[3]:


y = df['logS']


# In[4]:


y


# In[8]:


x = df.drop('logS', axis=1)
x


# DATA SPLITTING

# In[9]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)


# In[10]:


x_train


# In[11]:


x_test


# In[12]:


y_train


# In[13]:


y_test


# model building
# 

# linear regression
# 

# In[15]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)


# In[16]:


y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)


# In[17]:


y


# In[18]:


y_lr_train_pred


# In[19]:


y_lr_test_pred


# Evaluating Model Performance

# In[22]:


from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)


# In[23]:


lr_train_mse
lr_train_r2


# In[24]:


lr_train_mse


# In[25]:


lr_train_r2


# lr_test_mse

# In[27]:


lr_test_r2 


# In[28]:


print('LR MSE Train',lr_train_mse)
print('LR R2 Train',lr_train_r2)
print('LR MSE Test',lr_test_mse)
print('LR R2 Test',lr_test_r2)


# In[29]:


lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()


# In[42]:


lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
lr_results


# RANDOM FOREST REGRESSOR MODEL

# In[33]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)


# In[34]:


y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)


# In[35]:


from sklearn.metrics import mean_squared_error, r2_score
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)


# In[36]:


print('RF MSE Train',rf_train_mse)
print('RF R2 Train',rf_train_r2)
print('RF MSE Test',rf_test_mse)
print('RF R2 Test',rf_test_r2)


# In[43]:


rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()


# In[44]:


rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
rf_results


# MODEL COMPARISON

# In[45]:


df_models = pd.concat([lr_results, rf_results], axis=0)
df_models


# In[46]:


df_models.reset_index(drop=True)


# DATA VISUALIZATION
# 

# In[61]:


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.3)

z=np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train))
plt.xlabel("experimental LogS")
plt.ylabel("predict logS")


# In[ ]:




