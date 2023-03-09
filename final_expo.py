#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[71]:


import mysql.connector as sql
connection = sql.connect(user='root',host='localhost',passwd='Dakshith@2021',auth_plugin='mysql_native_password',
                         database='project')
mycursor = connection.cursor()


# In[72]:


data = pd.read_sql("SELECT * FROM project.sales_dataset",con=connection)
print(data)


# In[74]:


train = data.iloc[:115]
test = data.iloc[115:]
train


# In[78]:


Model = ExponentialSmoothing(test['sales'],
                                 trend    ='add',
                          initialization_method='heuristic',
                          seasonal = "mul", 
                          seasonal_periods=12, 
                          damped_trend=True).fit()


# In[79]:


train_pred =  Model.fittedvalues


# In[80]:


y_pred = Model.predict(30)


# In[81]:

# In[82]:


import pickle
pickle.dump(Model, open('model.pkl', 'wb'))

# In[ ]:




