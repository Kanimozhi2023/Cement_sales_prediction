#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import warnings
import pandas as pd
import mysql.connector as sql


# In[3]:


connection = sql.connect(user='root',host='localhost',passwd='Dakshith@2021',auth_plugin='mysql_native_password',
                         database='project')
mycursor = connection.cursor()


# In[4]:


data = pd.read_sql("SELECT * FROM project.sales_dataset",con=connection)
print(data)


# In[5]:


data.shape


# In[30]:


#AUTO EDA
import dtale
import plotly.express as px
d = dtale.show(data)
d.open_browser()


# In[6]:


#Manual EDA
data.isnull().sum()


# In[7]:


data.describe()


# In[14]:


#lineplot for netsales and demand
sns.lineplot(data=data,x=data.month,y=data.sales)


# In[10]:


plt.boxplot(data.sales)


# In[11]:


#Visualization of external features
sub = data[["demand","priceofproduct","limestonerequirement","noofhousesbuilt","gdpofindiaincrores"]]
sub.plot(subplots=True)


# In[12]:


#Checking correlation of external variables
sns.pairplot(data)


# In[13]:


#Heat map
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.show()


# In[15]:


#creating pipelines
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# In[16]:


#pipeline for preprocessing
X=data.iloc[:,1:]
y=data['sales']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.2)


# In[17]:


## Pieplining
numeric_features = X_train.select_dtypes(include=['int64','float64']).columns
numeric_preprocessor = Pipeline(
    steps=[
        ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("scaler", StandardScaler())
        ,
    ]
)


# In[18]:


from sklearn import set_config
set_config(display='diagram')
numeric_preprocessor


# In[19]:


categorical_features = X_train.select_dtypes(exclude=['int64','float64']).columns
categorical_preprocessor = Pipeline(
    steps=[
        (
            "imputation_constant",
            SimpleImputer(fill_value="missing", strategy="constant"),
        ),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)


# In[20]:


preprocessor = ColumnTransformer(
             [
                 ("categorial", categorical_preprocessor,categorical_features),
                 ("numerical",numeric_preprocessor,numeric_features)
                 
             ])


# In[21]:


preprocessor


# In[22]:


pipe=Pipeline(
    [("preprocessor",preprocessor),("regressor",RandomForestRegressor())]
)


# In[23]:


from sklearn import set_config
set_config(display='diagram')
pipe


# In[24]:


pipe.fit(X_train,y_train)


# In[25]:


pred_rf = pipe.predict(X_test.dropna())


# In[76]:


sns.lineplot(data=pred_rf,label="pred_rf")
sns.lineplot(data=X_test['sales'],label="original_sales")


# In[26]:


#Mean squared errors
from sklearn.metrics import mean_squared_error
import math
mse = np.sqrt(mean_squared_error(X_test["sales"],pred_rf))
print("mean square error:",mse)
rmse = math.sqrt(mse)
print("RMSE score:",rmse)


# In[32]:


#adfuller test as a function
from statsmodels.tsa.stattools import adfuller
test_dataset = data[['sales','demand','priceofproduct','limestonerequirement','noofhousesbuilt','gdpofindiaincrores']]
def adfuller_test(series, sig=0.05, name=''):
    res = adfuller(series, autolag='AIC')    
    p_value = round(res[1], 3) 

    if p_value <= sig:
        print(f" {name} : P-Value = {p_value} => Stationary. ")
    else:
        print(f" {name} : P-Value = {p_value} => Non-stationary.")


# In[33]:


#convert the data into stationary using LOG diff()
data_differenced_log = np.log(test_dataset)
data_differenced = data_differenced_log-data_differenced_log.shift(1)
data_sta = data_differenced.dropna()
for name, column in data_sta.iteritems():
    adfuller_test(column, name=column.name)


# In[34]:


data_differenced.isnull().sum()


# In[35]:


data_differenced2 = data_differenced.diff().dropna()
data_differenced2


# # Exponential Smoothing

# In[36]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[37]:


#set index
df1 = data.set_index("month")


# In[38]:


train1 = df1.iloc[:115]
test1 = df1.iloc[115:]
train1


# In[77]:


Model = ExponentialSmoothing(train1['sales'],
                                 trend    ='add',
                          initialization_method='heuristic',
                          seasonal = "mul", 
                          seasonal_periods=28, 
                          damped_trend=True).fit()


# In[78]:


train_pred =  Model.fittedvalues


# In[79]:


y_pred = Model.predict(start=1,end=115)


# In[80]:


sns.lineplot(data=y_pred,label="sales_pred")
sns.lineplot(data=train1['sales'],label="original_sales")


# In[81]:


#Mean squared errors
from sklearn.metrics import mean_squared_error
import math
mse = np.sqrt(mean_squared_error(train1['sales'],y_pred.dropna()))
print("mean square error:",mse)
rmse = math.sqrt(mse)
print("RMSE score:",rmse)


# # SARIMA MODEL

# In[45]:


import statsmodels.api as sm


# In[47]:


#ACF AND PCF PLOTS
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data_differenced2['sales'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data_differenced2['sales'].iloc[13:],lags=40,ax=ax2)


# In[48]:


sarima=sm.tsa.statespace.SARIMAX(train1['sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=sarima.fit()


# In[49]:


pred_sarima=results.predict(start=1,end=115,dynamic=True)


# In[71]:


sns.lineplot(data=pred_sarima,label="sarima_pred")
sns.lineplot(data=train1['sales'],label='original_sales')


# In[51]:


#Mean squared errors
from sklearn.metrics import mean_squared_error
import math
mse = np.sqrt(mean_squared_error(train1['sales'],pred_sarima.dropna()))
print("mean square error:",mse)
rmse = math.sqrt(mse)
print("RMSE score:",rmse)


# # ARIMA MODEL

# In[52]:


model=sm.tsa.arima.ARIMA(train1['sales'],order=(1,1,1))
model_fit=model.fit()
model_fit.summary()


# In[53]:


pred_arima=results.predict(start=1,end=115,dynamic=True)


# In[72]:


sns.lineplot(data=pred_arima,label="arima_pred")
sns.lineplot(data=train1['sales'],label="original_sales")


# In[55]:


#Mean squared errors
from sklearn.metrics import mean_squared_error
import math
mse = np.sqrt(mean_squared_error(train1['sales'],pred_arima.dropna()))
print("mean square error:",mse)
rmse = math.sqrt(mse)
print("RMSE score:",rmse)


# # PROPHET MODEL

# In[60]:


trainp = data.iloc[:115]
trainp.rename(columns={'sales':'y','month':'ds'},inplace=True)
testp = data.iloc[115:]
testp.rename(columns={'sales':'y','month':'ds'},inplace=True)


# In[61]:


from prophet import Prophet
model1=Prophet(interval_width=0.9) 
model1.fit(trainp)


# In[63]:


forecast1 = model1.predict(trainp)
forecast1=forecast1[['ds','yhat']]
forecast1.tail()


# In[64]:


final_df=pd.concat((forecast1['yhat'],trainp),axis=1)
final_df


# In[66]:


#Mean squared errors
from sklearn.metrics import mean_squared_error
import math
mse = np.sqrt(mean_squared_error(final_df[['y']],final_df[['yhat']].dropna()))
print("mean squared errors:",mse)
rmse = math.sqrt(mse)
print("RMSE scores:",rmse)


# In[73]:


sns.lineplot(data=final_df['y'],label="original_sales")
sns.lineplot(data=final_df['yhat'],label="prophet_pred")


# In[ ]:




