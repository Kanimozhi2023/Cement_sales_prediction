#import libraries
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import pandas as pd
import mysql.connector as sql

connection = sql.connect(user='root',host='localhost',passwd='Dakshith@2021',auth_plugin='mysql_native_password',
                         database='project')
mycursor = connection.cursor()

data = pd.read_excel(r"C:\Users\DHANRAJ\Downloads\ML model\salesxl.xlsx")
print(data)

data.shape
#Auto EDA
import dtale
d = dtale.show(data)
d.open_browser()

#Manual EDA
data.isnull().sum()

data.describe()

#lineplot for netsales and demand
sns.lineplot(data=data,x=data.month,y=data.sales)

plt.boxplot(data.sales)

#Visualization of external features
sub = data[["demand","priceofproduct","limestonerequirement","noofhousesbuilt","gdpofindiaincrores"]]
sub.plot(subplots=True)

#Checking correlation of external variables
sns.pairplot(data)

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

#creating pipelines
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#pipeline for preprocessing
X=data.iloc[:,1:]
y=data['sales']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.2)


## Pieplining
numeric_features = X_train.select_dtypes(include=['int64','float64']).columns
numeric_preprocessor = Pipeline(
    steps=[
        ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("scaler", StandardScaler())
        ,
    ]
)

from sklearn import set_config
set_config(display='diagram')
numeric_preprocessor

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


preprocessor = ColumnTransformer(
             [
                 ("categorial", categorical_preprocessor,categorical_features),
                 ("numerical",numeric_preprocessor,numeric_features)
                 
             ])

preprocessor

pipe=Pipeline(
    [("preprocessor",preprocessor),("regressor",RandomForestRegressor())]
)

from sklearn import set_config
set_config(display='diagram')
pipe


pipe.fit(X_train,y_train)


pred_rf = pipe.predict(X_test.dropna())

sns.lineplot(data=pred_rf,label="pred_rf")
sns.lineplot(data=X_test['sales'],label="original_sales")

#Mean squared errors
from sklearn.metrics import mean_squared_error
import math
mse = np.sqrt(mean_squared_error(X_test["sales"],pred_rf))
print("mean square error:",mse)
rmse = math.sqrt(mse)
print("RMSE score:",rmse)

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

#convert the data into stationary using LOG diff()
data_differenced_log = np.log(test_dataset)
data_differenced = data_differenced_log-data_differenced_log.diff()
data_sta = data_differenced.dropna()
data_differenced.isnull().sum()
data_differenced2 = data_differenced.diff().dropna()
data_differenced2
for name, column in data_differenced2.iteritems():
    adfuller_test(column, name=column.name)

#EXPONENTIAL SMOOTHING
   
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#set index
df1 = data.set_index("month")


train1 = df1.iloc[:75]
test1 = df1.iloc[75:]
train1

Model = ExponentialSmoothing(train1['sales'],
                                 trend    ='add',
                          initialization_method='heuristic',
                          seasonal = "mul", 
                          seasonal_periods=28, 
                          damped_trend=True).fit()

train_pred =  Model.fittedvalues

y_pred = Model.predict(start=1,end=75)
sns.lineplot(data=y_pred,label="sales_pred")
sns.lineplot(data=train1['sales'],label="original_sales")


#Mean squared errors
mse = np.sqrt(mean_squared_error(train1['sales'],y_pred))
print("mean square error:",mse)
rmse = math.sqrt(mse)
print("RMSE score:",rmse)


# # SARIMA MODEL

import statsmodels.api as sm

#ACF AND PCF PLOTS
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data_differenced2['sales'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data_differenced2['sales'].iloc[13:],lags=30,ax=ax2)

sarima=sm.tsa.statespace.SARIMAX(train1['sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=sarima.fit()

pred_sarima=results.predict(start=1,end=75,dynamic=True)


sns.lineplot(data=pred_sarima,label="sarima_pred")
sns.lineplot(data=train1['sales'],label='original_sales')


#Mean squared errors
mse = np.sqrt(mean_squared_error(train1['sales'],pred_sarima))
print("mean square error:",mse)
rmse = math.sqrt(mse)
print("RMSE score:",rmse)


# # ARIMA MODEL


model=sm.tsa.arima.ARIMA(train1['sales'],order=(1,1,1))
model_fit=model.fit()
model_fit.summary()


pred_arima=results.predict(start=1,end=75,dynamic=True)


sns.lineplot(data=pred_arima,label="arima_pred")
sns.lineplot(data=train1['sales'],label="original_sales")

#Mean squared errors
mse = np.sqrt(mean_squared_error(train1['sales'],pred_arima.dropna()))
print("mean square error:",mse)
rmse = math.sqrt(mse)
print("RMSE score:",rmse)

# # PROPHET MODEL

trainp = data.iloc[:75]
trainp.rename(columns={'sales':'y','month':'ds'},inplace=True)
testp = data.iloc[75:]
testp.rename(columns={'sales':'y','month':'ds'},inplace=True)


from prophet import Prophet
model1=Prophet(interval_width=0.9) 
model1.fit(trainp)

forecast1 = model1.predict(trainp)
forecast1=forecast1[['ds','yhat']]
forecast1.tail()


final_df=pd.concat((forecast1['yhat'],trainp),axis=1)
final_df


#Mean squared errors
mse = np.sqrt(mean_squared_error(final_df[['y']],final_df[['yhat']].dropna()))
print("mean squared errors:",mse)
rmse = math.sqrt(mse)
print("RMSE scores:",rmse)


sns.lineplot(data=final_df['y'],label="original_sales")
sns.lineplot(data=final_df['yhat'],label="prophet_pred")


