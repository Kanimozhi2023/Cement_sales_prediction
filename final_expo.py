import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import mysql.connector as sql
connection = sql.connect(user='root',host='localhost',passwd='Dakshith@2021',auth_plugin='mysql_native_password',
                         database='project')
mycursor = connection.cursor()


data = pd.read_sql("SELECT * FROM project.sales",con=connection)
print(data)

train = data.iloc[:75]
test = data.iloc[75:]
train

## Pieplining
numeric_features = train.select_dtypes(include=['int64','float64']).columns
numeric_preprocessor = Pipeline(
    steps=[
        ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("scaler", StandardScaler())
        ,
    ]
)


categorical_features = train.select_dtypes(exclude=['int64','float64']).columns
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

pipe=Pipeline(
    [("preprocessor",preprocessor),('regressor',)]
)


Model = ExponentialSmoothing(train['sales'],
                                 trend    ='add',
                          initialization_method='heuristic',
                          seasonal = "add", 
                          seasonal_periods=28, 
                          damped_trend=True).fit()

train_pred =  Model.fittedvalues

y_pred = Model.predict(30)
sns.lineplot(data=y_pred,label="sales_pred")
sns.lineplot(data=train['sales'],label="original_sales")


import pickle
pickle.dump(Model, open('model.pkl', 'wb'))


