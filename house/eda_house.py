# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# %%
train = pd.read_csv('data_house/train.csv')
test = pd.read_csv('data_house/test.csv')

# %%
train

# %%
test

# %%
train.iloc[:, (train.isnull().sum() != 0).values].isnull().sum()

# %%
train.iloc[1]

# %%
train.loc[2, ['Id', 'Alley']]

# %%
train.iloc[:, (train.isnull().sum() != 0).values].isnull().sum()/train.shape[0]

# %%
test.iloc[:, (test.isnull().sum() != 0).values].isnull().sum()/test.shape[0]

# %%
train = train.dropna(axis=1)
train

# %%
train.isnull().sum().sum()

# %%
train.describe()

# %%
pd.get_dummies(train)

# %%
train.corr()

# %%
train.corr()["SalePrice"][np.abs(train.corr()["SalePrice"]) >= 0.6]

# %%
train = train.select_dtypes(include=np.number)
train

# %%
train.dtypes

# %%
y_train = train['SalePrice'].values
y_train

# %%
train = train.drop(columns = ['Id', 'SalePrice'])
train

# %%
x_train = train.values

# %%
x_train, y_train

# %%
linreg = LinearRegression()
linreg.fit(x_train, y_train)

# %%
np.sqrt(mean_squared_error(y_train, linreg.predict(x_train)))

# %%

# %%

# %%

# %%

# %%

# %%
