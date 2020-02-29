import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("train.csv")

len(data['date'])

data.head()

data.groupby(['item_nbr']).count()

splitdate = [data['date'][i].split('-') for i in range(len(data['date']))]

# +
year = []
month = []
day = []

for i in range(3370464):
    for j in range(3):
        if j==0:
            year.append(int(splitdate[i][j]))
        elif j==1:
            month.append(int(splitdate[i][j]))
        elif j==2:
            day.append(int(splitdate[i][j]))
           
# -

print(year[:5])
print(month[:5])
print(day[:5])


dates = pd.DataFrame({'year':year,'month':month,'day':day})
dates.head()

newdata = pd.concat([data,dates], axis=1)
newdata = newdata.drop(columns='date')

newdata.head()

label_encoder = LabelEncoder()
newdata['onpromotion']=label_encoder.fit_transform(newdata['onpromotion'])

newdata.head()

newdata.loc[newdata['onpromotion']!=0]




