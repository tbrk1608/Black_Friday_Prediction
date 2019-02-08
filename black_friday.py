import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
#%%
data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sub  = pd.read_csv("Sample_Submission_Tm9Lura.csv")
data.info()
data.isnull().sum()

#%%
data['Product_Category_2'].fillna(value = 0, inplace = True)
data['Product_Category_3'].fillna(value = 0, inplace = True)

test['Product_Category_2'].fillna(value = 0.0, inplace = True)
test['Product_Category_3'].fillna(value = 0.0, inplace = True)

#%%
data.Age[data['Age'] == "0-17"]  = 15
data.Age[data['Age'] == "18-25"] = 21 
data.Age[data['Age'] == "26-35"] = 30
data.Age[data['Age'] == "36-45"] = 40
data.Age[data['Age'] == "46-50"] = 48
data.Age[data['Age'] == "51-55"] = 53
data.Age[data['Age'] == "55+"]   = 60
data['Age'] = data['Age'].astype(int)

test.Age[test['Age'] == "0-17"]  = 15
test.Age[test['Age'] == "18-25"] = 21 
test.Age[test['Age'] == "26-35"] = 30
test.Age[test['Age'] == "36-45"] = 40
test.Age[test['Age'] == "46-50"] = 48
test.Age[test['Age'] == "51-55"] = 53
test.Age[test['Age'] == "55+"]   = 60
test['Age'] = test['Age'].astype(int)

data.Stay_In_Current_City_Years[data['Stay_In_Current_City_Years'] == "0"] = 0
data.Stay_In_Current_City_Years[data['Stay_In_Current_City_Years'] == "1"] = 1
data.Stay_In_Current_City_Years[data['Stay_In_Current_City_Years'] == "2"] = 2
data.Stay_In_Current_City_Years[data['Stay_In_Current_City_Years'] == "3"] = 3
data.Stay_In_Current_City_Years[data['Stay_In_Current_City_Years'] == "4+"] = 4
data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].astype(int)

test.Stay_In_Current_City_Years[test['Stay_In_Current_City_Years'] == "0"] = 0
test.Stay_In_Current_City_Years[test['Stay_In_Current_City_Years'] == "1"] = 1
test.Stay_In_Current_City_Years[test['Stay_In_Current_City_Years'] == "2"] = 2
test.Stay_In_Current_City_Years[test['Stay_In_Current_City_Years'] == "3"] = 3
test.Stay_In_Current_City_Years[test['Stay_In_Current_City_Years'] == "4+"] = 4
test['Stay_In_Current_City_Years'] = test['Stay_In_Current_City_Years'].astype(int)

data.Gender[data['Gender'] == "M"] = 1
data.Gender[data['Gender'] == "F"] = 2
data['Gender'] = data['Gender'].astype(int)

test.Gender[test['Gender'] == "M"] = 1
test.Gender[test['Gender'] == "F"] = 2
test['Gender'] = test['Gender'].astype(int)

data.City_Category[data['City_Category'] == "A"] = 0
data.City_Category[data['City_Category'] == "B"] = 1
data.City_Category[data['City_Category'] == "C"] = 2
data['City_Category'] = data['City_Category'].astype(int)

test.City_Category[test['City_Category'] == "A"] = 0
test.City_Category[test['City_Category'] == "B"] = 1
test.City_Category[test['City_Category'] == "C"] = 2
test['City_Category'] = test['City_Category'].astype(int)

#%%
df = data.groupby('User_ID')['Purchase'].mean()
df = df.reset_index()
df['Rate'] = 1
df.Rate[(df['Purchase'] <= 5000)] = 4
df.Rate[(df['Purchase'] > 5000) & (df['Purchase'] <= 10000)] = 3
df.Rate[(df['Purchase'] > 10000) & (df['Purchase'] <= 15000)] = 2
df['Rate'] = df['Rate'].astype(int)
df = df.drop('Purchase',1)

#len(set(data['Product_ID']) - set(test['Product_ID'])) - 186 new products
#len(set(data['User_ID']) - set(test['User_ID']))

data = pd.merge(data, df, on="User_ID")
test = pd.merge(test, df, on="User_ID")


df2 = data.groupby('Product_ID')['Purchase'].mean()
df2 = df2.reset_index()
df2['Product_Rate'] = 1
df2.Product_Rate[(df2['Purchase'] <= 5000)] = 4
df2.Product_Rate[(df2['Purchase'] > 5000) & (df2['Purchase'] <= 10000)] = 3
df2.Product_Rate[(df2['Purchase'] > 10000) & (df2['Purchase'] <= 15000)] = 2
df2['Product_Rate'] = df2['Product_Rate'].astype(int)
df2 = df2.drop('Purchase',1)

data = pd.merge(data, df2, on="Product_ID")
test = pd.merge(test, df2, on="Product_ID",how = "left")

#%%
y = 'Purchase'
features = [c for c in data.columns if c != y]
cat_f = [f for f in features if data.dtypes[f] == object]
num_f = [f for f in features if f not in cat_f]

num_f = data[num_f].values
encoders = {}
encoded_features = []

for f in cat_f:
    encoders[f] = OrdinalEncoder()
    encoded_features.append(encoders[f].fit_transform(data[f].values.reshape(-1, 1)))

cat_f = np.column_stack(encoded_features)
features = np.concatenate([num_f, cat_f], axis=1)

y = data[y]
#%%
plt.figure(figsize=(12,5))
sns.heatmap(data.corr(),annot=True)
plt.show()
#%%
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.4,random_state=213)
model = xgb.XGBRegressor(colsample_bytree = 0.9,learning_rate=0.1,max_depth=16,min_child_weight=40,n_jobs=2,subsample=0.9,random_state=213)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print (np.sqrt(mean_squared_error(y_pred, y_test)))

#%%
tfeatures = [c for c in test.columns]
tcat_f = [f for f in tfeatures if test.dtypes[f] == object]
tnum_f = [f for f in tfeatures if f not in tcat_f]

tnum_f = test[tnum_f].values
tencoders = {}
tencoded_features = []

for f in tcat_f:
    tencoders[f] = OrdinalEncoder()
    tencoded_features.append(tencoders[f].fit_transform(test[f].values.reshape(-1, 1)))

tcat_f = np.column_stack(tencoded_features)
tfeatures = np.concatenate([tnum_f, tcat_f], axis=1)

#%%
test_pred = model.predict(tfeatures)
sub['User_ID'] = test['User_ID']
sub['Product_ID'] = test['Product_ID']
sub['Purchase'] = test_pred
#sub.Purchase[sub['Purchase'] < 0] = 12
sub.to_csv("sub.csv", index = False)