
# Required Library

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
conf = SparkConf()
conf.setMaster("local").setAppName("My app")
sc = SparkContext.getOrCreate(conf=conf)
spark = SparkSession(sc)
print("Current Spark version is : {0}".format(spark.version))

import pandas as pd
import sklearn
from sklearn import metrics
import numpy as np
from LinearR import linear_algo
from RidgeRegression import ridge_regression
from DecisionTree import decision_tree
from RandomForest import random_forest
from pyspark.ml.classification import RandomForestClassifier



train = pd.read_csv("train.csv", na_values={"Item_Visibility": [0]})

test = pd.read_csv("test.csv", na_values={"Item_Visibility": [0]})

train['source'] = 'train'

test['source'] = 'test'

data = pd.concat([train, test], ignore_index=True)

# Our main goal is item_outlet_Sales

train.head()


discpt = data.describe()

# The number of zero'es values

nan_descript = data.apply(lambda x: sum(x.isnull()))

# Unique values in each columns

uniq = data.apply(lambda x: len(x.unique()))

# grouping in each columns

col = ["Item_Fat_Content", "Item_Type", "Outlet_Location_Type", "Outlet_Size"]

for i in col:
    print("The frequency distribution of each catogorical columns is--" + i + "\n")
    print(data[i].value_counts())

# nan values are replaced in the Item_Weight column with  mean value

data.fillna({"Item_Weight": data["Item_Weight"].mean()}, inplace=True)

# status of  nan values in the dataframe
nan_descript = data.apply(lambda x: sum(x.isnull()))
# No nan values in Item_Weight


data["Outlet_Size"].fillna(method="ffill", inplace=True)

nan_descript = data.apply(lambda x: sum(x.isnull()))

#  item_visibility


visibilty_avg = data.pivot_table(values="Item_Visibility", index="Item_Identifier")

itm_visi = data.groupby('Item_Type')

data_frames = []
for item, item_df in itm_visi:
    data_frames.append(itm_visi.get_group(item))
for i in data_frames:
    i["Item_Visibility"].fillna(value=i["Item_Visibility"].mean(), inplace=True)
    i["Item_Outlet_Sales"].fillna(value=i["Item_Outlet_Sales"].mean(), inplace=True)

new_data = pd.concat(data_frames)

nan_descript = new_data.apply(lambda x: sum(x.isnull()))

# Cleaning of dataset is completed.
new_data["Item_Fat_Content"].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'}, inplace=True)

new_data["Item_Fat_Content"].value_counts()

# one-hot-Coding method for getting the categorical variables
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data = new_data
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Item_Type', 'Outlet_Type']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])
# One Hot Coding:
data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type',
                                     'Item_Type'])

# Exporting the datas

train = data.loc[data['source'] == "train"]

test = data.loc[data['source'] == "test"]

# Drop unnecessary columns:
test.drop(['Item_Outlet_Sales', 'source'], axis=1, inplace=True)
# here we are droping the "Item_Outlet_Sales because this only we want to be predicted from the model that we are going to built
train.drop(['source'], axis=1, inplace=True)

# Export files as modified versions:
train.to_csv("train_modified.csv", index=False)
test.to_csv("test_modified.csv", index=False)

# baseline model as  it is non -predicting model and also commenly known as informed guess

# Mean based:
mean_sales = train['Item_Outlet_Sales'].mean()

# Define a dataframe with IDs for submission:
base1 = test[['Item_Identifier', 'Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

# Export submission file
base1.to_csv("alg0.csv", index=False)


# Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier', 'Outlet_Identifier']

'''Machine Learning'''


def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    # Perform cross-validation:
    cv_score = sklearn.model_selection.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20,
                                                       scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))

    # Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (
    np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])

    # Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)


# Linear Regression
linear_algo(train, test, target, IDcol)
#
# # Ridge Regression
ridge_regression(train, test, target, IDcol)
#
# # decision Tree
decision_tree(train, test, target, IDcol)

# Random Forest
random_forest(train, test, target, IDcol)