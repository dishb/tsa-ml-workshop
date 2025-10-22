import pandas as p
import numpy as n
import matplotlib.pyplot as pl
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

#Actual Codey part of the Code
link = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = p.read_csv(link)
data.head()
data.info()
data.describe()

#Plotty Stuff (Doesn't work in Code Spaces)
#sn.pairplot(data[['rm','lstat','ptratio','medv']])
#pl.show()
#pl.figure(figsize=(8,8))
#sn.heatmap(data.corr(),annot=True,cmap='coolwarm')
#pl.show()

#Actual Machine Learning Part
X = data[['rm','lstat','ptratio']]
y = data['medv']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=32
)
#random_state is the seed basically
#test_size can be inbetween 0 and 1 and is just the percentage of how much is used for testing the model vs training the model

#Linear Regression Model
#model = LinearRegression()
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)

#Tree Model Thingy
#model = RandomForestRegressor(n_estimators=10, random_state=69)
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#print("Error Amount:", mean_absolute_error(y_test, y_pred))
#print("Rating:", r2_score(y_test, y_pred))

#Gradient Boosting
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=67
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Error Amount:", mean_absolute_error(y_test, y_pred))
print("Rating:", r2_score(y_test, y_pred))

#Accuracy Test Code Stuff for Linear
#pl.scatter(y_test, y_pred)
#pl.xlabel("Actual")
#pl.ylabel("Predicted")
#pl.title("Actual vs Predicted")
#pl.show()
#print("Error Amount:", mean_absolute_error(y_test, y_pred))
#print("Rating:", r2_score(y_test, y_pred))
#coefficients = p.DataFrame({
#    'Feature': X.columns,
#    'Coefficient': model.coef_
#})
#print(coefficients)