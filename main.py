import pandas as p
import numpy as n
import matplotlib.pyplot as pl
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

#Actual Codey part of the Code
link = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = p.read_csv(link)
data.head()
data.info()
data.describe()
sn.pairplot(data[['rm','lstat','ptratio','medv']])
pl.show()
pl.figure(figsize=(8,8))
sn.heatmap(data.corr(),annot=True,cmap='coolwarm')
pl.show()

#Actual Machine Learning Part
X = data[['rm','lstat','ptratio']]
y = data['medv']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Accuracy Test Code Stuff
pl.scatter(y_test, y_pred)
pl.xlabel("Actual")
pl.ylabel("Predicted")
pl.title("Actual vs Predicted")
pl.show()
print("Error Amount:", mean_absolute_error(y_test, y_pred))
print("Rating:", r2_score(y_test, y_pred))
coefficients = p.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print(coefficients)