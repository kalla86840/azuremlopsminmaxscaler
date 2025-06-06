
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

X_train = pd.read_csv('X_train_normalized.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'model.joblib')
