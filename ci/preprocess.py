
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

script_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(script_dir, '..', 'cars_dataset_no_brand 300 rows.csv'))

X = df.drop('SalePrice_USD', axis=1).select_dtypes(include=['number'])
y = df['SalePrice_USD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

joblib.dump(scaler, 'scaler.joblib')
joblib.dump(X_test, 'X_test.joblib')
joblib.dump(y_test, 'y_test.joblib')
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('X_train_scaled.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
