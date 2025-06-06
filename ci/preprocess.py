import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, '..', 'cars_dataset_no_brand 300 rows.csv')

df = pd.read_csv(csv_path)

X = df.drop('SalePrice_USD', axis=1).select_dtypes(include='number')
y = df['SalePrice_USD']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save processed files in ci/ folder
pd.DataFrame(X_scaled, columns=X.columns).to_csv(os.path.join(script_dir, 'X_scaled.csv'), index=False)
pd.DataFrame(y).to_csv(os.path.join(script_dir, 'y.csv'), index=False)
joblib.dump(scaler, os.path.join(script_dir, 'scaler.joblib'))