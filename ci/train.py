import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Load dataset from root
script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, '..', 'cars_dataset_no_brand 300 rows.csv')
df = pd.read_csv(csv_path)

# Feature-target split
X = df.drop('SalePrice_USD', axis=1).select_dtypes(include='number')
y = df['SalePrice_USD']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Scale
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save model and scaler
model = LinearRegression()
model.fit(X_train_scaled, y_train)

joblib.dump(model, os.path.join(script_dir, 'model.joblib'))
joblib.dump(scaler, os.path.join(script_dir, 'scaler.joblib'))