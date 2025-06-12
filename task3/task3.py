import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('train.csv')

print(df.head())

features = ['OverallQual', 'GrLivArea']
target = 'SalePrice'

df = df[features + [target]].dropna()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

new_house = {
    'OverallQual': 7,
    'GrLivArea': 2000
}

new_house_df = pd.DataFrame([new_house])
predicted_price = model.predict(new_house_df)
print(f'Predicted House Price: ${predicted_price[0]:.2f}')

