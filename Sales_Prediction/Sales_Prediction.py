import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Import Dataset
train = pd.read_csv('train.csv')
store = pd.read_csv('store.csv')

# Preprocessing
month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}


def check_bogo(row):
    if pd.isna(row['ContinuousBogoMonths']):
        return 0
    months = row['ContinuousBogoMonths'].split(',')
    for i in months:
        if month_map[i] == row['Month'] and row['BOGO'] == 1:
            return 1
    return 0


def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))


df = pd.merge(train, store, on='Store_id', how='left')

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['WeekOfYear'] = (df['Date'].dt.day_of_year // 7) + 1
df['HasRival'] = np.where((df['Year'] >= df['RivalEntryYear']) & (df['Month'] >= df['RivalOpeningMonth']), 1, 0)
df['ContinuousBogoinMonths'] = df.apply(check_bogo, axis=1)
df['ContinuousBogoinYear'] = np.where((df['ContinuousBogoSinceYear'] <= df['Year']), 1, 0)
df['DistanceToRivalStore'] = df['DistanceToRivalStore'].fillna(np.median(df['DistanceToRivalStore']))
df = df.fillna(0)
df = df.drop('ContinuousBogoSinceYear', axis=1)
df = df.drop('ContinuousBogoMonths', axis=1)
df = df.drop('RivalEntryYear', axis=1)
df = df.drop('RivalOpeningMonth', axis=1)
df = df.drop('NumberOfCustomers', axis=1)
df = df.drop('ContinuousBogo', axis=1)
df = df.drop('DayOfWeek', axis=1)
df = df.drop('Date', axis=1)

label_encoder = LabelEncoder()
df['Stock variety'] = label_encoder.fit_transform(df['Stock variety'])
label_encoder = LabelEncoder()
df['RetailType'] = label_encoder.fit_transform(df['RetailType'])

df = df[df["Is_Open"] != 0]
df[df['Sales'] == 0] = 1

X = df[['Store_id', 'BOGO', 'Holiday', 'RetailType',
        'Stock variety', 'DistanceToRivalStore', 'ContinuousBogoSinceWeek',
        'Year', 'Month', 'Day', 'WeekOfYear', 'HasRival',
        'ContinuousBogoinMonths', 'ContinuousBogoinYear']]

Y = df['Sales']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# LinearRegression Train and Test
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
y_pred_lr = model_LR.predict(X_test)

rmspe_value = rmspe(Y_test, y_pred_lr)
linear_reg_rmse = np.sqrt(mean_squared_error(Y_test, y_pred_lr))
print('the RMSE for linear regression is: ', linear_reg_rmse)
print('the RMSPE for linear regression is:', rmspe_value)
print(f"Linear regression R2 Score: {model_LR.score(X_test, Y_test)}")

# RandomForest Train and Test
model_RF = RandomForestRegressor(n_estimators=10, random_state=42)
model_RF.fit(X_train, Y_train)
y_pred_rf = model_RF.predict(X_test)

rmspe_value = rmspe(Y_test, y_pred_rf)
random_fo_rmse = np.sqrt(mean_squared_error(Y_test, y_pred_rf))
print('the RMSE for Random Forest is: ', random_fo_rmse)
print('the RMSPE for linear regression is:', rmspe_value)
print(f"Random Forest R2 Score: {model_RF.score(X_test, Y_test)}")

# Feature importance
importances = model_RF.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

print("Feature Importance:")
for i in range(len(feature_names)):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

plt.barh(range(len(importances)), importances[indices], align="center")
plt.yticks(range(len(importances)), feature_names[indices])
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
