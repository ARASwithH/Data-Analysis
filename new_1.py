import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


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

df = pd.merge(train, store, on='Store_id' ,how='left')

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['WeekOfYear'] = (df['Date'].dt.day_of_year // 7) + 1
df['HasRival'] = np.where((df['Year'] >= df['RivalEntryYear']) & (df['Month'] >= df['RivalOpeningMonth']), 1, 0)
df['TotalSaleOfWeek'] = df.groupby(['Store_id' , 'WeekOfYear', 'Year'])['Sales'].transform('sum')
df['ContinuousBogoinMonths'] = df.apply(check_bogo, axis=1)
df['ContinuousBogoinYear'] = np.where((df['ContinuousBogoSinceYear'] <= df['Year']), 1, 0)
df['DistanceToRivalStore'] = df['DistanceToRivalStore'].fillna(np.mean(df['DistanceToRivalStore']))
df = df.fillna(0)
df = df.drop('ContinuousBogoSinceYear', axis=1)
df = df.drop('ContinuousBogoMonths', axis=1)
df = df.drop('RivalEntryYear', axis=1)
df = df.drop('RivalOpeningMonth', axis=1)
df = df.drop('NumberOfCustomers', axis=1)
df = df.drop('ContinuousBogo', axis=1)
df = df.drop('DayOfWeek', axis=1)
# df = df.drop('Year', axis=1)
df = df.drop('Date', axis=1)

print(df.columns)


<<<<<<< HEAD

merged_df = merged_df.drop('NumberOfCustomers', axis=1)
merged_df = merged_df.drop('Date', axis=1)
print(merged_df)
=======
>>>>>>> 3689d58e4e578b2c6f50cc02256da1c0256b1975

