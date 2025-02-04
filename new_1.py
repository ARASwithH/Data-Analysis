import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

train = pd.read_csv('train.csv')
store = pd.read_csv('store.csv')

month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

merged_df = pd.merge(train, store, on='Store_id' ,how='left')

merged_df['Date'] = pd.to_datetime(merged_df['Date'])

merged_df['Year'] = merged_df['Date'].dt.year
merged_df['Month'] = merged_df['Date'].dt.month
merged_df['Day'] = merged_df['Date'].dt.day
merged_df['WeekOfYear'] = (merged_df['Date'].dt.day_of_year // 7) + 1

merged_df['HasRival'] = np.where(
    (merged_df['Year'] >= merged_df['RivalEntryYear']) &
    (merged_df['Month'] >= merged_df['RivalOpeningMonth']), 
    1, 
    0
)

merged_df['TotalSaleOfWeek'] = merged_df.groupby(['Store_id' , 'WeekOfYear', 'Year'])['Sales'].transform('sum')

# merged_df['TotalSaleOfMonth'] = merged_df.groupby(['Store_id' , 'Month', 'Year'])['Sales'].transform('sum')
# merged_df['ContinuousBogoMonths'] = merged_df['ContinuousBogoMonths'].apply(lambda x: [month_map[month.strip()] for month in x.split(',')])
merged_df = merged_df.drop('RivalEntryYear', axis=1)
merged_df = merged_df.drop('RivalOpeningMonth', axis=1)
merged_df = merged_df.drop('NumberOfCustomers', axis=1)
merged_df = merged_df.drop('Date', axis=1)
print(merged_df)

