import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and Merge 

train_set = pd.read_csv('train.csv')
store_set = pd.read_csv('store.csv')

total_sales = train_set.groupby('Store_id')['Sales'].sum().reset_index()

store_set = store_set.merge(total_sales, on='Store_id', how='left')

store_set.rename(columns={'Sales': 'Total Sales'}, inplace=True)

print(store_set)



# # Preprocessing

# mean_val = df['DistanceToRivalStore'].mean()
# df['DistanceToRivalStore'].fillna(mean_val)
# df = df.fillna(0)

# df['Date'] = pd.to_datetime(df['Date'])
# df['Year'] = df['Date'].dt.year
# df['Month'] = df['Date'].dt.month
# df['Day'] = df['Date'].dt.day
# df['WeekOfYear'] = df['Date'].dt.day_of_week % 7 + 1
# df.drop('Date', axis=1)

# df.drop('NumberOfCustomers', axis=1)






# # Dividing Train and Test

# df = df.sort_values(by='Date')
# spl = int(len(df) * 0.7)
# train = df[:spl]
# test = df[spl:]



# # scaler = StandardScaler()
# # scaler.fit(train)