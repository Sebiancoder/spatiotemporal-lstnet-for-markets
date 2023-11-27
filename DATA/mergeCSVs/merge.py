import pandas as pd

# Load each CSV file into DataFrames
df1 = pd.read_csv('DATA/mergeCSVs/RapeseedFuturesHistoricalData.csv')
df2 = pd.read_csv('DATA/mergeCSVs/USCornFuturesHistoricalData.csv')
df3 = pd.read_csv('DATA/mergeCSVs/USSoybeansFuturesHistoricalData.csv')
df4 = pd.read_csv('DATA/mergeCSVs/USWheatFuturesHistoricalData.csv')
df5 = pd.read_csv('DATA/mergeCSVs/CastorseedFuturesHistoricalData.csv')
df6 = pd.read_csv('DATA/mergeCSVs/ADMHistoricalData.csv')
df7 = pd.read_csv('DATA/mergeCSVs/BGHistoricalData.csv')
df8 = pd.read_csv('DATA/mergeCSVs/DDHistoricalData.csv')
df9 = pd.read_csv('DATA/mergeCSVs/gsgetf.csv')


# Merge the DataFrames on the "Data" column
# merged_df = df1.merge(df2, on='Date', how='inner')
df1 = df1.drop(['Open', 'High', 'Low', 'Vol.', 'Change %'], axis=1)
merged_df = pd.merge(df1,df2[['Date','Price']],on='Date', how='left')
merged_df = pd.merge(merged_df,df3[['Date','Price']],on='Date', how='left')
merged_df = merged_df.rename(columns={"Price_x": "Rapeseed", "Price_y": "Corn", "Price": "Soybeans"})
merged_df = pd.merge(merged_df,df4[['Date','Price']],on='Date', how='left')
merged_df = pd.merge(merged_df,df5[['Date','Price']],on='Date', how='left')
merged_df = merged_df.rename(columns={"Price_x": "Wheat", "Price_y": "Castorseed"})
merged_df = merged_df.dropna()
merged_df = pd.merge(merged_df,df6[['Date','Price']],on='Date', how='left')
merged_df = pd.merge(merged_df, df7[['Date','Price']],on='Date', how='left')
merged_df = merged_df.rename(columns={"Price_x": "ADM", "Price_y": "BG"})
merged_df = pd.merge(merged_df, df8[['Date','Price']],on='Date', how='left')
merged_df = pd.merge(merged_df, df9[['Date', 'Price']], on='Date', how='left')
merged_df = merged_df.rename(columns={"Price_x": "DD", "Price_y": "GSG"})



merged_df = merged_df.replace(',','', regex=True)
merged_df.astype({col: float for col in merged_df.columns[1:]})


# merged_df = merged_df.merge(df3, on='Date', how='inner')
# merged_df = merged_df.merge(df4, on='Date', how='inner')
# merged_df = merged_df.merge(df5, on='Date', how='inner')

# Select only the "Price" columns
# price_columns = ['Price_1', 'Price_2', 'Price_3', 'Price_4', 'Price']
# merged_df = merged_df[price_columns]

# Rename the columns to have more descriptive names if needed
# price_columns = ['Price_x', 'Price_y', 'Price_x_x', 'Price_y_x', 'Price']

# The merged DataFrame now contains the "Price" data from all five CSV files, joined on the "Data" column.
merged_df.to_csv('DATA/mergeCSVs/output.csv', index=False)