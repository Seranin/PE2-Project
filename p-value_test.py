import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Load the dataset
data = pd.read_csv("household_power_consumption.txt", delimiter=";")

# Replace '?' with NaN
data.replace('?', float("NaN"), inplace=True)

# Drop rows with missing values
data.dropna(inplace=True)

# Assuming 'Global_active_power' is the column you want to test
time_series = data['Global_active_power'][:10000]

# Run the ADF test
result = adfuller(time_series)

# Print the results
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
