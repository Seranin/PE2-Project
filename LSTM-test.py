import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#Hyper-Parameters
unit_count = 25
time_steps = 5
batch_size_count = 32
epoch_number = 100

model_title = f"LSTM_Time_steps_{time_steps}_Epoch_{epoch_number}_Batch size_{batch_size_count}_Unit count_{unit_count}"


data = pd.read_csv('Modified_Dataset_Null_as_zero.csv', index_col='Datetime', parse_dates=True)

#Datascaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

# Function for time_steps
def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)



# Splitting the dataset
X, y = create_dataset(scaled_data, time_steps)

# Reshape the data for LSTM [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split the data
split = int(0.8 * len(data))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Build the model
model = Sequential()
model.add(LSTM(units=unit_count, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=unit_count))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=epoch_number, batch_size=batch_size_count)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
print("Train Loss:", train_loss)

test_loss = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", test_loss)

model.save( "./Models/" + model_title + ".keras")

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)


train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# Plot the results
plt.figure(figsize=(12,8))
plt.title(model_title)
plt.figtext(0.5, 0.15, f"Train Loss:{round(train_loss,6)} Test Loss:{round(test_loss,6)}", ha="center", fontsize=18)
plt.plot(data.index[:len(train_predictions)], data.values[:len(train_predictions)], label='Original Data (Train)')
plt.plot(data.index[len(train_predictions):split], data.values[len(train_predictions):split], label='Original Data (Test)')
plt.plot(data.index[time_steps:len(train_predictions)+time_steps], train_predictions, label='Predictions (Train)')
plt.plot(data.index[split+time_steps:], test_predictions, label='Predictions (Test)')
plt.legend()
plt.savefig("./Figures/" + model_title + ".png")
plt.show()

