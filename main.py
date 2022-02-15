import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras import Sequential




# user inputs wat crypto to analyze
crypto_curr = input("What crypto would you like us to predict about? (BTC, ETH, etc)\n").upper()
against_curr = input("What currency would you like us to predict it against? (USD, EUR, etc)\n").upper()
number_of_day = input("Would you like to predict\n1. The next day\n2. More than one day into the future\n")
if number_of_day == '1':
    future_days = 0
elif number_of_day == '2':
    future_days = int(input("How many days into the future?\n"))


# training
start_training = dt.datetime(2015, 1, 1)
end_training = dt.datetime.now()
training_data = web.DataReader(f'{crypto_curr}-{against_curr}', 'yahoo', start_training, end_training)

# prep training data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_training_data = scaler.fit_transform(training_data['Close'].values.reshape(-1, 1))

# init number of days and training lists
training_days = 60
x_train, y_train = [], []

# fill training lists with data
for data in range(training_days, len(scaled_training_data) - future_days):
    x_train.append(scaled_training_data[data - training_days:data, 0])  # append packages of 70 days
    y_train.append(scaled_training_data[data + future_days, 0])  # append the last day in the package. data is not included in training days

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# init neural network
model = Sequential()

# Layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# testing model
test_start = dt.datetime(2021, 1, 1)
test_end = dt.datetime.now()
test_data = web.DataReader(f'{crypto_curr}-{against_curr}', 'yahoo', test_start, test_end)
# prices from chart on yahoo
real_prices = test_data['Close'].values  # only look at the closing prices in the list

# combine both data sets (test data and training)
total_set = pd.concat((training_data['Close'], test_data['Close']), axis=0)

# Init inputs for the model to make prediction
model_inputs = total_set[len(total_set) - len(test_data) - training_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

# create test data
x_test = []
for i in range(training_days, len(model_inputs)):
    x_test.append(model_inputs[i-training_days:i, 0])
# make it a np array and shape it
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
price_prediction = model.predict(x_test)
price_prediction = scaler.inverse_transform(price_prediction)

# plot predictions in matplotlib
# plt.title(f'{crypto_curr} - {against_curr} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.plot(real_prices, color='Black', label='Actual Prices')
plt.plot(price_prediction, color='green', label='Predicted Prices')

plt.show()

# actual prediction
real_data = [model_inputs[len(model_inputs) - training_days:len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f'{crypto_curr} will be at {prediction}')




