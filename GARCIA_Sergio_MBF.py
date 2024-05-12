# Sergio Garcia
# Neuronal Network Model to predict stock prices


import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Paramètres
# Actions à predire (10 entreprises dans l'indice CAC40)
symbols = ['AAPL', 'BAC', 'AXP', 'KO', 'CVX', 'OXY', 'BRK-B', 'EC', 'BLK','MCO',
           'PARA', 'V', 'MA', 'TSLA', 'TSN']
symbol_to_company = {'AAPL': 'Apple', 'BAC' : 'Bank of America', 'AXP' : 'Amex',
                     'KO' : 'Coca-Cola', 'CVX' : 'Chevron', 'OXY' : 'Occ Petrol',
                     'BRK-B' : 'Berks-Hath', 'EC': 'Ecopetrol', 'BLK' : 'BlackRock',
                    'MCO' : 'Moodys', 'PARA' : 'Paramount' , 'V': 'Visa', 
                    'MA': 'Mastercard', 'TSLA' : 'Tesla', 'TSN' : 'Tyson Foods'}
column = 'Close'
freq = '1d'
prediction_days = 60
start_date = '2019-01-01'
end_date = '2024-01-01'


# Construction du modèle
def download_stock_data(symbol, start_date, end_date, interval):
    stock_data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    return stock_data


def prepare_data(stock_data, prediction_days,column):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[column].values.reshape(-1, 1))

    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data)):  
        x_train.append(scaled_data[x - prediction_days:x,0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler


def build_model(input_shape, dropout_rate=0.2, neurons=50, optimizer='adam', loss='mean_squared_error'):
    model = Sequential()

    model.add(LSTM(units=neurons, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
           
    model.add(LSTM(units=neurons, return_sequences=True))
    model.add(Dropout(dropout_rate))
           
    model.add(LSTM(units=neurons))
    model.add(Dropout(dropout_rate))
           
    model.add(Dense(units=1))
    model.compile(optimizer=optimizer, loss=loss)
    return model


def train_model(model, x_train, y_train, epochs, batch_size):
    model = KerasRegressor(build_fn=build_model, epochs=epochs, batch_size=batch_size, verbose=0)
    param_grid = {
        'dropout_rate': [0.2, 0.3, 0.4],  
        'neurons': [50, 64, 128],  
        'optimizer': ['adam', 'rmsprop']  
    }
           
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    model = build_model(input_shape=(x_train.shape[1], 1), dropout_rate=best_params['dropout_rate'],
                        neurons=best_params['neurons'], optimizer=best_params['optimizer'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model


def prepare_test_data(model_inputs, prediction_days, scaler):
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_test


def make_predictions(model, x_test, scaler):
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices

    # Comparer visuellement les valeurs réelles avec les prédictions
def plot_predictions(actual_prices, predicted_prices, symbol):
    plt.plot(actual_prices, color="black", label="Actual Price")
    plt.plot(predicted_prices, color="blue", label="Predicted Price")
    plt.title(f"Stock Price Prediction - {symbol}")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()


def main():
    predictions_dict = {}
    last_observed_prices = {}
    for symbol in symbols:
        comp_data = download_stock_data(symbol, start_date, end_date, freq)
        x_train, y_train, scaler = prepare_data(comp_data, prediction_days, column)
        input_shape = (x_train.shape[1], 1)
        model = build_model(input_shape)

        epochs = 50
        batch_size = 64
        print(f"Training model for symbol {symbol}")
        train_model(model, x_train, y_train, epochs, batch_size)
        print(f"Training for symbol {symbol} completed\n")

        test_start = dt.datetime(2023, 1, 1)
        test_end = dt.datetime.now()
        test_comp1 = download_stock_data(symbol, test_start, test_end, freq)

        total_dataset = pd.concat((comp_data[column], test_comp1[column]),axis=0)
        model_inputs = total_dataset[len(total_dataset) - len(test_comp1) - prediction_days:]
        model_inputs = model_inputs.values.reshape(-1,1)
        model_inputs = scaler.transform(model_inputs)  
        
        x_test = prepare_test_data(model_inputs, prediction_days, scaler)
        predicted_prices = make_predictions(model, x_test, scaler)

        actual_prices = test_comp1[column].values
        plot_predictions(actual_prices, predicted_prices,symbol)

        real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

        last_observed_price = scaler.inverse_transform(real_data[-1])[-1][0]
        last_observed_prices[symbol] = last_observed_price

        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)

        print(f'Prediction: {prediction[0][0]}')
        predictions_dict[symbol] = prediction[0][0]



if __name__ == "__main__":
    main()

# Tableau des résultats
today = dt.datetime.now().strftime("%d-%m-%Y")
current_date = dt.datetime.now()
tomorrow = current_date + dt.timedelta(days=1)
tomorrow = tomorrow.strftime("%d-%m-%Y")

print("\nPredictions:")
table_data = [(symbol, symbol_to_company.get(symbol, symbol), last_observed_prices[symbol], prediction,
               (prediction - last_observed_prices[symbol]) / last_observed_prices[symbol] * 100)
              for symbol, prediction in predictions_dict.items()]
headers = ["Symbol", "Company Name", f"Last Observed Price ({today})", f"Prediction ({tomorrow})", "Percent Change"]
print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

