import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras import saving
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, Input
from keras._tf_keras.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_mars_lstm(show_metrics=True):
    # Load the dataset without attempting to parse dates
    d = pd.read_json('MarsWeatherData.json')
    filtered_df = d["data"]["soles"]
    df = pd.DataFrame(filtered_df)

    #uv index has Very_High, High, Moderate, Low, Very_Low
    #season has month in it so I need to drop that so it's just the number
    #sunrise and sunset has colon, probably need to fix that
    #opacity seems to only be sunny?
    df = df.set_index("terrestrial_date")

    # put these back in later maybe just don't want to bother right now
    # "season", "sunrise", "sunset", "ls"
    variables = ["min_temp", "max_temp", "pressure", "local_uv_irradiance_index", "min_gts_temp", "max_gts_temp", "ls"]
    df = df[variables]

    #replace '--' with NaN so ffill() can replace it with last relevant value
    #Not optimal, but what can you do. NaN values are despised in ML
    df = df.replace('--', np.nan)
    df = df.ffill()
    print(df)

    df = df.iloc[::-1] #make it so that the dataframe is in ascending order for the dates
    print(df)
    #I haven't seen any very_low values, but maybe there could be one in the future
    uv_numeric = {
        "Very_Low": 0,
        "Low": 1,
        "Moderate": 2,
        "High": 3,
        "Very_High": 4
    }

    #so far, the data only has sunny, but based on description, there could also be cloudy or windy
    # opacity_numeric = {
    #     "Sunny": 0,
    #     "Cloudy": 1,
    #     "Windy": 2
    # }

    # df['atmo_opacity'] = df['atmo_opacity'].map(opacity_numeric)
    df['local_uv_irradiance_index'] = df['local_uv_irradiance_index'].map(uv_numeric)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df)

    # Define sequence length and features
    sequence_length = 50  # Number of time steps in each sequence 50 seems to do best
    num_features = len(df.columns)

    # Create sequences and corresponding labels
    sequences = []
    labels = []
    for i in range(len(scaled_data) - sequence_length):
        seq = scaled_data[i:i+sequence_length]
        label = scaled_data[i+sequence_length]  #This will make it predict every variable
        sequences.append(seq)
        labels.append(label)

    # Convert to numpy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)

    # Split into train and test sets
    train_size = int(0.85 * len(sequences))
    train_x, test_x = sequences[:train_size], sequences[train_size:]
    train_y, test_y = labels[:train_size], labels[train_size:]

    model = Sequential()
    model.add(Input(shape = (train_x.shape[1], train_x.shape[2])))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(train_x.shape[2])) #Right now the shape is 7 as there are 7 variables to make predictions for

    model.compile(loss="mse", optimizer="adam", metrics=["mae"])

    history = model.fit(train_x, train_y, epochs=30, batch_size=16)

    model.save("lstm_model.keras")

    predictions = model.predict(test_x)
    if show_metrics:
        # Calculate evaluation metrics
        mae = mean_absolute_error(test_y, predictions)
        mse = mean_squared_error(test_y, predictions)
        rmse = np.sqrt(mse)

        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)

        for i in range(len(variables)):
            # Calculate evaluation metrics
            mae = mean_absolute_error(test_y[:][:,i], predictions[:][:,i])
            mse = mean_squared_error(test_y[:][:,i], predictions[:][:,i])
            rmse = np.sqrt(mse)

            print(f"{variables[i]} Mean Absolute Error (MAE):", mae)
            print(f"{variables[i]} Mean Squared Error (MSE):", mse)
            print(f"{variables[i]} Root Mean Squared Error (RMSE):", rmse)

        true_temp = scaler.inverse_transform(test_y)
        predicted_temp = scaler.inverse_transform(predictions)

        #Still need to test this
        for i in range(len(variables)):
            # Plotting predicted and actual temperatures
            plt.figure(figsize=(10, 6))
            plt.plot(df.index[-100:], true_temp[-100:][:,i], label='Actual')
            # print("dates: ", df.index[-10:], "actual: ", true_temp[-10:][:,1])
            # print("dates: ", df.index[-10:], "predicted: ", predicted_temp[-10:][:,1])
            plt.plot(df.index[-100:], predicted_temp[-100:][:,i], label='Predicted')
            plt.title('LSTM: Prediction vs Actual')
            plt.xlabel('Terrestrial Date')
            plt.ylabel(variables[i])
            plt.gcf().autofmt_xdate()
            plt.xticks(np.arange(0,100,9))
            plt.tight_layout()
            plt.legend()
            plt.show()

def lstm_next_day_pred():
    # down below loads the lstm model to predict the next day based on the last/most recent 50 days
    d = pd.read_json('MarsWeatherData.json')
    filtered_df = d["data"]["soles"]
    df = pd.DataFrame(filtered_df)

    df = df.set_index("terrestrial_date")

    variables = ["min_temp", "max_temp", "pressure", "local_uv_irradiance_index", "min_gts_temp", "max_gts_temp", "ls"]
    df = df[variables]

    # this just takes the first 50 most recent records so the computer 
    # doesn't have to do so much processing
    df = df[:50] 

    df = df.replace('--', np.nan)
    df = df.ffill()
    print(df)

    df = df.iloc[::-1] #make it so that the dataframe is in ascending order for the dates
    print(df)

    uv_numeric = {
        "Very_Low": 0,
        "Low": 1,
        "Moderate": 2,
        "High": 3,
        "Very_High": 4
    }

    df['local_uv_irradiance_index'] = df['local_uv_irradiance_index'].map(uv_numeric)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df)


    loaded_model = saving.load_model("lstm_model.keras")

    last_window_3d = np.expand_dims(scaled_data, axis=0)
    predicted_scaled = loaded_model.predict(last_window_3d)
    future_pred = scaler.inverse_transform(predicted_scaled)

    return future_pred

if __name__ == "__main__":
    future_pred = lstm_next_day_pred()
    print(future_pred)