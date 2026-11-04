# XGBoost & LSTM testing
# Purpose is to test the accuracy of XGBoost and LSTM when predicting Martian weather
# Save figures and rsme/accuracy math stuff to save for paper
# Maybe save that for app too as it can be a cool display saying "here look at this!"
#______________________________________________________________________________________

#could save data as csv file once I'm happy with the feature column used for training
#could save the predicted values so we don't have to run the models each time
#the program is called. Instead only run them when date/info has been updated.
#so i guess we need somewhere to store predicted dates that we can pull from
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, TimeDistributed
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

#https://stackoverflow.com/questions/69906416/forecast-future-values-with-lstm-in-python
#The link above has a really good explanation for forcasting for future dates.
def mars_lstm():
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
        # label = scaled_data[i+sequence_length][1]  # 'max_temp' column index
        label = scaled_data[i+sequence_length]  # 'max_temp' column index
        sequences.append(seq)
        labels.append(label)

    # Convert to numpy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)

    # Split into train and test sets
    train_size = int(0.85 * len(sequences))
    train_x, test_x = sequences[:train_size], sequences[train_size:]
    train_y, test_y = labels[:train_size], labels[train_size:]

    print("Train X shape:", train_x.shape)
    print("Train Y shape:", train_y.shape)
    print("Test X shape:", test_x.shape)
    print("Test Y shape:", test_y.shape)

    model = Sequential()
    model.add(Input(shape = (train_x.shape[1], train_x.shape[2])))
    model.add(LSTM(100))
    # model.add(LSTM(100, input_shape=(
    #     train_x.shape[1], train_x.shape[2])))
    model.add(Dropout(0.2))
    # model.add(LSTM(50, return_sequences=False))
    # model.add(Dropout(0.2))
    model.add(Dense(train_x.shape[2]))
    # model.add(Dense(1, activation="linear"))

    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    #model.compile(loss="mse", optimizer="adam", metrics=["mae", keras.metrics.R2Score()]) #very close to just "mae", but slightly worse
    # model.compile(loss="mse", optimizer="rmsprop", metrics=["mae"])

    history = model.fit(train_x, train_y, epochs=30, batch_size=16) #added shuffle false, still need to test

    # # Define callbacks
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # model_checkpoint = ModelCheckpoint('best_model_weights.h5', monitor='val_loss', save_best_only=True)

    # # Train the model
    # history = model.fit(
    #     train_x, train_y,
    #     epochs=100,
    #     batch_size=64,
    #     shuffle=False,
    #     validation_split=0.2,  # Use part of the training data as validation
    #     callbacks=[early_stopping, model_checkpoint]
    # )

    # # Evaluate the best model on the test set
    # best_model = load_model('best_model_weights.h5')
    # test_loss = best_model.evaluate(test_x, test_y)
    # print("Test Loss:", test_loss)

    # Plot training & validation loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend(['Train', 'Validation'], loc='upper right')
    # plt.show()

    # Assuming you have trained the model and have the 'best_model' object
    # Also, 'test_x' and 'test_y' should be available

    # Predict temperatures using the trained model
    #predictions = best_model.predict(test_x)
    predictions = model.predict(test_x)

    # Calculate evaluation metrics
    mae = mean_absolute_error(test_y, predictions)
    mse = mean_squared_error(test_y, predictions)
    rmse = np.sqrt(mse)

    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    # Calculate evaluation metrics
    mae = mean_absolute_error(test_y[:][:,1], predictions[:][:,1])
    mse = mean_squared_error(test_y[:][:,1], predictions[:][:,1])
    rmse = np.sqrt(mse)

    print("Temp Mean Absolute Error (MAE):", mae)
    print("Temp Mean Squared Error (MSE):", mse)
    print("Temp Root Mean Squared Error (RMSE):", rmse)

    # y_true values
    #test_y_copies = np.repeat(test_y.reshape(-1, 1), test_x.shape[-1], axis=-1)
    # true_temp = scaler.inverse_transform(test_y_copies)[:,1] #2 is for max_temp index
    true_temp = scaler.inverse_transform(test_y)

    # predicted values
    prediction = model.predict(test_x)
    # prediction_copies = np.repeat(prediction, num_features, axis=-1)
    #prediction_copies = np.repeat(prediction.reshape(-1,1), test_x.shape[-1], axis=-1)
    # predicted_temp = scaler.inverse_transform(prediction_copies)[:,1]
    predicted_temp = scaler.inverse_transform(prediction)
    
    # Plotting predicted and actual temperatures
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-100:], true_temp[-100:][:,1], label='Actual')
    print("dates: ", df.index[-10:], "actual: ", true_temp[-10:][:,1])
    print("dates: ", df.index[-10:], "predicted: ", predicted_temp[-10:][:,1])
    plt.plot(df.index[-100:], predicted_temp[-100:][:,1], label='Predicted')
    plt.title('LSTM: Temperature Prediction vs Actual')
    plt.xlabel('Terrestrial Date')
    plt.ylabel('Max_Temp')
    plt.gcf().autofmt_xdate()
    plt.xticks(np.arange(0,100,9))
    plt.tight_layout()
    plt.legend()
    plt.show()

    latest_sequence = scaled_data[-sequence_length:]
    last_window_3d = np.expand_dims(latest_sequence, axis=0)
    predicted_scaled = model.predict(last_window_3d)
    # prediction_copies = np.repeat(predicted_scaled, num_features, axis=-1)
    # prediction_copies = np.repeat(predicted_scaled.reshape(-1,1), test_x.shape[-1], axis=-1)
    #predicted_temp = scaler.inverse_transform(prediction_copies)[:,1]
    predicted_temp = scaler.inverse_transform(predicted_scaled)
    print("Predictions:", predicted_temp)
    for p in predicted_temp:
        for x in p:
            print(x)

    # dummy_array = np.zeros((1, num_features))
    # dummy_array[0, 0] = predicted_scaled[0, 0]
    # predicted = scaler.inverse_transform(dummy_array)[0][0]
    # print(predicted)
    
    print(f"predicted maximum temperature for the next day: {predicted_temp[0][1]}")

#_____________________________________________________________________________________
#XGBoost
def mars_xgboost():
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
    numeric_var = ["min_temp", "max_temp", "pressure", "min_gts_temp", "max_gts_temp", "ls"]
    for n in numeric_var:
        df[n] = pd.to_numeric(df[n])
    print(df)

    df = df.iloc[::-1] #make it so that the dataframe is in ascending order for the dates
    print(df.info())

    X, y = df.drop('max_temp', axis=1), df[['max_temp']]

    # Extract text features
    cats = X.select_dtypes(exclude=np.number).columns.tolist()

    # Convert to Pandas category
    for col in cats:
        X[col] = X[col].astype('category')
    print(X.dtypes)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Create regression matrices
    dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    # Define hyperparameters
    params = {"objective": "reg:squarederror", 
              "tree_method": "hist", 
              "eta": 0.1, 
            #   'subsample': 0.8,
              }

    evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]
    n = 55
    # it's xgboost train
    model = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=n,
    evals=evals,
    )

    preds = model.predict(dtest_reg)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    # Plotting predicted and actual temperatures
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-100:], y[-100:], label='Actual')
    #print("dates: ", df.index[-100:], "actual: ", y[-100:])
    #print("dates: ", df.index[-100:], "predicted: ", preds[-100:])
    plt.plot(df.index[-100:], preds[-100:], label='Predicted')
    plt.title('XGBoost: Temperature Prediction vs Actual')
    plt.xlabel('Terrestrial Date')
    plt.ylabel('Max_Temp')
    plt.gcf().autofmt_xdate()
    plt.xticks(np.arange(0,100,9))
    plt.tight_layout()
    plt.legend()
    plt.show()

mars_lstm()
#mars_xgboost()