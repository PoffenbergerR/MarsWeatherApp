from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from pathlib import Path

current_path = Path(__file__).parent.absolute()

def create_mars_xgboost(show_metrics=True):
    # Load the dataset without attempting to parse dates
    d = pd.read_json(str(current_path.parent) + '/data/MarsWeatherData.json')
    filtered_df = d["data"]["soles"]
    df = pd.DataFrame(filtered_df)

    df = df.set_index("terrestrial_date")

    variables = ["min_temp", "max_temp", "pressure", "local_uv_irradiance_index", 
                "min_gts_temp", "max_gts_temp", "ls"]
    
    df = df[variables]
    uv_numeric = {
        "Very_Low": 0,
        "Low": 1,
        "Moderate": 2,
        "High": 3,
        "Very_High": 4
    }
    df['local_uv_irradiance_index'] = df['local_uv_irradiance_index'].map(uv_numeric)
    #replace '--' with NaN so ffill() can replace it with last relevant value
    #Not optimal, but what can you do. NaN values are despised in ML
    df = df.replace('--', np.nan)
    df = df.ffill()

    df = df.iloc[::-1] #make it so that the dataframe is in ascending order for the dates
    # print(df.info())

    # create lag data by shifting a day ahead so xgboost will try to use 
    # X_train (day before) to predict y_train (current/day ahead)
    df["lag_min_temp"] = df["min_temp"].shift(1)
    df = df.dropna()

    df["lag_max_temp"] = df["max_temp"].shift(1)
    df = df.dropna()

    df["lag_pressure"] = df["pressure"].shift(1)
    df = df.dropna()

    df["lag_uv_index"] = df["local_uv_irradiance_index"].shift(1)
    df = df.dropna()

    df["lag_min_gts_temp"] = df["min_gts_temp"].shift(1)
    df = df.dropna()

    df["lag_max_gts_temp"] = df["max_gts_temp"].shift(1)
    df = df.dropna()

    df["lag_ls"] = df["ls"].shift(1)
    df = df.dropna()

    lag_features = ['lag_min_temp', 'lag_max_temp', 'lag_pressure', 
                       'lag_uv_index', 'lag_min_gts_temp', 'lag_max_gts_temp',
                       'lag_ls']

    for n in variables:
        df[n] = pd.to_numeric(df[n])
    for n in lag_features:
        df[n] = pd.to_numeric(df[n])

    X, y = df.drop(variables, axis=1), df.drop(lag_features + ["ls"], axis=1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Create regression matrices
    dtrain_reg = xgb.DMatrix(X_train, y_train)
    dtest_reg = xgb.DMatrix(X_test, y_test)

    params = {"objective": "reg:squarederror", 
            "tree_method": "hist",
            "multi_strategy": "multi_output_tree", 
            "eta": 0.1, 
        #   'subsample': 0.8,
            }

    evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]
    n = 55
    model = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=n,
    evals=evals,
    )

    preds = model.predict(dtest_reg)

    if show_metrics:
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)

        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)

        for i in range(len(variables) - 1):
            mae = mean_absolute_error(y_test.iloc[:, i], preds[:, i])
            mse = mean_squared_error(y_test.iloc[:, i], preds[:, i])
            rmse = np.sqrt(mse)

            print(f"{variables[i]} Mean Absolute Error (MAE):", mae)
            print(f"{variables[i]} Mean Squared Error (MSE):", mse)
            print(f"{variables[i]} Root Mean Squared Error (RMSE):", rmse)
        
        # Plotting predicted and actual temperatures
        for i in range(len(variables) - 1):
            plt.figure(figsize=(10, 6))
            plt.plot(y.index[-100:], y_test.iloc[-100:, i], label='Actual')
            print(y_test.iloc[-1:, i])
            #print("dates: ", df.index[-100:], "actual: ", y[-100:])
            #print("dates: ", df.index[-100:], "predicted: ", preds[-100:])
            plt.plot(y.index[-100:], preds[-100:, i], label='Predicted')
            print(preds[-1:, i])
            plt.title('XGBoost: Prediction vs Actual')
            plt.xlabel('Terrestrial Date')
            plt.ylabel(f'{variables[i]}')
            plt.gcf().autofmt_xdate()
            plt.xticks(np.arange(0,100,9))
            plt.tight_layout()
            plt.legend()
            plt.show()

    d_next_day = xgb.DMatrix(X_test[-1:])
    future_preds = model.predict(d_next_day)
    print(future_preds)
    model.save_model(str(current_path) + "/xgboost_model.json")

def xgboost_next_day_pred():
    d = pd.read_json(str(current_path.parent) + '/data/MarsWeatherData.json')
    filtered_df = d["data"]["soles"]
    df = pd.DataFrame(filtered_df)

    df = df.set_index("terrestrial_date")
    variables = ["min_temp", "max_temp", "pressure", "local_uv_irradiance_index", 
                 "min_gts_temp", "max_gts_temp", "ls"]
    
    df = df[variables]
    df = df.rename(columns={"min_temp": "lag_min_temp", "max_temp": "lag_max_temp", "pressure": "lag_pressure",
                       "local_uv_irradiance_index": "lag_uv_index", "min_gts_temp": "lag_min_gts_temp",
                       "max_gts_temp": "lag_max_gts_temp", "ls": "lag_ls"})
    df = df[:1]
    print(df)

    uv_numeric = {
        "Very_Low": 0,
        "Low": 1,
        "Moderate": 2,
        "High": 3,
        "Very_High": 4
    }
    df['lag_uv_index'] = df['lag_uv_index'].map(uv_numeric)
    df = df.replace('--', np.nan)
    df = df.ffill()

    df = df.iloc[::-1] #make it so that the dataframe is in ascending order for the dates
    lag_features = ['lag_min_temp', 'lag_max_temp', 'lag_pressure', 
                    'lag_uv_index', 'lag_min_gts_temp', 'lag_max_gts_temp',
                    'lag_ls']
    for n in lag_features:
        df[n] = pd.to_numeric(df[n])

    loaded_model = xgb.Booster()
    loaded_model.load_model(str(current_path) + "/xgboost_model.json")
    d_next_day = xgb.DMatrix(df)
    future_pred = loaded_model.predict(d_next_day)
    return future_pred

if __name__ == "__main__":
    create_mars_xgboost()
    furutre_pred = xgboost_next_day_pred()
    print(furutre_pred)