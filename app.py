import streamlit as st
from datetime import date
import pandas as pd
from pathlib import Path

data_path = Path(__file__).resolve().parent / "data" / "MarsWeatherData.json"
d = pd.read_json(data_path)
filtered_df = d["data"]["soles"]
df = pd.DataFrame(filtered_df)
df = df.set_index("terrestrial_date")
df.rename(columns={"min_temp": "Min Temperature", "max_temp": "Max Temperature",
                   "pressure": "Pressure", "atmo_opacity":"Atmosphere Opacity",
                   "local_uv_irradiance_index": "UV Irradiance Index",
                   "min_gts_temp": "Min Ground Temperature",
                   "max_gts_temp": "Max Ground Temperature",
                   "sunrise": "Sunrise", "sunset": "Sunset", "season": "Season"}, inplace=True)

lstm_preds_path = Path(__file__).resolve().parent / "data" / "lstm_predictions.csv"
lstmdf = pd.read_csv(lstm_preds_path)
lstmdf = lstmdf.set_index("terrestrial_date")

most_recent_date = df.index[0]
min_date = df.index[-1]
day_ahead = lstmdf.index[-1]

def fill_data(chosen=most_recent_date):
    var = ["Min Temperature", "Sunrise", "Max Temperature", "Sunset", 
             "Min Ground Temperature", "UV Irradiance Index", "Max Ground Temperature",
             "Atmosphere Opacity", "Pressure", "Season"]

    left2, right2 = weather_data.columns(2)
    #This is for when the requested day does not exist
    if chosen not in df.index:
        track = 0
        for v in var:
            if track%2 == 0:
                left2.container(border=True).text(f"{v}: -- ")
            else:
                right2.container(border=True).text(f"{v}: -- ")
            track += 1

    else:
        if st.session_state.temp_unit == "Celsius":
            left2.container(border=True).text(f"{var[0]}: {df.at[chosen, var[0]]}°C")
            left2.container(border=True).text(f"{var[2]}: {df.at[chosen, var[2]]}°C")
            left2.container(border=True).text(f"{var[4]}: {df.at[chosen, var[4]]}°C")
            left2.container(border=True).text(f"{var[6]}: {df.at[chosen, var[6]]}°C")

        elif st.session_state.temp_unit == "Fahrenheit":
            if df.at[chosen, var[0]] != "--":
                left2.container(border=True).text(f"{var[0]}: {round(int(df.at[chosen, var[0]]) * 1.8 + 32)}°F")
            else:
                left2.container(border=True).text(f"{var[0]}: --°F")
            if df.at[chosen, var[2]] != "--":
                left2.container(border=True).text(f"{var[2]}: {round(int(df.at[chosen, var[2]]) * 1.8 + 32)}°F")
            else:
                left2.container(border=True).text(f"{var[2]}: --°F")
            if df.at[chosen, var[4]] != "--":
                left2.container(border=True).text(f"{var[4]}: {round(int(df.at[chosen, var[4]]) * 1.8 + 32)}°F")
            else:
                left2.container(border=True).text(f"{var[4]}: --°F")
            if df.at[chosen, var[6]] != "--":
                left2.container(border=True).text(f"{var[6]}: {round(int(df.at[chosen, var[6]]) * 1.8 + 32)}°F")
            else:
                left2.container(border=True).text(f"{var[6]}: --°F")

        elif st.session_state.temp_unit == "Kelvin":
            if df.at[chosen, var[0]] != "--":
                left2.container(border=True).text(f"{var[0]}: {round(int(df.at[chosen, var[0]]) + 273.15)}°K")
            else:
                left2.container(border=True).text(f"{var[0]}: --°K")
            if df.at[chosen, var[2]] != "--":
                left2.container(border=True).text(f"{var[2]}: {round(int(df.at[chosen, var[2]]) + 273.15)}°K")
            else:
                left2.container(border=True).text(f"{var[2]}: --°K")
            if df.at[chosen, var[4]] != "--":
                left2.container(border=True).text(f"{var[4]}: {round(int(df.at[chosen, var[4]]) + 273.15)}°K")
            else:
                left2.container(border=True).text(f"{var[4]}: --°K")
            if df.at[chosen, var[6]] != "--":
                left2.container(border=True).text(f"{var[6]}: {round(int(df.at[chosen, var[6]]) + 273.15)}°K")
            else:
                left2.container(border=True).text(f"{var[6]}: --°K")
        
        #for the pressure
        if df.at[chosen, var[8]] == "--":
            left2.container(border=True).text(f"{var[8]}: --")
        elif st.session_state.pressure_unit == "Pascal (Pa)":
            left2.container(border=True).text(f"{var[8]}: {df.at[chosen, var[8]]} Pa")
        elif  st.session_state.pressure_unit == "Millibar (mbar)":
            left2.container(border=True).text(f"{var[8]}: {round(int(df.at[chosen, var[8]])/100)} mbar")
        elif st.session_state.pressure_unit == "Standard atmosphere (atm)":
            left2.container(border=True).text(f"{var[8]}: {round(int(df.at[chosen, var[8]])/101325, ndigits=5)} atm")

        #data with no unit of measurement options
        right2.container(border=True).text(f"{var[1]}: {df.at[chosen, var[1]]}")
        right2.container(border=True).text(f"{var[3]}: {df.at[chosen, var[3]]}")
        right2.container(border=True).text(f"{var[5]}: {df.at[chosen, var[5]]}")
        right2.container(border=True).text(f"{var[7]}: {df.at[chosen, var[7]]}")
        right2.container(border=True).text(f"{var[9]}: {df.at[chosen, var[9]]}")

def change_date():
    chosen_date = st.session_state.my_date_picker
    date_str = chosen_date.strftime("%Y-%m-%d")
    fill_data(date_str)

#This just centers the title
# l, m, r = st.columns([1, 4, 1])
# m.markdown("""# :orange[Mars Weather Forecast]""")
st.markdown("<h1 style='text-align: center; color: #921f02;'>Mars Weather Forecast</h1>", unsafe_allow_html=True)

image_path = Path(__file__).resolve().parent / "images" / "mars.png"

left, center, right= st.columns([1, 2, 1])

with center:
    st.image(
    
        str(image_path),
        caption="This self-portrait of NASA's Curiosity Mars rover shows the vehicle on " \
        "Vera Rubin Ridge, which it's been investigating for the past several months. " \
        "Directly behind the rover is the start of a clay-rich slope scientists are eager to " \
        "begin exploring. In the coming week, Curiosity will begin to climb this slope. " \
        "North is on the left and west is on the right, with Gale Crater's rim on the horizon " \
        "of both edges.",
        width=500
)

left, middle, right = st.columns([2, 1, 2])
middle.date_input(
    "Terrestrial Date", 
    value=date.fromisoformat(most_recent_date), 
    min_value=date.fromisoformat(min_date),
    max_value=date.fromisoformat(day_ahead),
    on_change=change_date,
    key="my_date_picker",
)

if "my_date_picker" not in st.session_state:
    st.session_state.my_date_picker = date.fromisoformat(most_recent_date)

if "temp_unit" not in st.session_state:
    st.session_state.temp_unit = "Celsius"

if "pressure_unit" not in st.session_state:
    st.session_state.pressure_unit = "Pascal (Pa)"

sol = middle.empty()
ls = middle.empty()
weather_data = st.empty()

with sol.container():
    chosen_date = st.session_state.my_date_picker
    date_str = chosen_date.strftime("%Y-%m-%d")
    if date_str not in df.index:
        sol.container().text(f"SOL: --", 
                             help="A day on Mars is a sol, which is about 24 hours and 40 minutes. " \
                             "Here, the sol represents the number of days Curiosity has been on Mars.")
    else:
        s = df.at[date_str, "sol"]
        sol.container().text(f"SOL: {s}", 
                             help="A day on Mars is a sol, which is about 24 hours and 40 minutes. " \
                             "Here, the sol represents the number of days Curiosity has been on Mars.")
   
with ls.container():
    chosen_date = st.session_state.my_date_picker
    date_str = chosen_date.strftime("%Y-%m-%d")
    if date_str not in df.index:
        ls.container().text(f"LS: --",
                            help="A Martian year lasts about two Earth's year. Solar longitude "
                            "is an angle that gives the position of Mars on its orbit.")
    else:
        l = df.at[date_str, "ls"]
        ls.container().text(f"LS: {l}",
                            help="A Martian year lasts about two Earth's year. Solar longitude "
                            "is an angle that gives the position of Mars on its orbit.")

with weather_data.container():
    chosen_date = st.session_state.my_date_picker
    date_str = chosen_date.strftime("%Y-%m-%d")
    fill_data(date_str)
    
    
st.divider()
with st.container():
    chosen_date = st.session_state.my_date_picker
    chosen = chosen_date.strftime("%Y-%m-%d")

    st.subheader('LSTM Model Predictions')
    pred_left, pred_right = st.columns(2)
    if chosen not in lstmdf.index:
        with pred_left:
            st.metric(
                        label="Predicted Min Temperature",
                        value="--"
                    )
            st.metric(
                        label="Predicted Max Temperature",
                        value="--"
                    )
            st.metric(
                        label="Predicted Min Ground Temperature",
                        value="--"
                    )
            st.metric(
                        label="Predicted Max Ground Temperature",
                        value="--"
                    )

            
        with pred_right:
            st.metric(
                        label="Predicted Atomospheric Pressure",
                        value="--"
                    )
            st.metric(
                        label="Predicted UV Index",
                        value="--"
                    )
            
            st.metric("Prediction Error", "--")
    else:
        if st.session_state.temp_unit == "Celsius":
            with pred_left:
                st.metric(
                            label="Predicted Min Temperature",
                            value=f"{round(int(lstmdf.at[chosen, 'min_temp']))} °C"
                        )
                st.metric(
                            label="Predicted Max Temperature",
                            value=f"{round(int(lstmdf.at[chosen, 'max_temp']))} °C"
                        )
                st.metric(
                            label="Predicted Min Ground Temperature",
                            value=f"{round(int(lstmdf.at[chosen, 'min_gts_temp']))} °C"
                        )
                st.metric(
                            label="Predicted Max Ground Temperature",
                            value=f"{round(int(lstmdf.at[chosen, 'max_gts_temp']))} °C"
                        )

        elif st.session_state.temp_unit == "Fahrenheit":
            with pred_left:
                st.metric(
                            label="Predicted Min Temperature",
                            value=f"{round(int(lstmdf.at[chosen, 'min_temp']) * 1.8 + 32)} °F"
                        )
                st.metric(
                            label="Predicted Max Temperature",
                            value=f"{round(int(lstmdf.at[chosen, 'max_temp']) * 1.8 + 32)} °F"
                        )
                st.metric(
                            label="Predicted Min Ground Temperature",
                            value=f"{round(int(lstmdf.at[chosen, 'min_gts_temp']) * 1.8 + 32)} °F"
                        )
                st.metric(
                            label="Predicted Max Ground Temperature",
                            value=f"{round(int(lstmdf.at[chosen, 'max_gts_temp']) * 1.8 + 32)} °F"
                        )

        elif st.session_state.temp_unit == "Kelvin":
            with pred_left:
                st.metric(
                            label="Predicted Min Temperature",
                            value=f"{round(int(lstmdf.at[chosen, 'min_temp']) + 273.15)} °K"
                        )
                st.metric(
                            label="Predicted Max Temperature",
                            value=f"{round(int(lstmdf.at[chosen, 'max_temp']) + 273.15)} °K"
                        )
                st.metric(
                            label="Predicted Min Ground Temperature",
                            value=f"{round(int(lstmdf.at[chosen, 'min_gts_temp']) + 273.15)} °K"
                        )
                st.metric(
                            label="Predicted Max Ground Temperature",
                            value=f"{round(int(lstmdf.at[chosen, 'max_gts_temp']) + 273.15)} °K"
                        )

        if  st.session_state.pressure_unit == "Pascal (Pa)":
            with pred_right:
                st.metric(
                            label="Predicted Atomospheric Pressure",
                            value=f"{round(int(lstmdf.at[chosen, 'pressure']))} Pa"
                        )
        if  st.session_state.pressure_unit == "Millibar (mbar)":
            with pred_right:
                st.metric(
                            label="Predicted Atomospheric Pressure",
                            value=f"{round(int(lstmdf.at[chosen, 'pressure'])/100)} mbar"
                        )
        if  st.session_state.pressure_unit == "Standard atmosphere (atm)":
            with pred_right:
                st.metric(
                            label="Predicted Atomospheric Pressure",
                            value=f"{round(int(lstmdf.at[chosen, 'pressure'])/101325, ndigits=5)} atm"
                        )
                
        with pred_right:
            st.metric(
                        label="Predicted UV Index",
                        value=f"{lstmdf.at[chosen, 'local_uv_irradiance_index']}"
                    )
            
            st.metric("Overall MAE", "0.04665", 
                      help="The overall Mean Average Error made by the LSTM model for the testing data predictions.")
        
#Sidebar features
with st.sidebar:
    st.radio(
        "Temperature Unit of Measurement",
        ("Celsius", "Fahrenheit", "Kelvin"),
        on_change=change_date,
        key = "temp_unit"
    )

with st.sidebar:
    st.radio(
        "Pressure Unit of Measurement",
        ("Pascal (Pa)", "Millibar (mbar)", "Standard atmosphere (atm)"),
        on_change=change_date,
        key = "pressure_unit"
    )