import streamlit as st
import json
from datetime import date
import pandas as pd
from pathlib import Path

d = pd.read_json('data/MarsWeatherData.json')
filtered_df = d["data"]["soles"]
df = pd.DataFrame(filtered_df)
df = df.set_index("terrestrial_date")
df.rename(columns={"min_temp": "Min Temperature", "max_temp": "Max Temperature",
                   "pressure": "Pressure", "atmo_opacity":"Atmosphere Opacity",
                   "local_uv_irradiance_index": "UV Irradiance Index",
                   "min_gts_temp": "Min Ground Temperature",
                   "max_gts_temp": "Max Ground Temperature",
                   "sunrise": "Sunrise", "sunset": "Sunset"}, inplace=True)

most_recent_date = df.index[0]
min_date = df.index[-1]

def fill_data(chosen=most_recent_date):
    var = ["Min Temperature", "Sunrise", "Max Temperature", "Sunset", 
             "Min Ground Temperature", "UV Irradiance Index", "Max Ground Temperature",
             "Atmosphere Opacity", "Pressure"]

    left2, right2 = weather_data.columns(2)
    #This is for when the requested day does not exist
    if chosen not in df.index:
        track = 0
        for v in var:
            if track%2 == 0:
                left2.container(border=True).write(f"{v}: -- ")
            else:
                right2.container(border=True).write(f"{v}: -- ")
            track += 1

    else:
        if st.session_state.temp_unit == "Celsius":
            left2.container(border=True).write(f"{var[0]}: {df.at[chosen, var[0]]}°C")
            right2.container(border=True).write(f"{var[1]}: {df.at[chosen, var[1]]}")
            left2.container(border=True).write(f"{var[2]}: {df.at[chosen, var[2]]}°C")
            right2.container(border=True).write(f"{var[3]}: {df.at[chosen, var[3]]}")
            left2.container(border=True).write(f"{var[4]}: {df.at[chosen, var[4]]}°C")
            right2.container(border=True).write(f"{var[5]}: {df.at[chosen, var[5]]}")
            left2.container(border=True).write(f"{var[6]}: {df.at[chosen, var[6]]}°C")
            right2.container(border=True).write(f"{var[7]}: {df.at[chosen, var[7]]}")
            #for the pressure
            if df.at[chosen, var[8]] == "--":
                left2.container(border=True).write(f"{var[8]}: --")
            elif st.session_state.pressure_unit == "Pascal (Pa)":
                left2.container(border=True).write(f"{var[8]}: {df.at[chosen, var[8]]} Pa")
            elif  st.session_state.pressure_unit == "Millibar (mbar)":
                left2.container(border=True).write(f"{var[8]}: {round(int(df.at[chosen, var[8]])/100)} mbar")
            elif st.session_state.pressure_unit == "Standard atmosphere (atm)":
                left2.container(border=True).write(f"{var[8]}: {round(int(df.at[chosen, var[8]])/101325, ndigits=5)} atm")

        elif st.session_state.temp_unit == "Fahrenheit":
            if df.at[chosen, var[0]] != "--":
                left2.container(border=True).write(f"{var[0]}: {round(int(df.at[chosen, var[0]]) * 1.8 + 32)}°F")
            else:
                left2.container(border=True).write(f"{var[0]}: --°F")
            right2.container(border=True).write(f"{var[1]}: {df.at[chosen, var[1]]}")
            if df.at[chosen, var[2]] != "--":
                left2.container(border=True).write(f"{var[2]}: {round(int(df.at[chosen, var[2]]) * 1.8 + 32)}°F")
            else:
                left2.container(border=True).write(f"{var[2]}: --°F")
            right2.container(border=True).write(f"{var[3]}: {df.at[chosen, var[3]]}")
            if df.at[chosen, var[4]] != "--":
                left2.container(border=True).write(f"{var[4]}: {round(int(df.at[chosen, var[4]]) * 1.8 + 32)}°F")
            else:
                left2.container(border=True).write(f"{var[4]}: --°F")
            right2.container(border=True).write(f"{var[5]}: {df.at[chosen, var[5]]}")
            if df.at[chosen, var[6]] != "--":
                left2.container(border=True).write(f"{var[6]}: {round(int(df.at[chosen, var[6]]) * 1.8 + 32)}°F")
            else:
                left2.container(border=True).write(f"{var[6]}: --°F")
            right2.container(border=True).write(f"{var[7]}: {df.at[chosen, var[7]]}")
            #for the pressure
            if df.at[chosen, var[8]] == "--":
                left2.container(border=True).write(f"{var[8]}: --")
            elif st.session_state.pressure_unit == "Pascal (Pa)":
                left2.container(border=True).write(f"{var[8]}: {df.at[chosen, var[8]]} Pa")
            elif  st.session_state.pressure_unit == "Millibar (mbar)":
                left2.container(border=True).write(f"{var[8]}: {round(int(df.at[chosen, var[8]])/100)} mbar")
            elif st.session_state.pressure_unit == "Standard atmosphere (atm)":
                left2.container(border=True).write(f"{var[8]}: {round(int(df.at[chosen, var[8]])/101325, ndigits=5)} atm")

        elif st.session_state.temp_unit == "Kelvin":
            if df.at[chosen, var[0]] != "--":
                left2.container(border=True).write(f"{var[0]}: {round(int(df.at[chosen, var[0]]) + 273.15)}°K")
            else:
                left2.container(border=True).write(f"{var[0]}: --°K")
            right2.container(border=True).write(f"{var[1]}: {df.at[chosen, var[1]]}")
            if df.at[chosen, var[2]] != "--":
                left2.container(border=True).write(f"{var[2]}: {round(int(df.at[chosen, var[2]]) + 273.15)}°K")
            else:
                left2.container(border=True).write(f"{var[2]}: --°K")
            right2.container(border=True).write(f"{var[3]}: {df.at[chosen, var[3]]}")
            if df.at[chosen, var[4]] != "--":
                left2.container(border=True).write(f"{var[4]}: {round(int(df.at[chosen, var[4]]) + 273.15)}°K")
            else:
                left2.container(border=True).write(f"{var[4]}: --°K")
            right2.container(border=True).write(f"{var[5]}: {df.at[chosen, var[5]]}")
            if df.at[chosen, var[6]] != "--":
                left2.container(border=True).write(f"{var[6]}: {round(int(df.at[chosen, var[6]]) + 273.15)}°K")
            else:
                left2.container(border=True).write(f"{var[6]}: --°K")
            right2.container(border=True).write(f"{var[7]}: {df.at[chosen, var[7]]}")
            #for the pressure
            if df.at[chosen, var[8]] == "--":
                left2.container(border=True).write(f"{var[8]}: --")
            elif st.session_state.pressure_unit == "Pascal (Pa)":
                left2.container(border=True).write(f"{var[8]}: {df.at[chosen, var[8]]} Pa")
            elif  st.session_state.pressure_unit == "Millibar (mbar)":
                left2.container(border=True).write(f"{var[8]}: {round(int(df.at[chosen, var[8]])/100)} mbar")
            elif st.session_state.pressure_unit == "Standard atmosphere (atm)":
                left2.container(border=True).write(f"{var[8]}: {round(int(df.at[chosen, var[8]])/101325, ndigits=5)} atm")

def change_date():
    chosen_date = st.session_state.my_date_picker
    date_str = chosen_date.strftime("%Y-%m-%d")
    fill_data(date_str)

with open("data/MarsWeatherData.json", "r") as file:
    data = json.load(file)

#This just centers the title
# l, m, r = st.columns([1, 4, 1])
# m.markdown("""# :orange[Mars Weather Forecast]""")
st.markdown("<h1 style='text-align: center; color: #921f02;'>Mars Weather Forecast</h1>", unsafe_allow_html=True)

image_path = Path(__file__).resolve().parent / "images" / "mars.png"

left, center, right= st.columns([1, 2, 1])

with center:
    st.image(
    
        str(image_path),
        caption="This self-portrait of NASA's Curiosity Mars rover shows the vehicle on Vera Rubin Ridge, which it's been investigating for the past several months.Directly behind the rover is the start of a clay-rich slope scientists are eager to begin exploring. In the coming week, Curiosity will begin to climb this slope. North is on the left and west is on the right, with Gale Crater's rim on the horizon of both edges.",
        width=500
)

left, middle, right = st.columns([2, 1, 2])
#middle.markdown("""###### :orange[Terrestrial Date]""")
middle.date_input(
    "Terrestrial Date", 
    value=date.fromisoformat(most_recent_date), 
    min_value=date.fromisoformat(min_date),
    max_value=date.fromisoformat(most_recent_date),
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
        sol.container().write(f"SOL: not recorded")
    else:
        s = df.at[date_str, "sol"]
        sol.container().write(f"SOL: {s}")

with ls.container():
    chosen_date = st.session_state.my_date_picker
    date_str = chosen_date.strftime("%Y-%m-%d")
    if date_str not in df.index:
        ls.container().write(f"LS: --")
    else:
        l = df.at[date_str, "ls"]
        ls.container().write(f"LS: {l}")

with weather_data.container():
    chosen_date = st.session_state.my_date_picker
    date_str = chosen_date.strftime("%Y-%m-%d")
    fill_data(date_str)
    
    
st.divider()
with st.container():
    st.subheader('Machine Model Temperature Predictions')
    
    pred_left, pred_right = st.columns(2)
    
    with pred_left:
        st.metric(
                     label="Predicted Min Temperature",
                     value="-- C"
                )
        st.metric(
                     label="Predicted Max Temperature",
                     value="-- C"
                )
        st.metric(
                     label="Predicted Min Ground Temperature",
                     value="-- C"
                )
        st.metric(
                     label="Predicted Max Ground Temperature",
                     value="-- C"
                )

        
    with pred_right:
        st.metric(
                     label="Predicted Atomospheric Pressure",
                     value="-- PA"
                )
        st.metric(
                     label="Predicted UV Index",
                     value="--"
                )
        
        st.metric("Prediction Error", "-- C")
            
        


#Sidebar features
with st.sidebar:
    st.radio(
        "Temperature Unit of Measurement",
        ("Celsius", "Fahrenheit", "Kelvin"),
        on_change=change_date,
        key = "temp_unit"
    )

#do I want to do this?"Pa", "mbar", "atm"
with st.sidebar:
    st.radio(
        "Pressure Unit of Measurement",
        ("Pascal (Pa)", "Millibar (mbar)", "Standard atmosphere (atm)"),
        on_change=change_date,
        key = "pressure_unit"
    )