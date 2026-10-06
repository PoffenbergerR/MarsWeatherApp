#TO DO: 
# Allow user to change units of measurement (also indicate what unit of measurement is being used), 
# make color/design look better, 
# insert descreptions for what things like SOL means, 
# add historical weather graph? Maybe with our machine learning prediciton comparison?, 
# add fun images showcasing the rover and Gale Crater

import streamlit as st
import json
from datetime import date
import pandas as pd

d = pd.read_json('MarsWeatherData.json')
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
    variables = ["Min Temperature", "Sunrise", "Max Temperature", "Sunset", 
             "Min Ground Temperature", "UV Irradiance Index", "Max Ground Temperature",
             "Atmosphere Opacity", "Pressure"]

    left2, right2 = weather_data.columns(2)
    #This is for when the requested day does not exist
    if chosen not in df.index:
        track = 0
        for v in variables:
            if track%2 == 0:
                left2.container(border=True).write(f"{v}: -- ")
            else:
                right2.container(border=True).write(f"{v}: -- ")
            track += 1

    else:
        track = 0
        for v in variables:
            if track%2 == 0:
                left2.container(border=True).write(f"{v}: {df.at[chosen, v]}")
            else:
                right2.container(border=True).write(f"{v}: {df.at[chosen, v]}")
            track += 1

def change_date():
    chosen_date = st.session_state.my_date_picker
    date_str = chosen_date.strftime("%Y-%m-%d")
    fill_data(date_str)

with open("MarsWeatherData.json", "r") as file:
    data = json.load(file)

#This just centers the title
# l, m, r = st.columns([1, 4, 1])
# m.markdown("""# :orange[Mars Weather Forecast]""")
st.markdown("<h1 style='text-align: center; color: #921f02;'>Mars Weather Forecast</h>", unsafe_allow_html=True)

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
