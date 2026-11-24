import requests
import json
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Data is gotten from
# Mars Science Laboratory – Curiosity (REMS) – live feed: 
# https://mars.nasa.gov/rss/api/?feed=weather&category=msl&feedtype=json
# data provided by Centro de Astrobiologia
# Currently, there is a PHP error that is preventing data from being retrieved.
# "Your PHP installation appears to be missing the MySQL extension which is required by WordPress."
# This is due to a problem on there end?
# I guess they updated the PHP and didn't make sure they have the mysql extension?
response = requests.get(
    "https://api.parse.bot/scraper/29bc9970-6740-474d-aad0-d77e3d8e4483/get_mars_weather",
    headers={"X-API-Key": os.getenv("MarsAPI")},
    params={"rover": "curiosity"},
)

data = response.json()

data_path = Path(__file__).resolve().parent / "xxxMarsWeatherData.json"
#has testing file so other file doesn't get overwritten
with open(data_path, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)