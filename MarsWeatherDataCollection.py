import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

response = requests.get(
    "https://api.parse.bot/scraper/29bc9970-6740-474d-aad0-d77e3d8e4483/get_mars_weather",
    headers={"X-API-Key": os.getenv("MarsAPI")},
    params={"rover": "curiosity"},
)

data = response.json()

with open('xxxMarsWeatherData.json', 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)