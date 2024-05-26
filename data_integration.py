import pandas as pd
import numpy as np

import openmeteo_requests
import requests_cache
from retry_requests import retry



def preprocess_timestamp(raw_data): # to reduce number of diffrent timestamps
    raw_data.sort_values(by='timestamps_UTC', inplace=True)
    raw_data.reset_index(drop=True, inplace=True)

    timestamp = raw_data['timestamps_UTC']
    lat = raw_data['lat']
    lon = raw_data['lon']

    short_timestamp = pd.Series(map(lambda x: x[:10], timestamp), name='short_timestamp')
    short_lat = pd.Series(map(lambda x: np.round(x, decimals=1), lat), name='short_lat')
    short_lon = pd.Series(map(lambda x: np.round(x, decimals=1), lon), name='short_lon')
    short_time_location_data = pd.concat([short_timestamp, short_lat, short_lon], axis=1)

    return short_time_location_data


def get_weather_data(short_time_location_data):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"

    weather_data = pd.DataFrame(columns=['timestamp', 'lat', 'lon', 'temperature', 'humidity', 'rain', 'snow_depth', 'weather_code', 'cloud_cover', 'evapotranspiration', 'wind_speed'])

    for _, row in short_time_location_data.iterrows():
        params = {
            "latitude": row['short_lat'],
            "longitude": row['short_lon'],
            "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "snow_depth", "weather_code", "cloud_cover", "et0_fao_evapotranspiration", "wind_speed_10m"],
            "start_date": row['short_timestamp'],
            "end_date": row['short_timestamp']
        }

        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        hourly = response.Hourly()

        temp_dict = {'short_timestamp': row['short_timestamp'],
                    'short_lat': row['short_lat'],
                    'short_lon': row['short_lon'],
                    'temperature': hourly.Variables(0).ValuesAsNumpy(),
                    'humidity': hourly.Variables(1).ValuesAsNumpy(),
                    'rain': hourly.Variables(2).ValuesAsNumpy(),
                    'snow_depth': hourly.Variables(3).ValuesAsNumpy(),
                    'weather_code': hourly.Variables(4).ValuesAsNumpy(),
                    'cloud_cover': hourly.Variables(5).ValuesAsNumpy(),
                    'evapotranspiration': hourly.Variables(6).ValuesAsNumpy(),
                    'wind_speed': hourly.Variables(7).ValuesAsNumpy()}

        temp_df = pd.DataFrame(data=temp_dict)
        weather_data = pd.concat([weather_data, temp_df], axis=0)

    weather_data.to_csv('data/weather_data.csv', sep=';')

    return weather_data


def integrate_data(raw_data):
    raw_data.sort_values(by='timestamps_UTC', inplace=True)
    raw_data.reset_index(drop=True, inplace=True)

    timestamp = raw_data['timestamps_UTC']
    lat = raw_data['lat']
    lon = raw_data['lon']

    # short_timestamp = pd.Series(map(lambda x: x[:14]+'00:00', timestamp), name='short_timestamp')
    short_timestamp = pd.Series(map(lambda x: x[:10], timestamp), name='short_timestamp')
    short_lat = pd.Series(map(lambda x: np.round(x, decimals=1), lat), name='short_lat')
    short_lon = pd.Series(map(lambda x: np.round(x, decimals=1), lon), name='short_lon')
    short_time_location_data = pd.concat([short_timestamp, short_lat, short_lon], axis=1)
    print(short_time_location_data.columns)
    temp_data = pd.concat([raw_data, short_time_location_data], axis=1)

    weather_data = get_weather_data(short_time_location_data)

    merged_data = temp_data.merge(weather_data, how = 'inner', on=['short_timestamp', 'short_lat', 'short_lon'])
    merged_data.drop(columns=['short_timestamp', 'short_lat', 'short_lon'], inplace=True)
    merged_data.to_csv('data/merged_data.csv', sep=';')


if __name__ == '__main__':
   raw_data = pd.read_csv('data/ar41_for_ulb_mini.csv', sep=';') # len = 17679273
   integrate_data(raw_data)