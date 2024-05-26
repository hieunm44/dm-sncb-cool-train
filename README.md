# Data Mining Project - SNCB Cool Train
<div align="center">
<a href="https://www.bruface.eu/">
<img src="https://www.bruface.eu/sites/default/files/BUA-BRUFACE_RGB_Update_DEF_1_0_0.jpg" height=100"/>
</a>
</div>

## Overview
This repo is our project "SNCB Cool Train" in the course "Data Mining" at Université Libre de Bruxelles (ULB).

## Built With
<div align="center">
<a href="https://open-meteo.com/">
   <img src="https://community-openhab-org.s3.dualstack.eu-central-1.amazonaws.com/original/3X/d/e/de6bed8f06b3e5a0ab03bb5d4369402988ec3d52.png" height=40 hspace=10/>
</a>
<a href="https://scikit-learn.org/stable/">
   <img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" height=40 hspace=10/>
</a>
<a href="https://plotly.com/">
   <img src="https://upload.wikimedia.org/wikipedia/commons/8/8a/Plotly-logo.png" height=40/>
</a>
</div>

## Setup
1. Here we only use a mini file `ar41_for_ulb_mini.csv`. You can get the full raw data file `ar41_for_ulb.csv` (1.9GB) provided by the lecturer.
2. Go to https://account.mapbox.com/access-tokens/ and create an access token, then put it at `mapbox_access_token=` in the file `app.py`.

## Usage
1. Clone the repo
   ```sh
   git clone https://github.com/hieunm44/dm-sncb-cool-train.git
   cd sdm-sncb-cool-train
   ```
2. Install necessary packages
   ```sh
   pip install -r requirements.txt
   ```
3. Get weather data from https://open-meteo.com/ through API calls then integrate to raw data.
   ```sh
   python3 data_integration.py
   ```
   Then two files `weather_data.csv` and `merged_data` will be created in the folder `data`.
4. Check the file `preprocess_data.ipynb` for data preprocessing and model training. Several data files and model files will be created.
6. Run the visualization app. Here we only use the file `sample_data.csv` (a portion of full data), so that the application can run smoothly.
   ```sh
   python3 app.py
   ```
<div align="center">
<img src="app_overview.png" width=100%/>
</div>
