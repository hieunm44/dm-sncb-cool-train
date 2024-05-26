from datetime import datetime
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, ctx, callback, dash_table
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
mapbox_access_token = ''

    
def create_time_series(data, feature, hovered_point, preds):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
                    x=data['timestamps_UTC'],
                    y=data[feature],
                    mode="lines+markers",
                    marker={"color": "navy", "size": 10},
                    line={"color": "deepskyblue"},
                    selectedpoints=hovered_point,
                    unselected={"marker": {"color": "deepskyblue", "size": 5},})
    )

    anomaly_ids = np.where(preds==1)[0]
    anomalies = data.iloc[anomaly_ids]
    
    fig.add_trace(go.Scatter(
                    x=anomalies['timestamps_UTC'],
                    y=anomalies[feature],
                    mode="markers",
                    marker={"color": "red", "size": 10},
                    customdata=preds
                    )
                )

    fig.update_layout(
        # xaxis_title='timestamp',
        yaxis_title=feature,
        height=150,
        margin={'l': 10, 'r': 10, 't': 20, 'b': 0},
        font={'size': 10},
        showlegend=False)

    return fig


def create_map(data, hovered_point, preds):
    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
                    lat=data['lat'],
                    lon=data['lon'],
                    mode='markers',
                    marker=go.scattermapbox.Marker(color='navy', size=10),
                    selectedpoints=hovered_point,
                    unselected={"marker": {"color": "deepskyblue", "size": 5}},
                    ))
    
    anomaly_ids = np.where(preds==1)[0]
    anomalies = data.iloc[anomaly_ids]

    fig.add_trace(go.Scattermapbox(
                    lat=anomalies['lat'],
                    lon=anomalies['lon'],
                    mode='markers',
                    marker=go.scattermapbox.Marker(color='red', size=10),
                    # customdata=preds
                    ))

    fig.update_layout(
        mapbox=dict(accesstoken=mapbox_access_token,
                    center=dict(
                        lat=data['lat'].iloc[hovered_point[0]],
                        lon=data['lon'].iloc[hovered_point[0]]
                    ),
                    zoom=9),
        height=800,
        margin={'l': 10, 'r': 10, 't': 10, 'b':10},
        showlegend=False
    )

    return fig


def create_time_card(data, index):
    card = dbc.Card([dbc.CardHeader(html.H1('Time')), dbc.CardBody(html.H2(f"{data['timestamps_UTC'][index]}"))], color='maroon', inverse=True)
    
    return card


def create_weather_card(data, index):
    card = dbc.Card([dbc.CardHeader(html.H1('Weather')), dbc.CardBody(html.H2(f"{weather_code[data['weather_code'][index]]}"))], color='mediumslateblue', inverse=True)
    
    return card


def create_temp_card(data, index):
    card = dbc.Card([dbc.CardHeader(html.H1('Temperature')), dbc.CardBody(html.H2(f"{np.round(data['temperature'][index], 2)}°C"))], color='darkorange', inverse=True)
    
    return card


def create_humid_card(data, index):
    card = dbc.Card([dbc.CardHeader(html.H1('Humidity')), dbc.CardBody(html.H2(f"{np.round(data['humidity'][index], 2)}%"))], color='seagreen', inverse=True)
    
    return card


def create_wind_card(data, index):
    card = dbc.Card([dbc.CardHeader(html.H1('Wind Speed')), dbc.CardBody(html.H2(f"{np.round(data['wind_speed'][index], 2)} km/h"))], color='lightslategray', inverse=True)
    
    return card


def create_rain_card(data, index):
    card = dbc.Card([dbc.CardHeader(html.H1('Rain')), dbc.CardBody(html.H2(f"{data['rain'][index]} mm"))], color='cadetblue', inverse=True)
    
    return card


def create_snow_card(data, index):
    card = dbc.Card([dbc.CardHeader(html.H1('Snow')), dbc.CardBody(html.H2(f"{data['snow_depth'][index]} m"))], color='pink', inverse=True)
    
    return card


def create_cloud_card(data, index):
    card = dbc.Card([dbc.CardHeader(html.H1('Cloud Cover')), dbc.CardBody(html.H2(f"{np.round(data['cloud_cover'][index], 2)}%"))], color='skyblue', inverse=True)
    
    return card


def create_evapo_card(data, index):
    card = dbc.Card([dbc.CardHeader(html.H1('Evapotranspiration')), dbc.CardBody(html.H2(f"{np.round(data['evapotranspiration'][index], 2)} mm"))], color='tan', inverse=True)
    
    return card


def create_status_table(data, features, index, across_feature_preds):
    status_list = ['Normal' if i==0 else 'Abnormal' for i in across_feature_preds] + list(data.loc[index, features[-2:]])
    note_list = []
    anm = 0
    for i, feat in enumerate(features):
        value = data.loc[index, feat]
        if 'AirTemp' in feat and across_feature_preds[i] == 1 and value > 65:
            note_list.append('High air temperature')
            final_note = f'Problem in {feat[-3:]} Air Temperature'
            anm = 1
        elif 'WatTemp' in feat and across_feature_preds[i] == 1 and value > 100:
            note_list.append('High water temperature')
            final_note = f'Problem in {feat[-3:]} Water Temperature'
            anm = 1
        elif 'OilTemp' in feat and across_feature_preds[i] == 1 and value > 115:
            note_list.append('High water temperature')
            final_note = f'Problem in {feat[-3:]} Oil Temperature'
            anm = 1
        elif 'OilPress' in feat and across_feature_preds[i] == 1 and value > 500:
            note_list.append('High oil pressure')
            final_note = f'Problem in {feat[-3:]} Oil Pressure'
            anm = 1
        elif 'RPM' in feat and value !=0:
            note_list.append('Train is running')
            if anm==1:
                final_status = 'Anomaly detected'
                final_note = 'Engine problem'
        elif 'RPM' in feat and value == 0:
            note_list.append('Train is parked at the station')
            final_status = 'Normal'
            final_note = 'Normal'    
        elif value == 0:
            note_list.append('Exactly 0 value')
            final_note = f'Problem in sensor'
        elif value > 10000:
            note_list.append('Noise')
            final_status = 'Noise detected'
            final_note = f'Noise in sensor'
        else:    
            note_list.append('Normal')
            final_status = 'Normal'
            final_note = 'Normal'

    
    status_list.append(final_status)
    note_list.append(final_note)

    table_df = {'Feature': features+['Overall'], 'Status': status_list, 'Note': note_list}
    df = pd.DataFrame(table_df)
    table = dash_table.DataTable(
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True} for i in df.columns
        ],
        data=df.to_dict('records'),
        style_cell={'textAlign': 'left', 'font-size': 20},
    )

    return table


@callback(
    Output('date-range', 'start_date'), Output('date-range', 'end_date'), Output('g1', 'figure'), Output('g2', 'figure'), Output('g3', 'figure'), Output('g4', 'figure'), Output('g5', 'figure'),
    Output('g6', 'figure'), Output('g7', 'figure'), Output('g8', 'figure'), Output('g9', 'figure'), Output('g10', 'figure'),
    Output('g11', 'figure'), Output('time', 'children'), Output('weather', 'children'), Output('temp', 'children'), Output('humid', 'children'),
    Output('wind', 'children'), Output('rain', 'children'), Output('snow', 'children'), Output('cloud', 'children'), Output('evapo', 'children'), Output('status-table', 'children'),
    Input('veh-id', 'value'), Input('date-range', 'start_date'), Input('date-range', 'end_date'), Input('method', 'value'),
    Input('g1', 'hoverData'), Input('g2', 'hoverData'), Input('g3', 'hoverData'), Input('g4', 'hoverData'), Input('g5', 'hoverData'),
    Input('g6', 'hoverData'), Input('g7', 'hoverData'), Input('g8', 'hoverData'), Input('g9', 'hoverData'), Input('g10', 'hoverData'), Input('g11', 'hoverData'),
    State('g1', 'figure'), State('g2', 'figure'), State('g3', 'figure'), State('g4', 'figure'), State('g5', 'figure'),
    State('g6', 'figure'), State('g7', 'figure'), State('g8', 'figure'), State('g9', 'figure'), State('g10', 'figure'), State('g11', 'figure'),
    State('g1', 'relayoutData'), State('g2', 'relayoutData'), State('g3', 'relayoutData'), State('g4', 'relayoutData'), State('g5', 'relayoutData'),
    State('g6', 'relayoutData'), State('g7', 'relayoutData'), State('g8', 'relayoutData'), State('g9', 'relayoutData'), State('g10', 'relayoutData'), State('g11', 'relayoutData')
)
def update_when_hover(veh_id, start_date, end_date, method, hovered_data_1, hovered_data_2, hovered_data_3, hovered_data_4, hovered_data_5,
                      hovered_data_6, hovered_data_7, hovered_data_8, hovered_data_9, hovered_data_10, hovered_data_11,
                      fig_state_1, fig_state_2, fig_state_3, fig_state_4, fig_state_5, fig_state_6, fig_state_7, fig_state_8, fig_state_9, fig_state_10, fig_state_11,
                      layout_1, layout_2, layout_3, layout_4, layout_5, layout_6, layout_7, layout_8, layout_9, layout_10, layout_11):
    hovered_index_1 = hovered_data_1['points'][0]['pointIndex']
    hovered_index_2 = hovered_data_2['points'][0]['pointIndex']
    hovered_index_3 = hovered_data_3['points'][0]['pointIndex']
    hovered_index_4 = hovered_data_4['points'][0]['pointIndex']
    hovered_index_5 = hovered_data_5['points'][0]['pointIndex']
    hovered_index_6 = hovered_data_6['points'][0]['pointIndex']
    hovered_index_7 = hovered_data_7['points'][0]['pointIndex']
    hovered_index_8 = hovered_data_8['points'][0]['pointIndex']
    hovered_index_9 = hovered_data_9['points'][0]['pointIndex']
    hovered_index_10 = hovered_data_10['points'][0]['pointIndex']
    hovered_index_11 = hovered_data_11['points'][0]['pointIndex']

    hovered_index_list = [hovered_index_1, hovered_index_2, hovered_index_3, hovered_index_4, hovered_index_5, hovered_index_6, hovered_index_7, hovered_index_8, hovered_index_9, hovered_index_10, hovered_index_11]
    fig_state_list = [fig_state_1, fig_state_2, fig_state_3, fig_state_4, fig_state_5, fig_state_6, fig_state_7, fig_state_8, fig_state_9, fig_state_10, fig_state_11]

    changed_id = ctx.triggered_id
    features = ['RS_E_InAirTemp_PC1', 'RS_E_InAirTemp_PC2', 'RS_E_OilPress_PC1', 'RS_E_OilPress_PC2', 'RS_E_WatTemp_PC1', 'RS_E_WatTemp_PC2', 'RS_T_OilTemp_PC1', 'RS_T_OilTemp_PC2', 'RS_E_RPM_PC1', 'RS_E_RPM_PC2']    

    veh_ids = np.where(data['mapped_veh_id']==veh_id)[0]
    sub_data_by_veh = data.iloc[veh_ids]
    sub_data_by_veh.reset_index(drop=True, inplace=True)


    if changed_id == 'veh-id':
        start_date_new = str(datetime.strptime(sub_data_by_veh['timestamps_UTC'].min(), date_format).date())
        end_date_new = str(datetime.strptime(sub_data_by_veh['timestamps_UTC'].max(), date_format).date())
        min_date = start_date_new
        max_date = end_date_new
        start_date_id = np.where(sub_data_by_veh['timestamps_UTC'].apply(lambda x: x[:10]) == start_date_new)[0][0]
        end_date_id = np.where(sub_data_by_veh['timestamps_UTC'].apply(lambda x: x[:10]) == end_date_new)[0][-1]
        sub_data = sub_data_by_veh.iloc[start_date_id:end_date_id+1]
        sub_data.reset_index(drop=True, inplace=True)
        hovered_point = 0

        pred_list = [np.load(f'preds/preds_{methods[method]}_{feat}.npy')[-300000:] for feat in features[:8]]
        sub_pred_list = []
        for pred in pred_list:
            sub_pred_by_veh = pred[veh_ids]
            # print(sub_pred_by_veh.shape())
            sub_pred = sub_pred_by_veh[start_date_id:end_date_id-1]
            sub_pred_list.append(sub_pred)

        all_sub_pred = np.array([0]*len(sub_pred_list[0]))
        for sub_pred in sub_pred_list:
            all_sub_pred = all_sub_pred | sub_pred
        
        across_feature_preds = [sp[hovered_point] for sp in sub_pred_list]

        fig_1 = create_time_series(sub_data, features[0], [hovered_point], sub_pred_list[0])
        fig_2 = create_time_series(sub_data, features[1], [hovered_point], sub_pred_list[1])
        fig_3 = create_time_series(sub_data, features[2], [hovered_point], sub_pred_list[2])
        fig_4 = create_time_series(sub_data, features[3], [hovered_point], sub_pred_list[3])
        fig_5 = create_time_series(sub_data, features[4], [hovered_point], sub_pred_list[4])
        fig_6 = create_time_series(sub_data, features[5], [hovered_point], sub_pred_list[5])
        fig_7 = create_time_series(sub_data, features[6], [hovered_point], sub_pred_list[6])
        fig_8 = create_time_series(sub_data, features[7], [hovered_point], sub_pred_list[7])
        fig_9 = create_time_series(sub_data, features[8], [hovered_point], [])
        fig_10 = create_time_series(sub_data, features[9], [hovered_point], [])
        fig_11 = create_map(sub_data, [hovered_point], all_sub_pred)

        fig_list = [fig_1, fig_2, fig_3, fig_4, fig_5, fig_6, fig_7, fig_8, fig_9, fig_10, fig_11]

    else:
        start_date_id = np.where(sub_data_by_veh['timestamps_UTC'].apply(lambda x: x[:10]) == start_date)[0][0]
        end_date_id = np.where(sub_data_by_veh['timestamps_UTC'].apply(lambda x: x[:10]) == end_date)[0][-1]
        sub_data = sub_data_by_veh.iloc[start_date_id:end_date_id+1]
        sub_data.reset_index(drop=True, inplace=True)


    if changed_id is None:
        hovered_point = 0
        sub_data = ini_sub_data
        min_date = ini_min_date
        max_date = ini_max_date
        pred_list = [np.load(f'preds/preds_{methods[method]}_{feat}.npy')[-300000:] for feat in features[:8]]
        sub_pred_list = []
        for pred in pred_list:
            sub_pred_by_veh = pred[veh_ids]
            # print(sub_pred_by_veh.shape())
            sub_pred = sub_pred_by_veh[start_date_id:end_date_id-1]
            sub_pred_list.append(sub_pred)

        all_sub_pred = np.array([0]*len(sub_pred_list[0]))
        for sub_pred in sub_pred_list:
            all_sub_pred = all_sub_pred | sub_pred

        across_feature_preds = [sub_pred[hovered_point] for sub_pred in sub_pred_list]
        # status_table = create_status_table(sub_data, features, hovered_point, across_feature_preds)

        fig_1 = create_time_series(sub_data, features[0], [hovered_point], sub_pred_list[0])
        fig_2 = create_time_series(sub_data, features[1], [hovered_point], sub_pred_list[1])
        fig_3 = create_time_series(sub_data, features[2], [hovered_point], sub_pred_list[2])
        fig_4 = create_time_series(sub_data, features[3], [hovered_point], sub_pred_list[3])
        fig_5 = create_time_series(sub_data, features[4], [hovered_point], sub_pred_list[4])
        fig_6 = create_time_series(sub_data, features[5], [hovered_point], sub_pred_list[5])
        fig_7 = create_time_series(sub_data, features[6], [hovered_point], sub_pred_list[6])
        fig_8 = create_time_series(sub_data, features[7], [hovered_point], sub_pred_list[7])
        fig_9 = create_time_series(sub_data, features[8], [hovered_point], [])
        fig_10 = create_time_series(sub_data, features[9], [hovered_point], [])
        fig_11 = create_map(sub_data, [hovered_point], all_sub_pred)

        fig_list = [fig_1, fig_2, fig_3, fig_4, fig_5, fig_6, fig_7, fig_8, fig_9, fig_10, fig_11]


    if changed_id in ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g11']:
        # print(hovered_index_list)
        hovered_point = hovered_index_list[int(changed_id[1:])-1]
        min_date = start_date
        max_date = end_date
        # pred_list = [np.load(f'preds/preds_{methods[method]}_{feat}.npy') for feat in features[:8]]

        fig_list = []
        for fig_state in fig_state_list:
            fig_state['data'][0]['selectedpoints'] = [hovered_point]
            fig_state['data'][1]['selectedpoints'] = [hovered_point]
            fig = go.Figure(fig_state)
            fig_list.append(fig)
        
        sub_pred_list = [fs['data'][1]['customdata'] for fs in fig_state_list[:8]]
        across_feature_preds = [sp[hovered_point] for sp in sub_pred_list]
        
    if changed_id == 'date-range':
        min_date = start_date
        max_date = end_date
        start_date_id = np.where(sub_data_by_veh['timestamps_UTC'].apply(lambda x: x[:10]) == min_date)[0][0]
        end_date_id = np.where(sub_data_by_veh['timestamps_UTC'].apply(lambda x: x[:10]) == max_date)[0][-1]
        hovered_point = 0
        
        pred_list = [np.load(f'preds/preds_{methods[method]}_{feat}.npy')[-300000:] for feat in features[:8]]
        sub_pred_list = []
        for pred in pred_list:
            sub_pred_by_veh = pred[veh_ids]
            # print(sub_pred_by_veh.shape())
            sub_pred = sub_pred_by_veh[start_date_id:end_date_id-1]
            sub_pred_list.append(sub_pred)

        all_sub_pred = np.array([0]*len(sub_pred_list[0]))
        for sub_pred in sub_pred_list:
            all_sub_pred = all_sub_pred | sub_pred
        
        across_feature_preds = [sp[hovered_point] for sp in sub_pred_list]

        fig_1 = create_time_series(sub_data, features[0], [hovered_point], sub_pred_list[0])
        fig_2 = create_time_series(sub_data, features[1], [hovered_point], sub_pred_list[1])
        fig_3 = create_time_series(sub_data, features[2], [hovered_point], sub_pred_list[2])
        fig_4 = create_time_series(sub_data, features[3], [hovered_point], sub_pred_list[3])
        fig_5 = create_time_series(sub_data, features[4], [hovered_point], sub_pred_list[4])
        fig_6 = create_time_series(sub_data, features[5], [hovered_point], sub_pred_list[5])
        fig_7 = create_time_series(sub_data, features[6], [hovered_point], sub_pred_list[6])
        fig_8 = create_time_series(sub_data, features[7], [hovered_point], sub_pred_list[7])
        fig_9 = create_time_series(sub_data, features[8], [hovered_point], [])
        fig_10 = create_time_series(sub_data, features[9], [hovered_point], [])
        fig_11 = create_map(sub_data, [hovered_point], all_sub_pred)

        fig_list = [fig_1, fig_2, fig_3, fig_4, fig_5, fig_6, fig_7, fig_8, fig_9, fig_10, fig_11]

    if changed_id == 'method':
        min_date = start_date
        max_date = end_date
        hovered_point = 0
        pred_list = [np.load(f'preds/preds_{methods[method]}_{feat}.npy')[-300000:] for feat in features[:8]]
        sub_pred_list = []
        for pred in pred_list:
            sub_pred_by_veh = pred[veh_ids]
            # print(sub_pred_by_veh.shape())
            sub_pred = sub_pred_by_veh[start_date_id:end_date_id-1]
            sub_pred_list.append(sub_pred)

        all_sub_pred = np.array([0]*len(sub_pred_list[0]))
        for sub_pred in sub_pred_list:
            all_sub_pred = all_sub_pred | sub_pred
        
        across_feature_preds = [sp[hovered_point] for sp in sub_pred_list]

        fig_1 = create_time_series(sub_data, features[0], [hovered_point], sub_pred_list[0])
        fig_2 = create_time_series(sub_data, features[1], [hovered_point], sub_pred_list[1])
        fig_3 = create_time_series(sub_data, features[2], [hovered_point], sub_pred_list[2])
        fig_4 = create_time_series(sub_data, features[3], [hovered_point], sub_pred_list[3])
        fig_5 = create_time_series(sub_data, features[4], [hovered_point], sub_pred_list[4])
        fig_6 = create_time_series(sub_data, features[5], [hovered_point], sub_pred_list[5])
        fig_7 = create_time_series(sub_data, features[6], [hovered_point], sub_pred_list[6])
        fig_8 = create_time_series(sub_data, features[7], [hovered_point], sub_pred_list[7])
        fig_9 = create_time_series(sub_data, features[8], [hovered_point], [])
        fig_10 = create_time_series(sub_data, features[9], [hovered_point], [])
        fig_11 = create_map(sub_data, [hovered_point], all_sub_pred)

        fig_list = [fig_1, fig_2, fig_3, fig_4, fig_5, fig_6, fig_7, fig_8, fig_9, fig_10, fig_11]


    time_card = create_time_card(sub_data, hovered_point)
    weather_card = create_weather_card(sub_data, hovered_point)
    temp_card = create_temp_card(sub_data, hovered_point)
    humid_card = create_humid_card(sub_data, hovered_point)
    wind_card = create_wind_card(sub_data, hovered_point)
    rain_card = create_rain_card(sub_data, hovered_point)
    snow_card = create_snow_card(sub_data, hovered_point)
    cloud_card = create_cloud_card(sub_data, hovered_point)
    evapo_card = create_evapo_card(sub_data, hovered_point)
    status_table = create_status_table(sub_data, features, hovered_point, across_feature_preds)


    layout_list = [layout_1, layout_2, layout_3, layout_4, layout_5, layout_6, layout_7, layout_8, layout_9, layout_10]
    # fig_list = [fig_1, fig_2, fig_3, fig_4, fig_5, fig_6, fig_7, fig_8, fig_9, fig_10]

    for layout, fig in zip(layout_list, fig_list):
        if layout and 'xaxis.range[0]' in layout:
            fig['layout']['xaxis']['range'] = [layout['xaxis.range[0]'], layout['xaxis.range[1]']] 
        if layout and 'yaxis.range[0]' in layout:
            fig['layout']['yaxis']['range'] = [layout['yaxis.range[0]'], layout['yaxis.range[1]']] 

    if layout_11 and 'mapbox.zoom' in layout_11:
        fig_state_11['layout']['mapbox']['zoom'] =  layout_11['mapbox.zoom']
    if layout_11 and 'mapbox.zoom' in layout_11:
        fig_state_11['layout']['mapbox']['center'] = layout_11['mapbox.center']

    return [min_date, max_date] + fig_list + [time_card, weather_card, temp_card, humid_card, wind_card, rain_card, snow_card, cloud_card, evapo_card, status_table]


if __name__ == '__main__':
    weather_code = {0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast', 45: 'Fog', 48: 'Depositing rime fog', 51: 'Light drizzle', 53: 'Moderate drizzle', 55: 'Dense drizzle',
                56: 'Light freezing drizzle', 57: 'Dense freezing drizzle', 61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain', 66: 'Light freezing rain', 67: 'Heavy freezing rain',
                71: 'Slight snow fall', 73: 'Moderate snow fall', 75: 'Heavy snow fall', 77: 'Snow grains', 80: 'Slight rain showers', 81: 'Moderate rain showers', 82: 'Violent rain showers',
                85: 'Slight snow showers', 86: 'Heavy snow showers', 95: 'Slight/moderate thunderstorm', 96: 'Thunderstorm with slight hail', 99: 'Thunderstorm with heavy hail'}

    app = Dash(__name__, external_stylesheets=[
        'https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP
    ])


    data = pd.read_csv('data/sample_data.csv', sep=';', index_col=[0])
    data = data.iloc[-30000:]
    veh_id_list = np.sort(data['mapped_veh_id'].unique())
    ini_veh_ids = np.where(data['mapped_veh_id']==veh_id_list[0])[0]
    ini_sub_data = data.iloc[ini_veh_ids]
    ini_sub_data.reset_index(inplace=True)
    date_format = '%Y-%m-%d %H:%M:%S'
    ini_min_date = datetime.strptime(ini_sub_data['timestamps_UTC'].min(), date_format).date()
    ini_max_date = datetime.strptime(ini_sub_data['timestamps_UTC'].max(), date_format).date()

    methods = {'Ellip Envelope': 'ellip_envelope', 'Isolation Forest': 'iso_forest', 'Local Outlier Factor': 'local_outlier_factor', 'K-Means': 'kmeans'}


    app.layout = html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col(html.H1('Anomaly Detection System'), width=4, style={'margin': 1}),
                    dbc.Col(html.H3('Vehicle ID:'), width=1, style={'margin': 5, 'align': 'right'}),
                    dbc.Col(dcc.Dropdown(veh_id_list, veh_id_list[0], id='veh-id'), width=1, style={'font-size': 15}),
                    dbc.Col(width=1),
                    dbc.Col(html.H3('Date Range:'), width=1, style={'margin': 5, 'align': 'right'}),
                    dbc.Col(dcc.DatePickerRange(
                                id='date-range',
                                start_date=ini_min_date,
                                end_date=ini_max_date,
                                min_date_allowed=ini_min_date,
                                max_date_allowed=ini_max_date,
                                display_format='YYYY-MM-DD'
                                ), style={'font-size': 15})
                ]),

                dcc.Graph(id='g1', hoverData={'points': [{'pointIndex': None}]}, figure={'data': [{'selectedpoints': [0]}]}),
                dcc.Graph(id='g2', hoverData={'points': [{'pointIndex': None}]}, figure={'data': [{'selectedpoints': [0]}]}),
                dcc.Graph(id='g3', hoverData={'points': [{'pointIndex': None}]}, figure={'data': [{'selectedpoints': [0]}]}),
                dcc.Graph(id='g4', hoverData={'points': [{'pointIndex': None}]}, figure={'data': [{'selectedpoints': [0]}]}),
                dcc.Graph(id='g5', hoverData={'points': [{'pointIndex': None}]}, figure={'data': [{'selectedpoints': [0]}]}),
                dcc.Graph(id='g6', hoverData={'points': [{'pointIndex': None}]}, figure={'data': [{'selectedpoints': [0]}]}),
                dcc.Graph(id='g7', hoverData={'points': [{'pointIndex': None}]}, figure={'data': [{'selectedpoints': [0]}]}),
                dcc.Graph(id='g8', hoverData={'points': [{'pointIndex': None}]}, figure={'data': [{'selectedpoints': [0]}]}),
                dcc.Graph(id='g9', hoverData={'points': [{'pointIndex': None}]}, figure={'data': [{'selectedpoints': [0]}]}),
                dcc.Graph(id='g10', hoverData={'points': [{'pointIndex': None}]}, figure={'data': [{'selectedpoints': [0]}]}),
                ]),
            dbc.Col([
                dbc.Row([
                    dbc.Col(),
                    dbc.Col(html.H3('Method:'), width=1, style={'margin': 5}),
                    dbc.Col(dcc.Dropdown(list(methods.keys()), list(methods.keys())[0], id='method'), width=2, style={'font-size': 15})
                    ], className='g-0'),
                html.Div([
                    dcc.Graph(id='g11', hoverData={'points': [{'pointIndex': None}]}, figure={'data': [{'selectedpoints': [0]}]}),
                    # html.Pre(id='d1', style=styles['pre']),
                    # html.Pre(id='d2', style=styles['pre']),
                ]),
                dbc.Row([
                    dbc.Col(dbc.Card(id='time')),
                    dbc.Col(dbc.Card(id='weather')),
                    dbc.Col(dbc.Card(id='temp')),
                    dbc.Col(dbc.Card(id='humid')),
                ], style={'margin': 5}),
                dbc.Row([
                    dbc.Col(dbc.Card(id='wind')),
                    dbc.Col(dbc.Card(id='rain')),
                    dbc.Col(dbc.Card(id='snow')),
                    dbc.Col(dbc.Card(id='cloud')),
                    dbc.Col(dbc.Card(id='evapo')),
                    
                ], style={'margin': 5}),
                dbc.Col([html.Div(id='status-table')], style={'margin': 10})
        ])
        ])
    ])

    app.run(debug=True)