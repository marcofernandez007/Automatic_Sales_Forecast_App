## import libraries
## we need to install folium, streamlit_folium, geopandas, etc..
import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.express as px
import folium
import geopandas as gpd
from streamlit_folium import folium_static
from shapely.geometry import Point, Polygon
st.set_page_config(layout='wide',page_title = "Map")
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

def map_heatmap_page():
    df = st.session_state.df
    skus = sorted(df['sku'].unique().tolist())

    # Load the world map data
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Filter the data to include only European countries
    europe = world[world['continent'] == 'Europe']
    europe = gpd.GeoDataFrame(pd.concat([europe, world[world['name'].isin(['Turkey', 'Cyprus', 'United Arab Emirates'])]]), crs=europe.crs)
    #add the missing countries
    malta_data = {'continent': 'Europe', 'geometry': [Polygon([
    (14.375, 35.85),
    (14.375, 35.95),
    (14.475, 35.95),
    (14.475, 35.85)
    ])]}  
    malta_row = gpd.GeoDataFrame(malta_data, geometry='geometry', crs=europe.crs)
    malta_row['name'] = 'Malta'
    monaco_data = {'continent': 'Europe', 'geometry': [Polygon([
    (7.416667, 43.733333),
    (7.425833, 43.745833),
    (7.408333, 43.751389),
    (7.408333, 43.745833)
    ])]}

# Create a GeoDataFrame for Monaco
    monaco_row = gpd.GeoDataFrame(monaco_data, geometry='geometry', crs=europe.crs)
    monaco_row['name'] = 'Monaco'

#europe = europe.append(malta_row, ignore_index=True)
    europe = gpd.GeoDataFrame(pd.concat([europe, malta_row, monaco_row]), crs=europe.crs).reset_index()
    europe.loc[2, 'geometry']= [Polygon([
    (6.18632, 49.463803),
    (6.65823, 49.201958),
    (8.099279, 49.017784),
    (7.593676, 48.333019),
    (7.466759, 47.620582),
    (7.192202, 47.449766),
    (6.736571, 47.541801),
    (6.768714, 47.287708),
    (6.037389, 46.725779),
    (6.022609, 46.27299),
    (6.5001, 46.429673),
    (6.843593, 45.991147),
    (6.802355, 45.70858),
    (7.096652, 45.333099),
    (6.749955, 45.028518),
    (7.007562, 44.254767),
    (7.549596, 44.127901),
    (7.435185, 43.693845),
    (6.529245, 43.128892),
    (4.556963, 43.399651),
    (3.100411, 43.075201),
    (2.985999, 42.473015),
    (1.826793, 42.343385),
    (0.701591, 42.795734),
    (0.338047, 42.579546),
    (-1.502771, 43.034014),
    (-1.901351, 43.422802),
    (-1.384225, 44.02261),
    (-1.193798, 46.014918),
    (-2.225724, 47.064363),
    (-2.963276, 47.570327),
    (-4.491555, 47.954954),
    (-4.59235, 48.68416),
    (-3.295814, 48.901692),
    (-1.616511, 48.644421),
    (-1.933494, 49.776342),
    (-0.989469, 49.347376),
    (1.338761, 50.127173),
    (1.639001, 50.946606),
    (2.513573, 51.148506),
    (2.658422, 50.796848),
    (3.123252, 50.780363),
    (3.588184, 50.378992),
    (4.286023, 49.907497),
    (4.799222, 49.985373),
    (5.674052, 49.529484),
    (5.897759, 49.442667),
    (6.18632, 49.463803)])]
    country_code_mapping = {
        'FR': 'France', 'DE': 'Germany', 'IT': 'Italy', 'BE': 'Belgium', 'LU': 'Luxembourg',
        'AT': 'Austria', 'ES': 'Spain', 'NL': 'Netherlands', 'PL': 'Poland', 'HU': 'Hungary',
        'SE': 'Sweden', 'SI': 'Slovenia', 'EE': 'Estonia', 'PT': 'Portugal', 'CH': 'Switzerland',
        'DK': 'Denmark', 'CY': 'Cyprus', 'CZ': 'Czechia', 'FI': 'Finland', 'LT': 'Lithuania',
        'RO': 'Romania', 'SK': 'Slovakia', 'NO': 'Norway', 'MT': 'Malta', 'GR': 'Greece',
        'LV': 'Latvia', 'MC': 'Monaco', 'AE': 'United Arab Emirates', 'IS': 'Iceland',
        'BG': 'Bulgaria', 'HR': 'Croatia', 'CA': 'Canada', 'IL': 'Israel', 'TR': 'Turkey',
        'CL': 'Chile', 'GP': 'Guadeloupe', 'GB': 'United Kingdom', 'IE': 'Ireland'
    }
    #Load the data and prepare the format
    df_map = pd.read_csv('data/all_months_for_python.csv')
    df_map = df_map[df_map['sku'].isin(skus)]

    df_map['country']=df_map['country'].replace(country_code_mapping)
    df_map['buy_date'] = pd.to_datetime(df_map['buy_date'])
    ###Select time range for the heatmap
    #options_drange = pd.date_range(start=df_map['buy_date'].min(), end=df_map['buy_date'].max())
    start_date = st.sidebar.date_input('Start date', min_value=df_map['buy_date'].min(), max_value=df_map['buy_date'].max(), value=df_map['buy_date'].min())
    end_date = st.sidebar.date_input('End date', min_value=df_map['buy_date'].min(), max_value=df_map['buy_date'].max(), value=df_map['buy_date'].max())
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_df = df_map[(df_map['buy_date'] >= start_date) & (df_map['buy_date'] <= end_date)]


    ###calculate the number of days in selected time range 
    num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days

    ###Select the product(sku) for the heatmap
    options_sku = sorted(df['sku'].unique().tolist())
    select_all = st.sidebar.checkbox('Select all products')
    if not select_all:
        selected_skus = st.sidebar.multiselect('Which product do you want?',options_sku)
    else: 
        selected_skus = options_sku
    df_country = filtered_df[filtered_df['sku'].isin(selected_skus)].groupby('country')['kaufdatum'].count().reset_index()
    #options_drange = pd.date_range(start=df_map['buy_date'].min(), end=df_map['buy_date'].max())


    # Merge the data with the GeoDataFrame
    europe = europe.merge(df_country, left_on='name', right_on='country')



    # Create a Streamlit map using Folium
    st.title('Map of sales in European countries')
    m = folium.Map(location=[50, 15], zoom_start=4)

    # Add a choropleth layer to the map
    choropleth_layer =folium.Choropleth(
        geo_data=europe,
        name='choropleth',
        data=europe,
        columns=['country', 'kaufdatum'],
        key_on='feature.properties.name',
        fill_color='RdPu',
        fill_opacity=0.8,
        line_opacity=0.2,
        legend_name='Sales'
    ).add_to(m)

    av_sum_list = ['Average', 'Sum']
    av_sum = st.sidebar.selectbox('Average or Sum?', av_sum_list)
    if av_sum == 'Sum':
        # Add average sales per day in the selected time range to the heatmap
        for idx, row in europe.iterrows():
            country_name = row['name']
            value = round(row['kaufdatum'])#use /num_days for average sales, 3)
             # Get the centroid of the country polygon
            centroid_lat, centroid_lon = row['geometry'].centroid.y, row['geometry'].centroid.x
        #centroid_lat, centroid_lon = row['geometry'].centroid.coords[0]
        
        # Add a marker with the sales value as the label
            folium.Marker(
                location=[centroid_lat, centroid_lon],
                popup=f'{country_name}: {value}',
                icon=folium.DivIcon(
                    icon_size=(60,20),
                    icon_anchor=(30,10),
                    html=f'<div style="font-size: 14pt; font-weight: bold; color: black; background-color: rgba(255, 255, 255, 0.7); ' 
                        f'border-radius: 5px; text-align: center; padding: 5px;">{value}</div>'
                )
        ).add_to(choropleth_layer)

    else:
        for idx, row in europe.iterrows():
            country_name = row['name']
            value = round(row['kaufdatum']/num_days, 3)
        # Get the centroid of the country polygon
            centroid_lat, centroid_lon = row['geometry'].centroid.y, row['geometry'].centroid.x
        #centroid_lat, centroid_lon = row['geometry'].centroid.coords[0]
        
        # Add a marker with the sales value as the label
            folium.Marker(
                location=[centroid_lat, centroid_lon],
                popup=f'{country_name}: {value}',
                icon=folium.DivIcon(
                    icon_size=(60,20),
                    icon_anchor=(30,10),
                    html=f'<div style="font-size: 14pt; font-weight: bold; color: black; background-color: rgba(255, 255, 255, 0.7); ' 
                        f'border-radius: 5px; text-align: center; padding: 5px;">{value}</div>'
                )
        ).add_to(choropleth_layer)

    # Display the map in Streamlit
    folium_static(m)
    #st.write(europe)
map_heatmap_page()
