# Copyright (c) we

import streamlit as st
from streamlit.logger import get_logger
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.graph_objects as go
from st_pages import Page, show_pages, add_page_title

import src.sf_functions.correction_functions as sfc


LOGGER = get_logger(__name__)



st.set_page_config(
    page_title="Welcome to the sales forecast board",
    #page_icon='logo.jpg',
    layout='wide'
)
#st.image('logo.jpg', use_column_width=False, width=150)

st.markdown("<font size='8'>**Welcome to the sales forecast board! 👋**</font>", unsafe_allow_html=True)


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




    
  
    
    

def run():


    # read in the files
    families=['A','B']
    selected_family = st.sidebar.selectbox('Which product family do you want?',families)

    if selected_family=='A': 
        df=pd.read_csv('../data/A_Final.csv')

    if selected_family=='B':
        df=pd.read_csv('../data/B_Final.csv')

        
    df.sort_values(['date', 'sku'], inplace=True) # sort by date and sku
    df=df.reset_index().drop(['index'], axis=1) # make new index after sorting
    df=sfc.date_unwind(df, 'date') # create columns with additional datetime information 
    df.query("date>='2021-09-01'", inplace=True)
    
    
    
    
    
    
    
    
    new_df=pd.DataFrame()


    for sku in df['sku'].unique().tolist(): 
        
        starting_date = df[df['sku']==sku]['date'].min()
        ending_date   = df[df['sku']==sku]['date'].max()
        temp=pd.DataFrame()
        temp['date']=pd.date_range(start=starting_date , end=ending_date , freq='d')
        temp['sku']=sku
        temp=temp.merge(df[df['sku']==sku], on='date', how='left')

        temp['stock'] = temp['stock'].interpolate(method='pad')  

        temp['sales']=temp['sales'].fillna(0)
        new_df=pd.concat([new_df, temp])






    new_df = new_df.rename(columns={"sku_x": "sku"})
    new_df = new_df.reset_index().drop(['index', 'sku_y'], axis=1)
    
    df=new_df.copy()
    
    
    
    df=sfc.date_unwind(df, 'date') # create columns with additional datetime information 
    
    
    
    
    

    df['stockout']=(((df['sales']==0) & (df['stock']<=4)) | ((df['sales']==0) & (df['stock']==0)) | ((df['reserved']>0) & (df['stock']<=1))   ).astype(int) # create column that is 1 if there is a stockout or 0 if there is no stockout
    
    for sku in df['sku'].unique().tolist():
        stock_condition = (df.loc[df['sku'] == sku, 'stock'] < df[df['sku'] == sku].groupby('month_year')['sales'].transform('mean')) & (df.loc[df['sku'] == sku, 'stock'] <= df.loc[df['sku'] == sku, 'sales'])
        df.loc[df['sku'] == sku, 'stock_condition'] = stock_condition
        df.loc[df['sku'] == sku, 'stockout'] = df.loc[df['sku'] == sku, 'stockout'] + stock_condition.astype(int)
    
    df['original_sales']=df['sales']
    df.loc[df['stockout']==2, 'stockout'] = 1

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  

    st.session_state.df = df
    
    #st.write(st.session_state.df)

    if 'key' not in st.session_state:
        st.session_state['key'] = 0
    
    
    
    
if __name__ == "__main__":
    run()
    
    from st_pages import Page, show_pages, add_page_title

        # Optional -- adds the title and icon to the current page
    #add_page_title()

        # Specify what pages should be shown in the sidebar, and what their titles 
        # and icons should be
    show_pages(
            [
                Page("dashboard.py", "Home"),
                Page("pages/time_series_page.py", "Time Series Data"),
                Page("pages/aggregation_page.py", "Aggregated Data"),
                Page("pages/seasonal_weights.py", "Seasonal Weights"),
                Page("pages/map_heatmap_page.py", "Map"),
                Page("pages/sales_forecast-XGB.py", "Sales Forecast")

            ]
    )  
