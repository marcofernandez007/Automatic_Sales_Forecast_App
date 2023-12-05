import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.graph_objects as go

import src.sf_functions.correction_functions as sfc

st.set_page_config(layout='wide',page_title = "Aggregated Data")
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

def make_plot(df_agg, columnname, x_axis_name):

    fig = go.Figure([
        go.Scatter(
            name='sales',
            x=df_agg[columnname],
            y=df_agg['sales'],
            line=dict(color='rgb(31, 119, 180)'),
            marker=dict(
                color='LightSkyBlue',
                size=10,
                line=dict(
                    color='MediumPurple',
                    width=2
                )
            ),
        ),
        go.Scatter(
            name='Upper STD',
            x=df_agg[columnname],
            y=df_agg['sales']+df_agg['std'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower STD',
            x=df_agg[columnname],
            y=df_agg['sales']-df_agg['std'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(
        xaxis_title= f'{x_axis_name}',
        yaxis_title='Number of sales',
        title='Aggregated number of sales',
        hovermode="x"
    )
    st.plotly_chart(fig,use_container_width=True) # plotly



def aggregation_page():
    
    st.sidebar.title("Aggregated data")  
    
    # Retrieve the data from session state
    df = st.session_state.df

    
    st.sidebar.markdown("**Select product(s)**")
    
    skus = sorted(df['sku'].unique().tolist())
    
    
    select_all = st.sidebar.checkbox('Select all products')
    if not select_all:
        selected_skus = st.sidebar.multiselect('Which product do you want?',skus)
    else: 
        selected_skus = skus
        
        
    
    

    options=['Daily', 'Weekly','Monthly','Yearly']

    
    st.sidebar.markdown("**Aggregation type**")
    selected_option = st.sidebar.selectbox('Select aggregation time interval',options)

    
    # Apply corrections (see src/sf_functions/correction_functions.py)
    df = sfc.corrections(df)
    
    
    if selected_skus:
        if selected_option =='Daily':
            df_time =pd.read_csv('data/all_months_for_python.csv').sort_values('buy_time')
            df_time = sfc.date_unwind(df_time, 'buy_date')
            df_time = df_time[df_time['sku'].isin(selected_skus)]
            df_time['buy_time'] = pd.to_timedelta(df_time['buy_time'])
            bins = pd.timedelta_range(start='00:00:00', end='23:59:59', freq='2H')
            df_time['time_bin'] = pd.cut(df_time['buy_time'], bins=bins, include_lowest=True).astype(str)
            bin_names = {
                '(-1 days +23:59:59.999999999, 0 days 02:00:00]': '0:00 - 2:00',
                '(0 days 02:00:00, 0 days 04:00:00]': '2:00 - 4:00',
                '(0 days 04:00:00, 0 days 06:00:00]': '4:00 - 6:00',
                '(0 days 06:00:00, 0 days 08:00:00]': '6:00 - 8:00',
                '(0 days 08:00:00, 0 days 10:00:00]': '8:00 - 10:00',
                '(0 days 10:00:00, 0 days 12:00:00]': '10:00 - 12:00',
                '(0 days 12:00:00, 0 days 14:00:00]': '12:00 - 14:00',
                '(0 days 14:00:00, 0 days 16:00:00]': '14:00 - 16:00',
                '(0 days 16:00:00, 0 days 18:00:00]': '16:00 - 18:00',
                '(0 days 18:00:00, 0 days 20:00:00]': '18:00 - 20:00',
                '(0 days 20:00:00, 0 days 22:00:00]': '20:00 - 22:00', 
                'nan': '22:00 - 24:00',
            }
            df_time['time_bin_name'] = pd.Categorical(df_time['time_bin'].astype(str).map(bin_names), categories=bin_names.values(), ordered=True)
            df_groupbytime = df_time.groupby(['buy_date','time_bin_name'])['sales'].sum().reset_index().groupby('time_bin_name')['sales'].mean().reset_index()
            df_groupbytime['std'] = df_time.groupby(['buy_date','time_bin_name'])['sales'].sum().reset_index().groupby('time_bin_name')['sales'].std().reset_index()['sales']
            make_plot(df_groupbytime, 'time_bin_name', 'Time of day')
        if selected_option == 'Weekly':
            df_agg        = sfc.date_unwind(df[df['sku'].isin(selected_skus)].groupby('date')['sales'].sum().reset_index(), 'date').groupby(["day_name", "day_num"])['sales'].mean().reset_index().sort_values('day_num') 
            df_agg['std'] = sfc.date_unwind(df[df['sku'].isin(selected_skus)].groupby('date')['sales'].sum().reset_index(), 'date').groupby(["day_name", "day_num"])['sales'].std().reset_index().sort_values('day_num')['sales']
            make_plot(df_agg, 'day_name', 'Day')

        if selected_option == 'Monthly':
            df_agg        = sfc.date_unwind(df[df['sku'].isin(selected_skus)].groupby('date')['sales'].sum().reset_index(), 'date').groupby(["day"])['sales'].mean().reset_index().sort_values('day') 
            df_agg['std'] = sfc.date_unwind(df[df['sku'].isin(selected_skus)].groupby('date')['sales'].sum().reset_index(), 'date').groupby(["day"])['sales'].std().reset_index().sort_values('day')['sales']
            make_plot(df_agg, 'day', 'Day')
        
        if selected_option == 'Yearly':
            df_agg        = sfc.date_unwind(df[df['sku'].isin(selected_skus)].groupby('date')['sales'].sum().reset_index(), 'date').groupby(["month"])['sales'].mean().reset_index().sort_values('month') 
            df_agg['std'] = sfc.date_unwind(df[df['sku'].isin(selected_skus)].groupby('date')['sales'].sum().reset_index(), 'date').groupby(["month"])['sales'].std().reset_index().sort_values('month')['sales']
            make_plot(df_agg, 'month', 'Month')
        


  
    
aggregation_page()
