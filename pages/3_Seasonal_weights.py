import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

import datetime
import dateutil.relativedelta
from datetime import datetime


import src.sf_functions.correction_functions as sfc
st.set_page_config(layout='wide',page_title = "Seasonal weights")
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

def weight_plot(monthly_base):
    # Create a Plotly figure
    fig = go.Figure()

    # Plot baseline
    #fig.add_trace(go.Scatter(x=monthly_base['month_year'], y=0, mode='lines', name='Baseline', line=dict(color='red')))

    # Plot data points
    fig.add_trace(go.Scatter(x=monthly_base['month_year'], y=monthly_base['seasonal_weights'], mode='markers', name='Seasonal Weights', marker=dict(color='blue')))

    fig.add_shape(
        #name = 'baseline',
        type='line',
        x0=monthly_base['month_year'][0],
        x1='2023 - 10',
        y0=0,
        y1=0,
        line=dict(color='red')
    )
    # Connect data points to baseline with vertical lines
    for i, month_year in enumerate(monthly_base['month_year']):
        fig.add_shape(
            type='line',
            x0=month_year,
            x1=month_year,
            y0=0,
            y1=monthly_base['seasonal_weights'][i],
            line=dict(color='gray', dash='solid')
        )

    # Update layout
    fig.update_layout(
        xaxis=dict(tickangle=45, tickmode='array', tickvals=monthly_base['month_year'], ticktext=monthly_base['month_year']),
        title='Seasonal Weights',
        xaxis_title='Month_Year',
        yaxis_title='Seasonal_weight',
        showlegend=True,
        height = 600,
        width= 1000,
    )

    # Show the plot
    st.plotly_chart(fig,use_container_width=True)

def sales_trend_plot(monthly_base):
    # Create a Plotly figure
    fig = go.Figure()

    # Plot baseline
    fig.add_trace(go.Scatter(x=monthly_base['month_year'], y=monthly_base['trend_baseline'], mode='lines', name='Trend', line=dict(color='red')))

    # Plot data points
    fig.add_trace(go.Scatter(x=monthly_base['month_year'], y=monthly_base['sales_per_day'], mode='markers', name='Sales per day', marker=dict(color='blue')))

    # Connect data points to baseline with vertical lines
    for i, month_year in enumerate(monthly_base['month_year']):
        fig.add_shape(
            type='line',
            x0=month_year,
            x1=month_year,
            y0=monthly_base['trend_baseline'][i],
            y1=monthly_base['sales_per_day'][i],
            line=dict(color='gray', dash='solid')
        )

    # Update layout
    fig.update_layout(
        xaxis=dict(tickangle=45, tickmode='array', tickvals=monthly_base['month_year'], ticktext=monthly_base['month_year']),
        title='Sales per day vs Trend',
        xaxis_title='Month_Year',
        yaxis_title='Sales per day',
        showlegend=True,
        height = 600,
        width= 1000,
    )

    # Show the plot
    st.plotly_chart(fig,use_container_width=True)

    
    
    
    
    
def new_seasonal_weights_plot(test):
    fig = go.Figure([
        go.Scatter(
            name='weights',
            x=test['month_name'],
            y=round(test['seasonal_weights']['mean'],2),
            hovertemplate='%{y}% Avg.<br>',
            line=dict(color='rgb(31, 119, 180)'),
            showlegend=False,
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
            name='Min',
            x=test['month_name'],
            y=round(test['seasonal_weights']['min'],2),
            hovertemplate='%{y}%<br>',
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Max',
            x=test['month_name'],
            y=round(test['seasonal_weights']['max'],2),
            hovertemplate='%{y}%<br>',
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        ),

            go.Scatter(
            name='Base',
            x=test['month_name'],
            y=np.array([0,0,0,0,0,0,0,0,0,0,0,0]),
            #y=test['trend_baseline']['mean'],
            line=dict(width=2),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            customdata=round(test['trend_baseline']['mean'], 2),
            hovertemplate='%{customdata} sales per day<br>',
            showlegend=False,
        )

    ])
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Weight %',
        title='Seasonal weights',
        hovermode="x"
    )
    # Show the plot
    st.plotly_chart(fig,use_container_width=True)


    
def create_table(test):
    temp=pd.DataFrame()
    temp['Date']= pd.to_datetime(test['month_name'], format='%B').dt.strftime('%d.%m.') 
    temp['Year']=str(datetime.today().year)
    temp['Date']=temp['Date']+temp['Year']
    temp['HP']=test['seasonal_weights']['mean']

    temp2= pd.DataFrame()
    temp2['Date']=pd.date_range(start=datetime.now().date().replace(month=1, day=1) , end=datetime.now().date().replace(month=12, day=31) , freq='d')
    temp2['month_name']=temp2['Date'].dt.month_name()  

    temp2=temp2.merge(test.droplevel(1, axis=1).iloc[:,0:4], on='month_name', how='left')

    output=pd.DataFrame()
    output['Date']=temp2['Date'].dt.strftime('%d.%m.%Y') 
    output['Year']=str(datetime.now().year)
    output['day']=temp2['Date'].dt.day
    output['HP']=temp2['seasonal_weights']

    return output



def seasonal_weights():
    
    st.sidebar.title("Seasonal weights")  
    
    df = st.session_state.df

    
    st.sidebar.markdown("**Select product(s)**")
    
    skus = df['sku'].unique().tolist()
    
    
    select_all = st.sidebar.checkbox('Select all products')
    if not select_all:
        selected_skus = st.sidebar.multiselect('Which product do you want?',skus)
    else: 
        selected_skus = skus
        
        
        
        
    # Apply corrections (see src/sf_functions/correction_functions.py)
    df = sfc.corrections(df)
    df = df[df['sku'].isin(selected_skus)]
    ##################################################################
    
    if selected_skus: 
        y=df.groupby('date')['sales'].sum().values
        X = np.arange(y.size)

        #fit = np.polyfit(X, y, deg=1)
        #fit_function = np.poly1d(fit)
        fit_function=df.groupby('date')['sales'].sum().rolling(window=365, min_periods=1).mean().values
        
        base=pd.DataFrame()
        base['date'] = df.groupby('date')['sales'].sum().reset_index()['date']
        base['trend']=fit_function #(X)
        base=sfc.date_unwind(base, 'date')
        base['month_year'] = base['month_year'].dt.strftime("%Y - %m")



        monthly_base=base.groupby('month_year')[['trend']].sum().reset_index()
        baseline_values = base.groupby('month_year')['trend'].mean().reset_index()
        monthly_base = pd.merge(monthly_base, baseline_values, on='month_year', how='left',suffixes=('', '_baseline'))

        monthly_base['old_sales']=df.groupby('month_year')[['sales']].sum().reset_index()[['sales']]
        monthly_base['sales_per_day']=df.groupby(['month_year', 'date'])['sales'].sum().groupby('month_year').mean().reset_index()['sales']

        monthly_base['seasonal_weights']=(monthly_base['old_sales']/monthly_base['trend']-1)*100
        
        monthly_base=sfc.date_unwind(monthly_base, 'month_year').sort_values('month_year')
        #weight_plot(monthly_base)
        #sales_trend_plot(monthly_base)

        df_plot= monthly_base.groupby('month_name').agg({              
                                                                    'trend_baseline':'mean', 
                                                                    'sales_per_day':'mean', 
                                                                    'seasonal_weights':['mean', 'min', 'max']
                                                                   }).reset_index()


        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        df_plot['month_name'] = pd.Categorical(df_plot['month_name'], categories=months, ordered=True)
        df_plot = df_plot.sort_values('month_name').reset_index().drop('index', axis=1)
        new_seasonal_weights_plot(df_plot)
        
        
        
        
        st.dataframe(create_table(df_plot))
        
        
#########


seasonal_weights()



