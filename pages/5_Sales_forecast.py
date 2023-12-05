# BASE
# ------------------------------------------------------
import numpy as np
import pandas as pd




# DATA VISUALIZATION
# ------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Datetime
#-------------------------------------------------------
import datetime
import dateutil.relativedelta

# Streamlit
#-------------------------------------------------------
import streamlit as st
import src.sf_functions.correction_functions as sfc

# XGBoost model
#-------------------------------------------------------
import xgboost as xgb
from xgboost import XGBRegressor

# CONFIGURATIONS
# ------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.4f}'.format

import warnings

warnings.filterwarnings('ignore')

st.set_page_config(layout='wide',page_title = "Sales forecast")
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




def get_data():
    # Retrieve the data from session state
    df = st.session_state.df

    st.sidebar.markdown("**Select product(s)**")
    
    skus = sorted(df['sku'].unique().tolist())
    
    
    select_all = st.sidebar.checkbox('Select all products')
    if not select_all:
        selected_skus = st.sidebar.multiselect('Which product do you want?',skus)
    else: 
        selected_skus = skus
    
    # Apply corrections (see src/sf_functions/correction_functions.py)
    df = sfc.corrections(df)
    
    
    
    dfa=sfc.date_unwind(df[df['sku'].isin(selected_skus)][['date', 'sales', 'stock', 'price_de']].groupby('date').sum().reset_index().sort_values('date'), 'date')
    return dfa, selected_skus




def get_dates(dfa):
    # Date range
    st.sidebar.markdown("**Start date for prediction**")

    options_drange = pd.date_range(start=dfa['date'].min(), end=dfa['date'].max()+ dateutil.relativedelta.relativedelta(days=1))
    start_date_train = pd.to_datetime( options_drange.min() )
    end_date_train = pd.to_datetime( st.sidebar.date_input(label='Salact a date', 
                                     min_value=options_drange.min(), 
                                     max_value=options_drange.max(), 
                                     value=options_drange.max())
                                   )


    

    

    #the predicton date range is 90 days after end_date_train
    start_date_predict = end_date_train + dateutil.relativedelta.relativedelta(days=1) 
    end_date_predict = end_date_train + dateutil.relativedelta.relativedelta(days=90) 


    st.sidebar.markdown("**Select a date range for plotting the data**")

    options_drange = pd.date_range(start=dfa['date'].min(), end=dfa['date'].max())
    start_plot_date = pd.to_datetime( st.sidebar.date_input('Start date for plotting', 
                                      min_value=options_drange.min(), 
                                      max_value=options_drange.max(), value=options_drange.min())
                                    )
    end_plot_date = pd.to_datetime( st.sidebar.date_input('End date for plotting', 
                                    min_value=options_drange.min(), 
                                    max_value=options_drange.max(), 
                                    value=options_drange.max())
                                  )    
    
    return start_date_train, end_date_train, start_date_predict, end_date_predict, start_plot_date, end_plot_date
    
    
    
def XGB_model(dfa,start_date_train,end_date_train,start_date_predict,end_date_predict):

    df_sku = dfa.copy()
    
    #ask user for price or take mode of price
    most_frequent_historic_price = round(df_sku['price_de'].mode()[0],2)
    price_for_forecasting = most_frequent_historic_price                 #st.number_input('Future price', value=most_frequent_historic_price)
    
    # create new dates to extend the dataframe for prediction of 90 days after the end_date_train
    start_date_extend = pd.to_datetime(df_sku.date.max()) + dateutil.relativedelta.relativedelta(days=2)
    end_date_extend = end_date_predict

    #create dataframe with new dates
    temp=pd.DataFrame()
    temp['date']=pd.date_range(start=start_date_extend , end=end_date_extend , freq='d')
    temp[['sales', 'price_de']] = None

    #concat the dataframe to the old one
    df_sku = pd.concat([df_sku, temp], ignore_index=True)


    #fill price column with user input
    df_sku['price_de'] = price_for_forecasting #df_sku.loc[last_valid_index_sku, 'price_de']

    df_sku = sfc.date_unwind(df_sku, 'date')
    

    # Add lagged sales feature (does not necessarily improve the forecast)
    #target_map = df_sku.set_index('date')['sales'].to_dict()
    #df_sku['sales_lag90']=(df_sku.set_index('date').index - pd.Timedelta('90 days')).map(target_map) 
    #df_sku['sales_lag90']=df_sku['sales'].shift(90)
    #df_sku['stock_lag90']=df_sku['stock'].shift(90)


    # Train-test split 
    X_train_sku = df_sku[(df_sku['date'] >= start_date_train) & (df_sku['date'] <= end_date_train)][['year', 'month', 'day','day_of_year', 'price_de']]
    y_train_sku = df_sku[(df_sku['date'] >= start_date_train) & (df_sku['date'] <= end_date_train)]['sales']

    X_test_sku = df_sku[(df_sku['date'] >= start_date_predict) & (df_sku['date'] <= end_date_predict)][['year', 'month', 'day', 'day_of_year', 'price_de']]
    y_test_sku = df_sku[(df_sku['date'] >= start_date_predict) & (df_sku['date'] <= end_date_predict)]['sales']




    xgbr_model_sku = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)


    xgbr_model_sku.fit(X_train_sku, y_train_sku)

    # Predict on the test set

    y_pred_sku = xgbr_model_sku.predict(X_test_sku)

    # Sales can not become negative
    y_pred_sku[y_pred_sku<0]=0
    
    return y_pred_sku







def get_monthly_sales(dfa, yhat,train_drange):
    dates=np.array([])
    predicted_sales=pd.DataFrame([ np.append(dates, train_drange.max() + dateutil.relativedelta.relativedelta(days=i+1)) for i in range(len(yhat)) ],yhat.flatten()).reset_index().rename(columns={0:'date', 'index':'sales_prediction'})
    compare_df = predicted_sales.merge(dfa, on='date', how='left').drop(['stock', 'price_de'], axis=1)
    return (compare_df
            .groupby('month_year')[['sales','sales_prediction']]
            .sum()
            .reset_index()
            .rename(columns={
                            "month_year": "Year-Month", 
                            "sales": "Actual monthly sales", 
                            "sales_prediction": "Predicted monthly sales"
                            }
                   )
           )



def plot(filtered_df, new_future_data):

    # Calculate daily sales
    p_daily_sales = filtered_df.groupby('date')['sales'].sum().reset_index()
    f_daily_sales = new_future_data.groupby('date')['sales_prediction'].sum().reset_index()

    # Create traces for past and future data
    trace_past = go.Scatter(x=p_daily_sales['date'], y=p_daily_sales['sales'], mode='lines', name='Original Sales', line=dict(color='dodgerblue'))
    trace_future = go.Scatter(x=f_daily_sales['date'], y=f_daily_sales['sales_prediction'], mode='lines', name='Sales Forecast', line=dict(color='crimson'))


    # Add a vertical line at the present day
    present_day = new_future_data['date'].min()
    trace_present_day = go.Scatter(x=[present_day, present_day], 
                                  y=[0, max(p_daily_sales['sales'].max(), 
                                            f_daily_sales['sales_prediction'].max())],
                                  mode='lines', 
                                  name='Start of prediction',
                                  line=dict(color='green', dash='dash'))

    # Create a layout
    layout = go.Layout(title='Sales Forecast',
                      xaxis=dict(title='Date'),
                      yaxis=dict(title='Sales'),
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=1.02,
                          xanchor="right",
                          x=1
                      )                       
                      )

    # Create a figure
    fig = go.Figure(data=[trace_past, trace_present_day, trace_future], layout=layout)


    st.plotly_chart(fig , use_container_width=True)



def plot_cumsum(compare_df):
    
    fig = go.Figure([
        go.Scatter(
            name='Actual cumulated sales',
            x=compare_df['date'],
            y=round(compare_df['sales_cumsum']),
            line=dict(color='rgb(30,144,255)'),
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
            name='Predicted cumulated sales',
            x=compare_df['date'],
            y=round(compare_df['sales_prediction_cumsum']),
            line=dict(color='rgb(220,20,60)'),
            marker=dict(
                color='LightSkyBlue',
                size=10,
                line=dict(
                    color='MediumPurple',
                    width=2
                )
            ),
        ),        
    ])
    fig.update_layout(
        xaxis_title= 'Date',
        yaxis_title='Cumulated sales number',
        title='Cumulated sales',
        hovermode="x"
    )
    st.plotly_chart(fig)

    
    

def plot_stock(compare_df):
    
    fig = go.Figure([
        go.Scatter(
            name='Actual stock',
            x=compare_df['date'],
            y=round(compare_df['stock']),
            line=dict(color='rgb(30,144,255)'),
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
            name='Predicted stock',
            x=compare_df['date'],
            y=round(compare_df['predicted_stock']),
            line=dict(color='rgb(220,20,60)'),
            marker=dict(
                color='LightSkyBlue',
                size=10,
                line=dict(
                    color='MediumPurple',
                    width=2
                )
            ),
        )   
    ])
    fig.update_layout(
        xaxis_title= 'Date',
        yaxis_title='Items in stock',
        title='Number of items in stock',
        hovermode="x"
    )
    st.plotly_chart(fig)

    
    
    
def create_tables(predicted_sales, dfa):

       
    compare_df = predicted_sales.copy().merge(dfa, on='date', how='left').drop(['price_de'], axis=1)

    #calculate baseline
    fullrange = predicted_sales.copy().merge(dfa, on='date', how='outer').drop(['stock', 'price_de'], axis=1).sort_values('date')
    fullrange.loc[fullrange['sales_prediction'].notna(), 'sales'] = fullrange['sales_prediction']
    fullrange['trend']=fullrange['sales'].rolling(window=365, min_periods=1).mean()
    compare_df=compare_df.merge(fullrange[['date','trend']], on='date', how='left')

    
    
    compare_df_monthly = (
                          sfc.date_unwind(compare_df, 'date')
                          .groupby('month_year')[['sales', 'sales_prediction', 'trend', 'day']]
                          .agg({'day': 'count', 'sales': 'sum', 'sales_prediction': 'sum', 'trend': 'mean'})
                          .reset_index()
                          .rename(columns={
                                          "month_year": "Year-Month", 
                                          "day": "Days included", 
                                          "sales": "Actual monthly sales", 
                                          "sales_prediction": "Predicted monthly sales", 
                                          "trend": 'Predicted base value'
                                          })
                         )
    
    #Uplift
    compare_df_monthly['Predicted Uplift %']=100*((compare_df_monthly["Predicted monthly sales"]/compare_df_monthly["Days included"])-compare_df_monthly['Predicted base value'])/(compare_df_monthly["Predicted monthly sales"]/compare_df_monthly["Days included"])
    compare_df_monthly["Year-Month"]=compare_df_monthly["Year-Month"].astype(str)
    compare_df_monthly['Deviation %']=(compare_df_monthly["Predicted monthly sales"]-compare_df_monthly["Actual monthly sales"])/compare_df_monthly["Actual monthly sales"] *100



    # print table with overall sales and predicted sales for the whole 90 day period
    compare_df_monthly_sum=pd.DataFrame(compare_df_monthly[['Actual monthly sales', 'Predicted monthly sales']].sum()).T.rename(
                                                                                                                                index={0:'sum'}, 
                                                                                                                                columns={
                                                                                                                                        'Actual monthly sales':'Actual sales', 
                                                                                                                                        'Predicted monthly sales':'Predicted sales'
                                                                                                                                        }
                                                                                                                                )
    
    
    compare_df_monthly_sum['Deviation %']=(compare_df_monthly_sum["Predicted sales"]-compare_df_monthly_sum["Actual sales"])/compare_df_monthly_sum["Actual sales"] *100

    return (
            compare_df, 
            compare_df_monthly[["Year-Month",'Days included', "Actual monthly sales","Predicted monthly sales",'Deviation %', 'Predicted base value', 'Predicted Uplift %']].set_index("Year-Month").round(
                                                                                                                                                                            {
                                                                                                                                                                             "Actual monthly sales":0,
                                                                                                                                                                             "Predicted monthly sales":0
                                                                                                                                                                            }), 
            compare_df_monthly_sum.round({
                                          "Actual sales":0,"Predicted sales":0
                                         })
           )
    
    
def sales_forecast_page():
    st.sidebar.title("Sales Forecast")  

    dfa, selected_skus = get_data()

    if selected_skus:
        
        start_date_train, end_date_train, start_date_predict, end_date_predict, start_plot_date, end_plot_date = get_dates(dfa)
       
        train_drange = pd.date_range(start=start_date_train, end=end_date_train)
        plot_df      = dfa.query("date >= @start_plot_date and date <= @end_plot_date")
        filtered_df  = dfa.query("date >= @start_date_train and date <= @end_date_train")
        
        # train model, predict and plot prediction next to original
        yhat = XGB_model(dfa,start_date_train,end_date_train,start_date_predict,end_date_predict)

        dates=np.array([])
        predicted_sales=pd.DataFrame(
                                     [ np.append(dates, train_drange.max() + dateutil.relativedelta.relativedelta(days=i+0)) for i in range(len(yhat)) ],
                                     yhat.flatten()
                                    ).reset_index().rename(columns={
                                                                    0      :'date', 
                                                                    'index':'sales_prediction'
                                                                   }
                                                          )
        
        plot(plot_df, predicted_sales)

        # Additional outputs

        compare_df, compare_df_monthly, compare_df_monthly_sum = create_tables(predicted_sales, dfa)
    
    
    
        st.markdown("Monthly sales for the prediction period (90 days)")
        st.dataframe(compare_df_monthly)
        
        st.markdown("Sales over the entire prediction period (90 days)")
        st.dataframe(compare_df_monthly_sum)          
        
        chart_sales, chart_stock = st.columns(2)
            
        with chart_sales:    
            # plot cumulated sales over time
            compare_df['sales_prediction_cumsum']=compare_df['sales_prediction'].cumsum()
            compare_df['sales_cumsum']=compare_df['sales'].cumsum()    
            plot_cumsum(compare_df)
        
        
        with chart_stock:
            # plot stock sales over time
            last_recorded_stock_from_training_set=filtered_df.loc[filtered_df['stock'].last_valid_index(), 'stock']  
            compare_df['predicted_stock']=-compare_df['sales_prediction'].cumsum()+last_recorded_stock_from_training_set
            compare_df['predicted_stock'][compare_df['predicted_stock']<0]=0
            plot_stock(compare_df)                
                
sales_forecast_page()

