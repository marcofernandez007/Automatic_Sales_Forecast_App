import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

import src.sf_functions.correction_functions as sfc

st.set_page_config(layout='wide', page_title = "Time series data")


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



def time_series_page():


    #location of the title(using HTML code contains CSS style information)
    st.markdown('<style>div.block-container{padding-top:2rem}</style>', unsafe_allow_html=True)

    st.sidebar.title("Time series data")  
    # Retrieve the data from session state
    df = st.session_state.df
    
    
    st.sidebar.markdown("**Select product(s)**")

    skus = sorted(df['sku'].unique().tolist())
    
    # SKU filter
    select_all = st.sidebar.checkbox('Select all products')
    if not select_all:
        selected_skus = st.sidebar.multiselect('Which product do you want?',skus)
    else: 
        selected_skus = skus
    
    # Apply corrections (see src/sf_functions/correction_functions.py)
    df = sfc.corrections(df)
    
    
    
    # Plotting filter
    colors = px.colors.qualitative.D3
    sku_color_map = {sku: color for sku, color in zip(selected_skus, colors)}
    sum_mean_sep_list = ['Sum', 'Average', 'Seperate']
    sum_mean_sep = st.sidebar.selectbox('How to plot sales of different products?', sum_mean_sep_list)
    



    
    # Date range filter
    st.sidebar.markdown("**Select a date range**")
    options_drange = pd.date_range(start=df['date'].min(), end=df['date'].max())
    start_date = st.sidebar.date_input('Start date', min_value=options_drange.min(), max_value=options_drange.max(), value=options_drange.min())
    end_date = st.sidebar.date_input('End date', min_value=options_drange.min(), max_value=options_drange.max(), value=options_drange.max())
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    


    ## Filter price and sales columns
    # Columns to keep other than price and sales
    default_columns = ['date', 'sku', 'stock', 'reserved', 'sales', 'country', 
                       'year', 'month', 'day', 'day_name', 'day_num', 'day_of_year', 'month_year', 'stockout', 'original_sales'
                       ]

    # Filter the DataFrame to keep the specified columns
    filtered_df = df.copy()

    # Apply filter option to DataFrame
    #Date
    filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]
    
    # Create a new df for price/sales per country
    no_geo_filtered_df = filtered_df.copy()




    # Apply sku and create a final Dataframe to plot
    df_plot=filtered_df[filtered_df['sku'].isin(selected_skus)].sort_values('date')
    df_plot_no_geo=no_geo_filtered_df[no_geo_filtered_df['sku'].isin(selected_skus)].sort_values('date')



    if selected_skus:
              
                tab1, tab2  = st.tabs(["Sales/Stock by product", "Price/Sales by country"])

                with tab1:
                    if sum_mean_sep == 'Sum':
                        df_plot_sum = df_plot[df_plot['sku'].isin(selected_skus)].groupby('date')[['sales','original_sales']].sum().reset_index()
                        df_plot_sum['trendline']= df_plot_sum['sales'].rolling(window=30,center=True,min_periods=1).mean()
                        fig = go.Figure()
                        if st.session_state['key'] == 1:
                            fig.add_trace(go.Scatter(x=df_plot_sum['date'], y=df_plot_sum['original_sales'],
                                         mode='lines',
                                         name='Original Sales',
                                         line=dict(width=0.9, color=px.colors.qualitative.D3[1]))) 
                            st.session_state['key'] = 0
                        fig.add_trace(go.Scatter(x=df_plot_sum['date'], y=df_plot_sum['sales'],
                                    mode='lines',
                                    name='Sales',
                                    line=dict(width=1.3, color=px.colors.qualitative.D3[0])))
                        # format for trendline
                        fig.add_trace(go.Scatter(x=df_plot_sum['date'], y=df_plot_sum['trendline'],
                                    mode='lines',
                                    name='Trend',
                                    line=dict(dash ='dot', width=4, color=px.colors.qualitative.D3[0]),
                                    ))
                        fig.update_layout(
                            title='Sales trend for selected products',
                            xaxis_title='Date',
                            yaxis_title='Sales',
                            legend_title=f'Sum sales of<br>{",<br>".join(selected_skus)}',
                            hovermode="x",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                    if sum_mean_sep == 'Average':
                        df_plot_av = df_plot[df_plot['sku'].isin(selected_skus)].groupby('date')[['sales', 'original_sales']].mean().reset_index()
                        df_plot_av['trendline']= df_plot_av['sales'].rolling(window=30,center=True,min_periods=1).mean()
                        fig = go.Figure()
                        if st.session_state['key'] == 1:
                            fig.add_trace(go.Scatter(x=df_plot_av['date'], y=df_plot_av['original_sales'],
                                         mode='lines',
                                         name='Original Sales',
                                         line=dict(width=0.9, color=px.colors.qualitative.D3[1]))) 
                            st.session_state['key'] = 0                        
                        fig.add_trace(go.Scatter(x=df_plot_av['date'], y=df_plot_av['sales'],
                                    mode='lines',
                                    name='Sales',
                                    line=dict(width=1.3, color=px.colors.qualitative.D3[0])))
                        # format for trendline
                        fig.add_trace(go.Scatter(x=df_plot_av['date'], y=df_plot_av['trendline'],
                                    mode='lines',
                                    name='Trend',
                                    line=dict(dash ='dot', width=4, color=px.colors.qualitative.D3[0]),
                                    ))
                        fig.update_layout(
                            title='Sales trend for selected products',
                            xaxis_title='Date',
                            yaxis_title='Sales',
                            legend_title=f'Average sales of<br>{",<br>".join(selected_skus)}',
                            hovermode="x",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )                            
                        )
                    if sum_mean_sep == 'Seperate':
                        fig = go.Figure()
                        for sku in selected_skus:
                            df_plot.loc[df_plot['sku']==sku, 'trendline']= df_plot[df_plot['sku']==sku]['sales'].rolling(window=30,center=True,min_periods=1).mean()
                        for i, sku in enumerate(selected_skus):
                            line_color = sku_color_map[sku]
                            # format for line sales sku
                            fig.add_trace(go.Scatter(x=df_plot[df_plot['sku'] == sku]['date'], y=df_plot[df_plot['sku'] == sku]['sales'],
                                        mode='lines',
                                        name=f'Sales - {sku}',
                                        line=dict(width=1.3, color=line_color)))
                            # format for trendline
                            fig.add_trace(go.Scatter(x=df_plot[df_plot['sku'] == sku]['date'], y=df_plot[df_plot['sku'] == sku]['trendline'],
                                        mode='lines',
                                        name=f'Trend - {sku}',
                                        line=dict(dash ='dot', width=4, color=line_color),
                                        ))
                            fig.update_layout(
                                title='Sales Trend for Selected SKUs',
                                xaxis_title='Date',
                                yaxis_title='Sales',
                                legend_title='SKU',
                                hovermode="x",
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )                                
                            )
                    
                    fig.update_xaxes(rangeslider_visible=False)

                    # Show the plot
                    st.plotly_chart(fig, use_container_width=True)
            

                    # Stock for selected
                    fig_stock = px.line(df_plot, x='date', y='stock', color='sku', color_discrete_sequence=["orange", "red", "green", "blue", "purple"]) # plotly
                    fig_stock.update_layout(
                                            title_text="Stock for selected products",
                                            legend=dict(
                                                orientation="h",
                                                yanchor="bottom",
                                                y=1.02,
                                                xanchor="right",
                                                x=1
                                            ))                    
                    fig_stock.update_xaxes(rangeslider_visible=False)   
                    st.plotly_chart(fig_stock, use_container_width=True) # plotly

                with tab2:
                        #Country filter
                   # if selected_countries: 
                        df['country'].fillna('DE', inplace=True)
                        country = sorted(df['country'].unique().tolist())
                        all_countries = st.checkbox("Select all countries", value=True)
                        if not all_countries:
                            selected_countries = st.multiselect("**Select country**", country)
                        else:
                            selected_countries = country
                        #st.write(country)
                            #Country
                        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]

                            # Convert country codes to lowercase for filtering
                        lowercase_countries = [country.lower() for country in selected_countries]

                        # Generate a list of columns to keep
                        price_columns_to_keep = [f'price_{country.lower()}' for country in lowercase_countries] # price_country code
                        sales_columns_to_keep = [f'sale_{country.lower()}' for country in lowercase_countries] # sale_country code


                        # Combine all columns to keep
                        all_columns_to_keep = default_columns + price_columns_to_keep + sales_columns_to_keep

                        # Filter the DataFrame to keep the specified columns
                        filtered_df = df[all_columns_to_keep]


                        # Avg price per country
                        df_plot_mean_price = df_plot_no_geo.groupby('date')[price_columns_to_keep].mean().reset_index()
                        fig_price = px.line(df_plot_mean_price, x='date', y=price_columns_to_keep,
                                    labels={'value': 'Price', 'variable': 'Country'},
                                    title=f'Daily average price per country',
                                    color_discrete_sequence=["orange", "red", "green", "blue", "purple"]
                                    )


                        st.plotly_chart(fig_price, use_container_width=True)


                        # Daily sales per country
                        df_plot_sum_sales = df_plot_no_geo.groupby('date')[sales_columns_to_keep].sum().reset_index()
                        fig_sales = px.line(df_plot_sum_sales, x='date', y=sales_columns_to_keep,
                                    labels={'value': 'Sales', 'variable': 'Country'},
                                    title=f'Daily Sales per country',
                                    color_discrete_sequence=["orange", "red", "green", "blue", "purple"]
                                    )

                        st.plotly_chart(fig_sales, use_container_width=True)



                 #   else:
                  #      text_with_formatting2 = "Please select at least one <span style='color:red; font-weight:bold;'>Country</span> on the side menu"
                  #      st.markdown(text_with_formatting2, unsafe_allow_html=True)
    else:
        text_with_formatting2 = "Please select at least one <span style='color:red; font-weight:bold;'>product</span> on the side menu"
        st.markdown(text_with_formatting2, unsafe_allow_html=True)


time_series_page()  







