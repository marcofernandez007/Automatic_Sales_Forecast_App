import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

def date_unwind(df, datecol):
    df[datecol]  = pd.to_datetime(df[datecol])
    df['year']  = df[datecol].dt.year
    df['month'] = df[datecol].dt.month
    df['day'] = df[datecol].dt.day
    df['day_name'] = df[datecol].dt.day_name()
    df['month_name'] = df[datecol].dt.month_name()  
    df['day_num'] = df[datecol].dt.dayofweek+1
    df['day_of_year'] = df[datecol].dt.dayofyear
    df['month_year'] = df[datecol].dt.to_period('M')
    df['week'] = df[datecol].dt.isocalendar().week.astype('int')
    return df



def stockout_correction(df):
    df_raw=df.copy()
    for sku in df_raw['sku'].unique().tolist():
        sku_condition = df_raw['sku'] == sku

        df_raw.loc[sku_condition, 'stockout_group'] = ((df_raw.loc[sku_condition, 'stockout'] == 0) | ( df_raw.loc[sku_condition, 'stockout'].shift() == 0)).cumsum()

        stockout_durations = df_raw[sku_condition].groupby(['stockout_group'])['date'].agg(['min', 'max'])
        stockout_durations['duration'] = (stockout_durations['max'] - stockout_durations['min']).dt.days + 1

        df_raw.loc[sku_condition, 'duration'] = df_raw.loc[sku_condition, 'stockout_group'].map(stockout_durations['duration'])
        df_raw.loc[(sku_condition) & (df_raw['duration'] > 7) , 'long_stockout'] = 1

    df_raw['long_stockout'] = df_raw['long_stockout'].fillna(0).astype(int) 


        
    for sku in df_raw['sku'].unique().tolist():
        sku_condition = df_raw['sku'] == sku
        long_stockout_condition = df_raw['long_stockout'] == 1
        before_nov_2022_condition = df_raw['month_year'] < '2022-11'
        df_raw.loc[(sku_condition) & (long_stockout_condition), 'sales'] = np.nan

        fit_function=df_raw[df_raw['sku']==sku]['sales'].rolling(window=365, min_periods=1).mean().values
        df_raw.loc[(sku_condition),'trend'] = fit_function #(X)

        
    for sku in df_raw['sku'].unique().tolist():
        sku_condition = df_raw['sku'] == sku
        long_stockout_condition = df_raw['long_stockout'] == 1
        before_nov_2022_condition = df_raw['month_year'] < '2022-11'
        
        d22 = df_raw[sku_condition & long_stockout_condition & before_nov_2022_condition][['date', 'sales', 'trend']]
        d22=d22.reset_index()

        imputedates1 = df_raw.loc[sku_condition & long_stockout_condition & before_nov_2022_condition,['date']] + pd.DateOffset(years=1)


        d23=pd.DataFrame(df_raw.loc[df_raw['date'].isin(imputedates1['date']), 'sales'].interpolate()).reset_index()
        d23['trend'] = pd.DataFrame(df_raw.loc[df_raw['date'].isin(imputedates1['date']), 'trend']).reset_index()['trend']
        d22['newsales']=d23['sales']
        d22['trend2']=d23['trend']
        d22 = d22.set_index('index')
        #d22['sales']=d22['newsales']- (d22['trend2']-d22['trend'])
        d22['sales']=d22['newsales']/ (d22['trend2']/d22['trend'])
        df_raw.loc[sku_condition & long_stockout_condition & before_nov_2022_condition, 'sales']=d22['sales']

        
    for sku in df_raw['sku'].unique().tolist():
        sku_condition = df_raw['sku'] == sku
        long_stockout_condition = df_raw['long_stockout'] == 1
        before_nov_2022_condition = df_raw['month_year'] < '2022-11'
        
        d22b = df_raw[sku_condition & long_stockout_condition & ~before_nov_2022_condition][['date', 'sales', 'trend']]
        d22b=d22b.reset_index()

        imputedates2 = df_raw.loc[sku_condition & long_stockout_condition & ~before_nov_2022_condition,['date']] - pd.DateOffset(years=1)


        d23b=pd.DataFrame(df_raw.loc[df_raw['date'].isin(imputedates2['date']), 'sales'].interpolate()).reset_index()
        d23b['trend'] = pd.DataFrame(df_raw.loc[df_raw['date'].isin(imputedates2['date']), 'trend']).reset_index()['trend']
        d22b['newsales']=d23b['sales']
        d22b['trend2']=d23b['trend']
        d22b = d22b.set_index('index')
        #d22b['sales']=d22b['newsales']- (d22b['trend2']-d22b['trend'])
        d22b['sales']=d22b['newsales']/ (d22b['trend2']/d22b['trend'])
        df_raw.loc[sku_condition & long_stockout_condition & ~before_nov_2022_condition, 'sales']=d22b['sales']
        
        
    for sku in  df_raw['sku'].unique().tolist():
        df_0=df_raw.copy()
        df_0.loc[(df_0['sku']==sku) & (df_0['stockout']!=0) & (~df_0['long_stockout']==1), 'sales']=df_0.groupby('month_year')['sales'].transform('mean')
        df_0['sales'][df_0['sales']<0]=0


    return df_0








def stock_sales_correction(df):
    df_raw=df.copy()
    for sku in  df_raw['sku'].unique():
        df_0=df_raw.copy()
        df_0.loc[(df_0['sku']==sku) & (df_0['stockout']!=0), 'sales']=np.nan
        df_1 = df_0[df_0['sku']==sku][['month_year', 'sales']]

        df_1['sales_new']=df_1['sales'].fillna(df_1.groupby('month_year')['sales'].transform('mean'))
        
        
        df_raw.loc[df_raw['sku'] == sku, 'sales'] = df_1['sales_new'] 

    return df_raw 



def new_product_correction(all_products, selection):
    
    old_products=all_products[ ~all_products['sku'].isin(selection) ]
    new_products=all_products[ all_products['sku'].isin(selection) ]

    temp=pd.DataFrame()
    temp['all_products_sales'] = all_products.groupby('date')['sales'].sum()
    temp['old_products_sales'] = old_products.groupby('date')['sales'].sum()
    temp['new_products_sales'] = new_products.groupby('date')['sales'].sum()
    temp['new_products_sales'] = temp['new_products_sales'].fillna(0)
    temp['new_product_correction_factor'] = (temp['new_products_sales']+temp['old_products_sales'])/temp['old_products_sales']
 
    old_products=old_products.merge(temp['new_product_correction_factor'] , on='date', how='left')
    
    
    old_products['sales'] = old_products['sales']*old_products['new_product_correction_factor']
    return old_products




def price_sales_correction(df):
    df_raw=df.copy()
    df_ret=df.copy()
    for sku in  df_raw['sku'].unique():
        sku_data = df_raw[df_raw['sku'] == sku]        
        df_temp = sku_data 
        
        model_ols = sm.OLS(df_temp['sales'], sm.add_constant(df_temp['price_de']), missing='drop')
        results = model_ols.fit()
        s = results.params['price_de'] 
        
        df_temp['sales_correction'] = (df_temp['price_de'] - df_temp['price_de'].mode()[0]) * s
        df_temp['price_corrected_sales'] = df_temp['sales'] - df_temp['sales_correction']        # Merge the corrected sales back to the original dataframe

        df_raw.loc[df_raw['sku'] == sku, 'price_corrected_sales'] = df_temp['price_corrected_sales'] 

        df_raw['price_corrected_sales'][df_raw['price_corrected_sales']<0]=0
        df_raw['price_corrected_sales'][(df_raw['sales']==0) & (df_raw['price_de'].isna())]=0
        df_raw['price_corrected_sales'][(df_raw['sales']>0) & (df_raw['price_de'].isna())]=df_raw['sales']
        df_ret['sales']=df_raw['price_corrected_sales']
    return df_ret


def special_days_correction(df):
    df_raw=df.copy()
    sku_list = df['sku'].unique().tolist()
    monthly_average = pd.DataFrame(df_raw.sort_values(by='month_year')['month_year'].unique().tolist(), columns=['month_year'])
    for sku in sku_list:
        average_sales_sku =df_raw[df_raw['sku'] == sku].groupby('month_year')['sales'].mean().reset_index()
        average_sales_sku.rename(columns={'sales': f'{sku}'}, inplace=True)
        monthly_average = pd.merge(monthly_average, average_sales_sku, on='month_year', how='outer')
        if not pd.isna(monthly_average[monthly_average['month_year'] == '2022-07'][sku].values[0]):
                df_raw.loc[df_raw[(df['sku'] == sku) & ((df_raw['date'] == '2022-07-12') | (df_raw['date'] == '2022-07-13'))].index, 'sales'] = monthly_average[monthly_average['month_year'] == '2022-07'][sku].values[0]
        if not pd.isna(monthly_average[monthly_average['month_year'] == '2022-11'][sku].values[0]):
                df_raw.loc[df_raw[(df['sku'] == sku) & ((df_raw['date'] == '2022-11-25') | (df_raw['date'] == '2022-11-26') | (df_raw['date'] == '2022-11-27') | (df_raw['date'] == '2022-11-28'))].index, 'sales'] = monthly_average[monthly_average['month_year'] == '2022-11'][sku].values[0]
        if not pd.isna(monthly_average[monthly_average['month_year'] == '2023-07'][sku].values[0]):
                df_raw.loc[df_raw[(df_raw['sku'] == sku) & ((df_raw['date'] == '2023-07-11') | (df_raw['date'] == '2023-07-12'))].index, 'sales'] = monthly_average[monthly_average['month_year'] == '2023-07'][sku].values[0]
    return df_raw

def corrections(df):

    st.sidebar.markdown("**Apply corrections to data**")
    
    switch_stockout_correction = st.sidebar.checkbox('Stock-out correction', help='At periods with stock-outs the sales numbers are imputed with data from the same month of the next available different year & rescaled by the monthly average.')
    if switch_stockout_correction: 
        df = stockout_correction(df)
        st.session_state['key'] = 1

    #switch_new_product_correction = st.sidebar.checkbox('New-product-correction', help='Remove products from the selection and add their sales to the remaining products.')
    #
    #if switch_new_product_correction:
    #    skus = df['sku'].unique().tolist()
    #    selected_new_skus = st.sidebar.multiselect('Which skus do you want to remove?',skus)
    #    df = new_product_correction(df, selected_new_skus)
    #    st.session_state['key'] = 1   
    
    switch_price_sales_correction = st.sidebar.checkbox('Price vs. sales correction', help='Remove the effect of price-changes from the sales numbers. I.e. a linear relationship between price and sales is assumend and the sales numbers are scaled by the factor "(current sales number)/(average sales number for the most frequent price)"')
    if switch_price_sales_correction: 
        df = price_sales_correction(df)
        st.session_state['key'] = 1
        
    switch_special_days_correction = st.sidebar.checkbox('Sale days correction', help='Remove the effect of sales days (Black Friday, Prime days etc.) from the sales by imputing the sales on these days with the daily average of the month.')
    if switch_special_days_correction:
        df = special_days_correction(df)
        st.session_state['key'] = 1
        
    return df