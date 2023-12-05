# Sales Forecast Project for SPICED Academy

This is an application for analyzing and forecasting future product sales based on historical sales numbers within the context of e-commerce.
The app uses the [XGBoost model](https://xgboost.readthedocs.io/en/stable/).

Once you have installed all required packages 

```
pip install -r requirements.txt
```

run the following command from the terminal to open the app in your default web browser:


```
streamlit run Home.py
```

Or try the [cloud version](https://automatic-sales-forecast-app.streamlit.app/)
(Please note that the XGBoost model used for forecasting will probably run slower on the (free) cloud server than on your local machine.)


At the moment, two datasets are already preloaded: Product families A and B. 

* <strong>Home</strong>: Select one of the product families.
* <strong>Time series data</strong>: View sales numbers and Warehouse stock over time. Select either all products or individual products. Select Sales/stock by product or by country (via the tab above the chart). You can also apply corrections to the data (look at the tooltips to see how the corrections impact the sales data).
* <strong>Aggregated data</strong>: View the seasonalities hidden in the sales data.
* <strong>Map</strong>: See the sales number associated with different European countries.
* <strong>Sales forecast</strong>: Here you can make 90-day sales forecasts. Select one or more products, corrections and the date from which you want to start the prediction. Everything *before* that date is used to train the forecasting model. So, if you set the *Start date for prediction* to August, 1'st, 2023 (or before), you can test the model performance against the actual sales. 


