# Sales Forecast Project for SPICED Academy

This is an application for analyzing and forecasting future product sales based on historical sales numbers within the context of e-commerce.
The app uses the [XGBoost model](https://xgboost.readthedocs.io/en/stable/)

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


At the moment, two datasets are preloaded: Product families A and B.

