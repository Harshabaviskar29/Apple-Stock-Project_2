#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import streamlit as st 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pickle import dump
from pickle import load
import warnings
warnings.filterwarnings('ignore')


# In[12]:


st.title('Model Deployment: Time Serise Forcasting')


# In[13]:


st.sidebar.header('User Input Period Of Forcasting')


# def user_input_features():
#     Start_Date = st.sidebar.date_input("Enter Start Date For Prediction")
#     End_Date = st.sidebar.date_input("Enter Start End For Prediction")
#     data = {'Start_Date':Start_Date,
#             'End_Date':End_Date}
#     
#     features = pd.DataFrame(data,index = [0])
#     return features 
#     
# df = user_input_features()
# st.subheader('User Input For Stock Prediction')
# st.write(df)

# In[14]:


Stock_df = load(open('Stock_Pred.sav', 'rb'))


# In[15]:


Days_pred = st.number_input('Number of Days',min_value=1)


# In[16]:


datetime = pd.date_range('2020-01-01', periods=Days_pred,freq='B')
date_df = pd.DataFrame(datetime,columns=['Date'])


# In[18]:


model_sarima_final = sm.tsa.SARIMAX(Stock_df.Close,order=(0,1,2),seasonal_order=(1,1,0,57))
sarima_fit_final = model_sarima_final.fit()
forecast = sarima_fit_final.predict(len(Stock_df),len(Stock_df)+Days_pred-1)
forecast_df = pd.DataFrame(forecast)
forecast_df.columns = ['Stock Price']


# In[20]:


st.subheader('Predicted Price For Stock')


# In[19]:


data_forecast = forecast_df.set_index(date_df.Date)
st.success('Forecasting stock price value for '+str(Days_pred)+' days')
st.write(data_forecast)


# In[21]:


st.subheader('Graphical Presentation Of Predicted Price')


# In[24]:


fig,ax = plt.subplots(figsize=(16,8),dpi=100)
ax.plot(Stock_df, label='Actual')
ax.plot(data_forecast,label='Forecast')
ax.set_title('Apple Stock Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.legend(loc='upper left',fontsize=12)
ax.grid(True)
st.pyplot(fig)

