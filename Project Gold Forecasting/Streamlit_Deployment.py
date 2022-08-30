#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib as plt
import quandl


# In[ ]:


#from annotated_text import annotated_text

header = st.container()
dataset = st.container()
model_training = st.container()


with header:
   
    st.title(" Gold Rate Forecast or Prediction ")
    #st.text('')
    
    st.image('/Users/jerryjosun/Downloads/gold-etf.jpeg')


with dataset:
    st.header("Gold dataset ")
    st.text("Found dataset from yfinance ..." )
    

   # Df = quandl.get('CHRIS/MCX_GC1',start='2006-01-01',end='2022-01-08')
    
    Df = yf.download('GLD', '2006-01-01', '2022-01-07', auto_adjust=True)
    st.write(Df.tail(10))
    
    st.subheader("History of Gold Price Series")
    close_data = pd.DataFrame(Df['Close'].value_counts()).head(20)
    #st.plot(close_data)
    
    Df = Df[['Close']]

# Drop rows with missing values
    Df = Df.dropna()

# Plot the closing price of GLD
    Df.Close.plot(figsize=(10, 7),color='r')
    

    st.bar_chart(close_data)

#with model_training:
 #   st.text(' ')

st.sidebar.header('About ')

st.sidebar.caption('Here, you will find out Gold Rate Forecast or Predictions for Today, Tomorrow & next 30 days.Also, along with it, you will also find Predictions for this Financial Year i.e. 2020-21 & next financial year i.e. 2021-22.In trading terms, the forecast and prediction help investors to understand the movement of gold in the market.It is very beneficial and tracks the operations, involvement, opening, and closing prices. It also counts profit, and losses on a daily, weekly, monthly, and yearly basis.There has been an increase in the demand and sales of gold for the past decade. The analysis clarifies the current and future condition of gold in the market via forecasts and predictions.Based on this, individual citizens, industrialists, and investors decide their next move.') 

# LinearRegression is a machine learning library for linear regression
from sklearn.linear_model import LinearRegression

# pandas and numpy are used for data manipulation
import pandas as pd
import numpy as np

# matplotlib and seaborn are used for plotting graphs
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
import yfinance as yf

# Read data
Df = yf.download('GLD', '2006-01-01', '2022-01-11', auto_adjust=True)
#Df = quandl.get('CHRIS/MCX_GC1',start='2006-01-01',end = 'current_date')#end='2022-01-22')
# Only keep close columns
Df = Df[['Close']]

# Drop rows with missing values
Df = Df.dropna()

# Plot the closing price of GLD
#Df.Close.plot(figsize=(10, 7),color='r')
#plt.ylabel("Gold ETF Prices")
#plt.title("Gold ETF Price Series")
#plt.show()

#Create histogram with density plot
#import seaborn as sns

             
             # Define explanatory variables
Df['S3'] = Df['Close'].rolling(window=3).mean()
Df['S15'] = Df['Close'].rolling(window=15).mean()
Df['next_day_price'] = Df['Close'].shift(-1)

Df = Df.dropna()
X = Df[['S3', 'S15']]

# Define dependent variable
y = Df['next_day_price']
            

# Split the data into train and test dataset
t = .8
t = int(t*len(Df))

# Train dataset
X_train = X[:t]
y_train = y[:t]

# Test dataset
X_test = X[t:]
y_test = y[t:]

# Create a linear regression model
linear = LinearRegression().fit(X_train, y_train)
#print("Linear Regression model")
print("Gold ETF Price (y) = %.2f * 3 Days Moving Average (x1) \+ %.2f * 15 Days Moving Average (x2) \+ %.2f (constant)" % (linear.coef_[0], linear.coef_[1], linear.intercept_))


predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(20, 14))
y_test.plot()
#plt.legend(['predicted_price', 'actual_price'])
#plt.ylabel("Gold ETF Price")
#plt.show()


# R square
r2_score = linear.score(X[t:], y[t:])*100
float("{0:.2f}".format(r2_score))



gold = pd.DataFrame()

gold['price'] = Df[t:]['Close']
gold['predicted_price_next_day'] = predicted_price
gold['actual_price_next_day'] = y_test
gold['gold_returns'] = gold['price'].pct_change().shift(-1)

gold['signal'] = np.where(gold.predicted_price_next_day.shift(1) < gold.predicted_price_next_day,1,0)

gold['strategy_returns'] = gold.signal * gold['gold_returns']
((gold['strategy_returns']+1).cumprod()).plot(figsize=(20,14),color='g')
plt.ylabel('Cumulative Returns')
plt.show()


sharpe = gold['strategy_returns'].mean()/gold['strategy_returns'].std()*(252**0.5)
'Sharpe Ratio %.2f' % (sharpe)



# import datetime and get today's date
import datetime as dt
current_date = dt.datetime.now()

# Get the data
data = yf.download('GLD', '2007-01-11', current_date, auto_adjust=True)
data['S3'] = data['Close'].rolling(window=3).mean()
data['S15'] = data['Close'].rolling(window=9).mean()
data = data.dropna()

# Forecast the price
data['predicted_gold_price'] = linear.predict(data[['S3', 'S15']])
data['signal'] = np.where(data.predicted_gold_price.shift(1) < data.predicted_gold_price,"Buy","No Position")

# Print the forecast

#st.dataframe(test)
data.astype(str).tail(1)[['signal','predicted_gold_price']].T
#data = data.astype(str)



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,y_train)

test_data_prediction = regressor.predict(X_test)

error_score = metrics.r2_score(y_test, test_data_prediction)
print("R squared error : ", error_score)

y_test = list(y_test)

plt.plot(y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


import pickle 
pickle_out = open("model.pkl", mode = "wb") 
pickle.dump(regressor, pickle_out) 
pickle_out.close()
pickle_in = open('model.pkl', 'rb') 
model = pickle.load(pickle_in)


@st.cache()

def prediction():

    Date = dt.datetime.now()
    
# Making predictions 
#prediction = classifier.predict([[Date]])
#if prediction < 1:
#    pred = 'BUY'
#else:
#    pred = 'HOLD'
#return pred

def main():
    
     # Here users can enter the customers data required to make prediction 
    date = st.date_input('Pick a date')
    # predicted_gold_price = st.slider('Select dayz here to predict Gold Price',min_value=0 , max_value= 30)
    result =""
        
    if st.button("Predict"): 
            result = prediction(predicted_gold_price,test_data_prediction) 
            st.success('Predicted Gold Price is {}'.format(result))
        
            s = data.astype(str).tail(1)[['signal','predicted_gold_price']].T
            st.success('Predicted Gold Price is {}'.format(s))
        
if __name__ =='__main__':      
     main()



# In[ ]:




