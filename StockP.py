
import streamlit as st
import prophet as p
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction App ')
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')

selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
#load the data
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')


st.subheader('Raw data')
st.write(data.tail())


# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()


# Predict forecast with Prophet.
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_train)

#predictions
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

#st plotly chart
st.write(f'Prediction plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)


st.title('Stock Dashboard')

tickers = ('TSLA', 'AAPL', 'MSFT' , 'BTC-USD', 'ETH-USD')

drowpdown = st.multiselect('Pick your assests', tickers);

start = st.date_input('Start',value = pd.to_datetime('2023-01-01'))
end = st.date_input('End', value= pd.to_datetime('today'))

def  relativeret(df):
    rel= df.pct_change()
    cumret =(1+rel).cumprod()-1
    cumret= cumret.fillna(0)
    return cumret

if len(drowpdown)> 0:
    df = relativeret(yf.download(drowpdown,start,end)['Adj Close'])
    st.line_chart(df)
