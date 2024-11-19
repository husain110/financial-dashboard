import streamlit as st
import requests
import pandas as pd
import feedparser  # For fetching news via RSS
import yfinance as yf
import financedatabase as fd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from datetime import datetime
import numpy as np
from pycoingecko import CoinGeckoAPI
from streamlit_option_menu import option_menu
import base64

# Streamlit layout config
st.set_page_config(layout="wide")

# Load the logo
logo = Image.open("logo.png")

# API key for Financial Modeling Prep
API_KEY = 'NsJaOqI9SKSeYr2q2FCJwXfkUg6IpGmB'

# API key for ExchangeRate-API 
API_KEY2 = "bbdcbe99ff9cbcf63cec4149"

@st.cache_data
def load_data():
    # Pulling list of all ETFs and Equities from financedatabase
    ticker_list = pd.concat([
        fd.ETFs().select().reset_index()[['symbol', 'name']],
        fd.Equities().select().reset_index()[['symbol', 'name']]
    ])
    ticker_list = ticker_list[ticker_list.symbol.notna()]
    ticker_list['symbol_name'] = ticker_list.symbol + " - " + ticker_list.name
    return ticker_list
# Load data once and make it globally available
ticker_list = load_data()

def get_crypto_data_with_logos():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',  # Fetch by market cap ranking
        'per_page': 100,  # Fetch 100 cryptocurrencies per page (you can adjust this as needed)
        'page': 1,  # Fetch data from the first page (you can add pagination if needed)
        'sparkline': False  # Disable sparkline for simplicity
    }
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else None


# World Indexes Data
us_indexes = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX', '^BVSP', '^GSPTSE', '^MXX', '^NYA', '^NDX', '^SPX']
europe_indexes = ['^FTSE', '^GDAXI', '^FCHI', '^N100', '^STOXX50E', '^MSEURO', '^EUR', '^AEX', '^SSMI', '^ATHEX', '^IBEX', '^OMXS30']
asia_indexes = ['^N225', '^HSI', '^AXJO', '^STI', '^KS11', '^BSESN', '^JPXNK400', '^AORD', '^SSEC', '^JKSE', '^TWII', '^NZ50', '^KSE100']

# Function to fetch real-time index data
def fetch_indexes(indexes):
    data = []
    for index in indexes:
        ticker = yf.Ticker(index)
        hist = ticker.history(period="1d")
        if not hist.empty:
            name = ticker.info.get('shortName', index)
            price = hist['Close'][0]
            change_percent = (hist['Close'][0] - hist['Open'][0]) / hist['Open'][0] * 100
            data.append({
                'name': name,
                'price': price,
                'change_percent': change_percent
            })
    return data

# Function to display stock data as cards
def display_cards(data, region_name):
    st.markdown(f"### {region_name} Market")
    card_content = ""
    for stock in data:
        price = f"{stock['price']:.2f}"
        change_class = "positive" if stock['change_percent'] >= 0 else "negative"
        card_content += f'''
        <div class="card">
            <h3 class="card-title">{stock['name']}</h3>
            <p class="card-price">${price}</p>
            <p class="card-change {change_class}">{stock['change_percent']:.2f}% Change</p>
        </div>
        '''
    st.markdown(f"""
        <div class="flex-container">
            {card_content}
        </div>
    """, unsafe_allow_html=True)

# Function to fetch top gainers data
def get_top_gainers():
    url = f"https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey={API_KEY}"
    response = requests.get(url)
    return response.json()

# Function to fetch top losers data
def get_top_losers():
    url = f"https://financialmodelingprep.com/api/v3/stock_market/losers?apikey={API_KEY}"
    response = requests.get(url)
    return response.json()

# Function to fetch highest volume data
def get_highest_volume():
    url = f"https://financialmodelingprep.com/api/v3/stock_market/actives?apikey={API_KEY}"
    response = requests.get(url)
    return response.json()

def display_stock_card(stock_data):
    card_content = ""
    for stock in stock_data:
        try:
            logo_url = f"https://logo.clearbit.com/{yf.Ticker(stock['symbol']).info['website'].replace('https://www.', '')}"
        except:
            logo_url = "https://via.placeholder.com/50"  # Placeholder image if no logo found
        card_content += f'''
        <div class="card">
            <img src="{logo_url}" alt="Logo" style="width: 50px; height: auto; margin-bottom: 10px;">
            <h3 class="card-title">{stock.get('name', 'No Name')}</h3>
            <p class="card-price">${stock.get('price', 'N/A')}</p>
            <p class="card-change change-positive">{stock.get('changesPercentage', 'N/A')}% Gain</p>
            <p class="card-change change-negative">${stock.get('change', 'N/A')} Change</p>
        </div>
        '''
    st.markdown(f"""
        <div class="flex-container">
            {card_content}
        </div>
    """, unsafe_allow_html=True)

# Initialize the CoinGecko API
cg = CoinGeckoAPI()

# Function to calculate Fear & Greed Index for both stocks and crypto
def calculate_fear_greed_index(close_price, volume):
    # Placeholder logic for calculating Fear & Greed Index
    index = (close_price + volume) % 100  # Simplified mock calculation
    return index.item() if isinstance(index, np.ndarray) else index

# Function to display the Fear & Greed Index gauge
def display_fear_greed_index(index):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=index,
        title={'text': "Fear & Greed Index"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 25], 'color': "red"},
                {'range': [25, 50], 'color': "orange"},
                {'range': [50, 75], 'color': "lightgreen"},
                {'range': [75, 100], 'color': "green"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 80}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def get_crypto_data(ticker):
    url = f"https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'ids': ticker.lower()
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if data:
        # CoinGecko provides logo in the 'image' field for each crypto
        price = data[0]['current_price']
        volume = data[0]['total_volume']
        logo_url = data[0]['image']
        return price, volume, logo_url
    else:
        raise ValueError(f"No data found for {ticker}")

# Function to get logo for stocks and crypto
def get_logo_url(ticker, asset_type):
    if asset_type == 'Crypto':
        _, _, logo_url = get_crypto_data(ticker)
    else:
        # For stocks, use Clearbit based on website
        try:
            logo_url = f"https://logo.clearbit.com/{yf.Ticker(ticker).info['website'].replace('https://www.', '')}"
        except:
            logo_url = "https://via.placeholder.com/50"
    return logo_url


# Batch fetch function for multiple symbols
def fetch_batch_data(symbols, endpoint):
    symbols_str = ','.join(symbols)
    url = f'https://financialmodelingprep.com/api/v3/{endpoint}/{symbols_str}?apikey={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None


    
# Load external CSS
with open('test/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with open("test\small_logo.png", "rb") as image_file:
    logo_base64 = base64.b64encode(image_file.read()).decode()

# Display the main title with the small logo
st.markdown(
    f"""
    <div class="title-container">
        <h1 class="dashboard-title">Real-Time Financial Dashboard</h1>
        <img src="data:image/png;base64,{logo_base64}" class="dashboard-logo">
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header('')
    mode = option_menu(
        menu_title=None,
        options=['Home', 
                 'Real-Time Stock Data', 
                 'Portfolio Analysis', 
                 'Dividend Tracking', 
                 'World Market Overview',
                 'Fear & Greed Index' , 
                 'Currency Converter'], # Menu options
        icons=['house','graph-up', 'briefcase', 'coin', 'globe', 'star', 'currency-exchange' ], # Icons for each option
        menu_icon="cast",        # Menu icon for the sidebar
        default_index=0,         # Starting mode
        styles={
            "container": {"padding": "0!important", "background-color": ""},
            "icon": {"color": "white", "font-size": "20px"}, 
            "nav-link": {"font-size": "18px", "text-align": "left", "margin": "0px", "--hover-color": "#555555"},
            "nav-link-selected": {"background-color": "#008CBA"},
        }
    )

if mode == 'Real-Time Stock Data':
    st.markdown('<div class="section-title">📈 Real-Time Stock Data</div>', unsafe_allow_html=True)
    ticker = st.text_input('Ticker', 'AAPL')

    # Button to update chart and news
    if st.button('Update'):
        if ticker:
            # Create tabs for "Live Chart" and "Live News"
            tab1, tab2 = st.tabs(["Live Chart", "Live News"])

            with tab1:
                st.markdown(f'<div class="section-title">Live TradingView Chart for {ticker.upper()}</div>', unsafe_allow_html=True)
                
                # TradingView widget for stock chart
                tradingview_widget = f"""
                <iframe
                src="https://s.tradingview.com/widgetembed/?symbol={ticker.upper()}&interval=D&theme=light&style=2&toolbar=1&withdateranges=1&allow_symbol_change=1&saveimage=1&studies=[]&hideideas=1&widgetbar=show"
                width="100%" height="610" frameborder="0" allowfullscreen></iframe>
                """
                st.markdown(tradingview_widget, unsafe_allow_html=True)

            with tab2:
                st.markdown(f'<div class="section-title">Latest News for {ticker.upper()}</div>', unsafe_allow_html=True)

                # Fetch stock news using Yahoo Finance RSS feed
                try:
                    rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker.upper()}"
                    news_feed = feedparser.parse(rss_url)
                    
                    if len(news_feed.entries) > 0:
                        for entry in news_feed.entries[:10]:  # Displaying the top 10news
                            st.markdown(f"""
                            <div class="news-card">
                                <div class="news-title">{entry.title}</div>
                                <a href="{entry.link}" target="_blank" class="news-link">Read full article</a>
                                <div class="news-date">Published: {entry.published}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.write("No news found.")

                except Exception as e:
                    st.error(f"Failed to fetch news: {str(e)}")



elif mode == 'Portfolio Analysis':
    st.markdown('<div class="section-title">💼 Portfolio Analysis</div>', unsafe_allow_html=True)
    
    # Load ticker data
    ticker_list = load_data()

    # Portfolio Builder in the main page
    sel_tickers = st.multiselect(
        "",
        placeholder="Search tickers",
        options=ticker_list.symbol_name
    )

    # Filter selected tickers
    sel_tickers_list = ticker_list[ticker_list.symbol_name.isin(sel_tickers)].symbol

    # Date selector in the main page
    sel_dt1 = st.date_input('Start Date', value=pd.Timestamp(2024, 1, 1), format="YYYY-MM-DD")
    sel_dt2 = st.date_input('End Date', format="YYYY-MM-DD")

    # If there are selected tickers, fetch and process data
    if len(sel_tickers) > 0:
        # Fetching Stock Data and Calculating Metrics
        tickers = sel_tickers_list.tolist() if isinstance(sel_tickers_list, pd.Series) else sel_tickers_list
        yfdata = yf.download(tickers, start=sel_dt1, end=sel_dt2)['Close'].reset_index()

        # Transform data
        yfdata = yfdata.melt(id_vars=["Date"], var_name="ticker", value_name="price")
        yfdata['price_start'] = yfdata.groupby('ticker').price.transform('first')
        yfdata['price_pct'] = ((yfdata['price'] - yfdata['price_start']) / yfdata['price_start']) * 100


    # Tabs - Portfolio and Calculator
    tab1, tab2 = st.tabs(['Portfolio', 'Calculator'])

    # Tab 1: Portfolio Analysis
    with tab1:
        if len(sel_tickers) > 0:
            # All stocks plot
            st.subheader('All Stocks')
            fig = px.line(yfdata, x='Date', y='price_pct', color='ticker', markers=True)
            fig.add_hline(y=0, line_dash="dash", line_color="white")
            fig.update_layout(xaxis_title=None, yaxis_title=None)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

            # Individual stock plots
            st.subheader('Individual Stock')
            cols = st.columns(3)
            for i, ticker in enumerate(sel_tickers_list):
                # Adding logo
                try:
                    cols[i % 3].image(f"https://logo.clearbit.com/{yf.Ticker(ticker).info['website'].replace('https://www.', '')}", width=65)
                except:
                    cols[i % 3].subheader(ticker)

                # Stock metrics
                cols2 = cols[i % 3].columns(3)
                ticker_label = 'Close' if len(sel_tickers_list) == 1 else ticker
                cols2[0].metric(label="50-Day Average", value=round(yfdata[yfdata.ticker == ticker_label]['price'].tail(50).mean(), 2))
                cols2[1].metric(label="1-Year Low", value=round(yfdata[yfdata.ticker == ticker_label]['price'].min(), 2))
                cols2[2].metric(label="1-Year High", value=round(yfdata[yfdata.ticker == ticker_label]['price'].max(), 2))

                # Stock plot
                fig = px.line(yfdata[yfdata.ticker == ticker], x='Date', y='price', markers=True)
                fig.update_layout(xaxis_title=None, yaxis_title=None)
                cols[i % 3].plotly_chart(fig, use_container_width=True)
        st.write("Portfolio Content")

    # Tab 2: Calculator
    with tab2:
        if len(sel_tickers) > 0:
            # Initialize amounts dictionary to store the investment per ticker
            amounts = {}
            total_inv = 0  # To hold the total investment

            # Input investment amounts for each ticker
            for i, ticker in enumerate(sel_tickers_list):
                cols = st.columns((0.2, 0.4, 0.4))  # Adjust column widths for better alignment

                # Display company logo
                try:
                    cols[0].image(f"https://logo.clearbit.com/{yf.Ticker(ticker).info['website'].replace('https://www.', '')}", width=65)
                except:
                    cols[0].subheader(ticker)

                # Display the label for investment
                cols[1].write(f"Investment for {ticker}")

                # Number input for investment amount
                amount = cols[2].number_input("", key=ticker, step=50)
                total_inv += amount
                amounts[ticker] = amount

            # Display total investment
            st.write(f"**Total Investment:** {total_inv}")

            # Goal input
            goal = st.number_input('Goal:', key='goal', step=50)

            # Prepare data for plotting the goal-related chart if yfdata exists
            if 'yfdata' in globals():
                df = yfdata.copy()
                df['amount'] = df.ticker.map(amounts) * (1 + df.price_pct)

                # Group data by date to sum the investment
                dfsum = df.groupby('Date').amount.sum().reset_index()

                # Create plot
                fig = px.area(dfsum, x='Date', y='amount')
                fig.add_hline(y=goal, line_color='rgb(255,255,255)', line_dash='dash', line_width=3)

                # Check if the goal is met within the time frame
                if dfsum[dfsum.amount >= goal].shape[0] == 0:
                    st.warning("The goal can't be reached within this time frame. Either change the goal amount or the time frame.")
                else:
                    goal_date = dfsum[dfsum.amount >= goal].Date.iloc[0]
                    fig.add_vline(x=goal_date, line_color='rgb(255,255,255)', line_dash='dash', line_width=3)
                    fig.add_trace(go.Scatter(
                        x=[goal_date + pd.DateOffset(days=7)],
                        y=[goal * 1.1],
                        text=f'Goal reached on {goal_date.strftime("%Y-%m-%d")}',
                        mode='text',
                        name='Goal',
                        textfont=dict(color='rgb(255,255,255)', size=20)
                    ))

                fig.update_layout(xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)
        st.write("Calculator Content")
 
# Function to get dividend data
def get_dividend_data(ticker):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{ticker}?apikey={API_KEY}"
    response = requests.get(url)
    
    # Check if the response is valid
    if response.status_code == 200:
        data = response.json()
        if 'historical' in data:
            return pd.DataFrame(data['historical'])
        else:
            st.error(f"No 'historical' data found for {ticker}.")
            return pd.DataFrame()
    else:
        st.error(f"Error fetching dividend data for {ticker}. Status Code: {response.status_code}")
        return pd.DataFrame()

def dividend_tracking():
    st.markdown('<div class="section-title">💸 Dividend Tracking</div>', unsafe_allow_html=True)
    
    # User Inputs on Main Page
    ticker = st.text_input("Enter Stock Ticker:").upper()
    num_shares = st.number_input("Enter Number of Shares:", min_value=1)
    purchase_date = st.date_input("Enter Purchase Date (YYYY-MM-DD):", value=datetime.now(), min_value=datetime(1980, 1, 1), max_value=datetime.now())
    
    if ticker:
        # Fetch company info from Yahoo Finance
        try:
            company_info = yf.Ticker(ticker)
            logo_url = f"https://logo.clearbit.com/{company_info.info['website'].replace('https://www.', '')}"
            company_name = company_info.info['shortName']
        except Exception as e:
            st.error(f"Error fetching company info: {e}")
            return
        
        # Fetch dividend data
        dividend_data = get_dividend_data(ticker)
        
        if not dividend_data.empty:
            # Filter dividends based on the purchase date
            dividend_data['date'] = pd.to_datetime(dividend_data['date'])
            filtered_dividends = dividend_data[dividend_data['date'] >= pd.to_datetime(purchase_date)]
            
            if filtered_dividends.empty:
                st.write(f"No dividends received for {ticker} since {purchase_date}.")
            else:
                # Calculate total dividend income
                filtered_dividends['total_dividend'] = filtered_dividends['dividend'] * num_shares
                total_income = filtered_dividends['total_dividend'].sum()

                # Display the company logo and total dividend income together
                st.markdown(f'''
                    <div class="company-card">
                        <img src="{logo_url}" width="50" class="company-logo">
                        <div class="company-info">
                            <h2>{company_name}</h2>
                            <h3>{ticker} - Total Dividend Income: ${total_income:.2f}</h3>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

                # Display the filtered dividend data in a centered table
                st.markdown('<div class="centered-content">', unsafe_allow_html=True)
                st.write(filtered_dividends[['date', 'dividend']])
                st.markdown('</div>', unsafe_allow_html=True)

# Check which mode is selected
if mode == 'Dividend Tracking':
    dividend_tracking()  # Call the dividend tracking function



if mode == 'World Market Overview':
    st.markdown('<div class="section-title">🌐 World Market Overview</div>', unsafe_allow_html=True)
    tabs = st.tabs(["World Indexes", "Top Stocks", "Top Cryptocurrencies", "Top Gainers", "Top Losers", "Market Highest Volume"])

    # Tab for World Indexes
    with tabs[0]:
        st.header("World Indexes")
        us_market_data = fetch_indexes(us_indexes)
        europe_market_data = fetch_indexes(europe_indexes)
        asia_market_data = fetch_indexes(asia_indexes)
        display_cards(us_market_data, "US")
        display_cards(europe_market_data, "Europe")
        display_cards(asia_market_data, "Asia")

    # Tab for Top Stocks
    with tabs[1]:
        st.header("Top 50 Stocks")
        symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'V', 'JNJ', 'WMT', 'JPM', 'PG', 'MA', 'DIS', 
                   'UNH', 'HD', 'PYPL', 'NFLX', 'ADBE', 'INTC', 'KO', 'PFE', 'NKE', 'MRK', 'VZ', 'XOM', 'CSCO', 'PEP', 
                   'COST', 'CVX', 'T', 'ABT', 'CRM', 'LLY', 'BAC', 'ORCL', 'ABBV', 'AVGO', 'QCOM', 'AMD', 'MDT', 'MCD', 
                   'HON', 'NEE', 'MS', 'RTX', 'IBM', 'UNP', 'SPGI', 'TXN']
        stock_data = fetch_batch_data(symbols, 'quote')

        if stock_data:
            cols = st.columns(4)
            for i, stock in enumerate(stock_data):
                logo_url = f"https://logo.clearbit.com/{yf.Ticker(stock['symbol']).info['website'].replace('https://www.', '')}"
                with cols[i % 4]:
                    st.markdown(f"""
                        <div class="stock-card">
                            <img src="{logo_url}" alt="Logo">
                            <div class="stock-name">{stock['name']}</div>
                            <div class="stock-price">${stock['price']:.2f}</div>
                            <div class="stock-change">{stock['changesPercentage']:.2f}% Change</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Error fetching stock data.")

    # Tab for Top Cryptocurrencies
    with tabs[2]:
        st.header("Top Cryptocurrencies")
        
        # Fetch crypto data with logos from CoinGecko API
        crypto_data = get_crypto_data_with_logos()

        if crypto_data:
            cols = st.columns(4)
            for i, crypto in enumerate(crypto_data):
                logo_url = crypto['image']
                with cols[i % 4]:
                    st.markdown(f"""
                        <div class="crypto-card">
                            <img src="{logo_url}" alt="{crypto['name']} logo" style="width:50px; height:auto; margin-bottom:10px;">
                            <div class="crypto-name">{crypto['name']} USD</div>
                            <div class="crypto-price">${crypto['current_price']:.2f}</div>
                            <div class="crypto-change">{crypto['price_change_percentage_24h']:.2f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Error fetching cryptocurrency data.")

    # Tab for Top Gainers
    with tabs[3]:
        st.header("Top Gainers")
        stock_data = get_top_gainers()
        display_stock_card(stock_data)

    # Tab for Top Losers
    with tabs[4]:
        st.header("Top Losers")
        stock_data = get_top_losers()
        display_stock_card(stock_data)

    # Tab for Market Highest Volume
    with tabs[5]:
        st.header("Market Highest Volume")
        stock_data = get_highest_volume()
        display_stock_card(stock_data)

# Separate mode for Fear & Greed Index
elif mode == 'Fear & Greed Index':
    st.markdown('<div class="section-title">📊 Fear & Greed Index</div>', unsafe_allow_html=True)
    asset_type = st.selectbox("Select Asset Type", ['Stocks', 'Crypto'])
    ticker = st.text_input("Enter Stock Ticker or Crypto Symbol:")

    if st.button("Calculate Fear & Greed Index"):
        if ticker:
            try:
                # Display the logo for the asset
                logo_url = get_logo_url(ticker.lower(), asset_type)
                st.image(logo_url, width=100)

                if asset_type == 'Crypto':
                    # Fetching crypto data from CoinGecko
                    price, volume = get_crypto_data(ticker)
                else:
                    # Fetching stock data from Yahoo Finance
                    data = yf.download(ticker, period='1d')
                    if data.empty:
                        raise ValueError(f"No data available for {ticker.upper()}.")
                    price = data['Close'].values[-1]
                    volume = data['Volume'].values[-1]

                # Calculate Fear & Greed Index
                fear_greed_index = calculate_fear_greed_index(price, volume)

                # Display stock/crypto details
                st.subheader(f'{ticker.upper()} Data')
                if asset_type == 'Stocks':
                    st.table(data.tail(1))
                else:
                    st.write(f"Price: ${price}, Volume: {volume}")

                # Display Fear & Greed Index
                display_fear_greed_index(fear_greed_index)
            except Exception as e:
                st.error(f"Error fetching data for {ticker.upper()}: {e}")
        else:
            st.warning("Please enter a valid ticker.")


if mode == 'Currency Converter':
    st.markdown('<div class="section-title">💱 Currency Converter</div>', unsafe_allow_html=True)

    # Function to get conversion rate
    def get_conversion_rate(base_currency, target_currency):
        url = f"https://v6.exchangerate-api.com/v6/{API_KEY2}/pair/{base_currency}/{target_currency}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'conversion_rate' in data:
                return data['conversion_rate']
            else:
                st.error("Invalid currency pair.")
        else:
            st.error("Failed to fetch data from the API.")
        return None

    # Main app function
    def currency_converter():

        # User input for base and target currencies
        base_currency = st.selectbox("Select Base Currency", ['USD', 'EUR', 'INR', 'GBP', 'JPY', 'AUD', 'CAD'])
        target_currency = st.selectbox("Select Target Currency", [ 'EUR','USD', 'INR', 'GBP', 'JPY', 'AUD', 'CAD'])

        # User input for amount to convert
        amount = st.number_input(f"Enter amount in {base_currency}:", min_value=0.0)

        # Convert button
        if st.button("Convert"):
            if base_currency and target_currency and amount > 0:
                rate = get_conversion_rate(base_currency, target_currency)
                if rate:
                    converted_amount = rate * amount

                    # Display the result in a card format using custom HTML
                    st.markdown(f"""
                        <div style="background-color:whitesmoke;padding:20px;border-radius:10px;width:50%;margin:auto;text-align:center;">
                            <h3 style="color:white;">💱</h3>
                            <p style="color:black;font-size:20px;"><b>{amount} {base_currency}</b> = <b>{converted_amount:.2f} {target_currency}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("Please enter valid inputs.")

    # Calling the currency converter function
    currency_converter()


import base64

# Load the logo and convert it to base64
with open("test\logo.png", "rb") as image_file:
    logo_base64 = base64.b64encode(image_file.read()).decode()

# Display the logo on the Home section
if mode == 'Home':

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{logo_base64}" width="900">
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Add any other home content here
else:
    # Other sections logic
    pass
