import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import openai
from scipy.stats import linregress

# ---------------------------
# Helper Functions for Indicators
# ---------------------------
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, span_short=12, span_long=26, signal_span=9):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    return macd, signal

# ---------------------------
# Fetch Historical Data from Polygon.io
# ---------------------------
def fetch_stock_data(ticker, polygon_api_key, days_back=120):
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start_date}/{end_date}?apiKey={polygon_api_key}"
    )
    try:
        response = requests.get(url)
        data_json = response.json()
        if "results" in data_json:
            df = pd.DataFrame(data_json["results"])
            df["t"] = pd.to_datetime(df["t"], unit="ms")
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# ---------------------------
# Analyze VCP Pattern with Enhanced Technical Indicators
# ---------------------------
def analyze_vcp_pattern(data, lookback=15):
    if len(data) < lookback:
        return 0, {}
    data = data.sort_values("t")
    recent = data.tail(lookback)
    days = np.arange(len(recent))
    
    # Linear regression on volume over the lookback period
    slope, intercept, r_value, p_value, std_err = linregress(days, recent['v'])
    vol_contraction = slope < 0
    
    # Price consolidation: trading range is less than 5% of the average price
    price_range = recent['c'].max() - recent['c'].min()
    avg_price = recent['c'].mean()
    consolidation = (price_range / avg_price) < 0.05
    
    # Bullish trend: current close > 20-day moving average
    if len(data) >= 20:
        ma20 = data['c'].rolling(window=20).mean()
        bullish_trend = recent['c'].iloc[-1] > ma20.iloc[-1]
    else:
        bullish_trend = False
        ma20 = pd.Series([np.nan]*len(data))
    
    # Additional technical indicators
    rsi_series = compute_rsi(data['c'], window=14)
    current_rsi = rsi_series.iloc[-1] if not rsi_series.empty else np.nan
    
    macd, macd_signal = compute_macd(data['c'])
    current_macd = macd.iloc[-1] if not macd.empty else np.nan
    current_macd_signal = macd_signal.iloc[-1] if not macd_signal.empty else np.nan
    
    volatility = recent['c'].std()
    
    # Heuristic scoring
    score = 0
    if vol_contraction:
        score += 30
    if consolidation:
        score += 20
    if bullish_trend:
        score += 30
    if 40 <= current_rsi <= 60:
        score += 10
    if current_macd > current_macd_signal:
        score += 10
    
    probability = min(score, 100)
    details = {
        "vol_slope": slope,
        "consolidation": consolidation,
        "bullish_trend": bullish_trend,
        "current_rsi": current_rsi,
        "current_macd": current_macd,
        "current_macd_signal": current_macd_signal,
        "volatility": volatility,
        "base_score": score,
        "lookback_period": lookback,
        "price_range_pct": (price_range / avg_price) * 100,
        "current_close": recent['c'].iloc[-1],
        "ma20": ma20.iloc[-1] if len(data) >= 20 else None
    }
    return probability, details

# ---------------------------
# Calculate Trade Levels
# ---------------------------
def calculate_trade_levels(entry_price, risk_reward_ratio=3, risk_pct=0.02):
    stop_loss = entry_price * (1 - risk_pct)
    profit_target = entry_price + (entry_price - stop_loss) * risk_reward_ratio
    return stop_loss, profit_target

# ---------------------------
# OpenAI API Call using updated client syntax
# ---------------------------
def call_openai_assessment(ticker, summary_text, openai_api_key):
    # Initialize the OpenAI client using the updated client syntax
    client = openai.OpenAI(api_key=openai_api_key)
    
    best_prompt = (
        "You are a seasoned expert in swing trading and technical analysis specializing in Volume Contraction Pattern (VCP) setups. "
        "Evaluate the following technical analysis summary for a stock, which includes data on volume contraction over the last 15 days, "
        "price consolidation metrics (where the trading range is less than 5% of the average price), RSI, MACD readings, and a bullish trend "
        "confirmation via a 20-day moving average. Additionally, consider any provided metrics such as percentage changes, volatility measures, "
        "and momentum indicators that may indicate whether the observed contraction is statistically significant and part of a longer-term "
        "consolidation phase. Based solely on this detailed information, return only a single number between 0 and 100 representing the "
        "probability that the stock is in a valid VCP setup likely to result in a bullish breakout. Do not include any additional commentary or explanation."
    )
    messages = [
        {"role": "system", "content": best_prompt},
        {"role": "user", "content": summary_text}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.0,
            max_tokens=10,
        )
        result_text = response.choices[0].message.content.strip()
        probability = float(result_text)
        return probability
    except Exception as e:
        st.error(f"OpenAI API error for {ticker}: {e}")
        return None

# ---------------------------
# Main Streamlit Application
# ---------------------------
def main():
    st.title("VCP Swing Trade Analyzer")
    st.markdown(
        "This app analyzes stocks for Volume Contraction Pattern (VCP) setups using Polygon.io market data and leverages OpenAI for an AI-assisted probability assessment. "
        "The analysis includes additional technical indicators such as RSI, MACD, and volatility to enhance detection of true VCP setups."
    )
    
    # Retrieve API keys from Streamlit secrets
    try:
        polygon_api_key = st.secrets["POLYGON_API_KEY"]
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except Exception as e:
        st.error("API keys not found in st.secrets. Please add POLYGON_API_KEY and OPENAI_API_KEY to your secrets.toml file.")
        return

    st.sidebar.header("Trading Settings")
    risk_pct = st.sidebar.number_input("Risk per Trade (%)", value=2.0) / 100.0
    risk_reward_ratio = st.sidebar.number_input("Risk/Reward Ratio", value=3.0)
    
    st.header("Upload TradingView Watchlist")
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing your watchlist (must have a 'Ticker' column)",
        type="csv"
    )
    
    if uploaded_file is not None:
        watchlist_df = pd.read_csv(uploaded_file)
        if "Ticker" not in watchlist_df.columns:
            st.error("CSV file must contain a 'Ticker' column.")
            return
        tickers = watchlist_df["Ticker"].unique().tolist()
        st.write("Tickers found:", tickers)
        
        results = []
        progress_bar = st.progress(0)
        for i, ticker in enumerate(tickers):
            st.write(f"Processing {ticker}...")
            data = fetch_stock_data(ticker, polygon_api_key)
            if data.empty:
                st.write(f"No data available for {ticker}. Skipping.")
                continue
            
            probability, analysis_details = analyze_vcp_pattern(data, lookback=15)
            entry_price = data.sort_values("t")["c"].iloc[-1]
            stop_loss, profit_target = calculate_trade_levels(entry_price, risk_reward_ratio, risk_pct)
            
            summary_text = (
                f"Volume slope over last {analysis_details.get('lookback_period', 15)} days: {analysis_details.get('vol_slope', 0):.2f}. "
                f"Price consolidation: {'Yes' if analysis_details.get('consolidation', False) else 'No'} "
                f"(Range: {analysis_details.get('price_range_pct', 0):.2f}% of average). "
                f"Bullish trend: {'Yes' if analysis_details.get('bullish_trend', False) else 'No'} "
                f"(Current Close: {analysis_details.get('current_close', 0):.2f}, 20-day MA: {analysis_details.get('ma20', 'N/A')}). "
                f"RSI: {analysis_details.get('current_rsi', 0):.2f}. "
                f"MACD: {analysis_details.get('current_macd', 0):.2f} vs Signal: {analysis_details.get('current_macd_signal', 0):.2f}. "
                f"Volatility (std dev): {analysis_details.get('volatility', 0):.2f}. "
                f"Base Score: {analysis_details.get('base_score', 0)}."
            )
            
            if openai_api_key:
                ai_probability = call_openai_assessment(ticker, summary_text, openai_api_key)
                if ai_probability is not None:
                    probability = (probability + ai_probability) / 2
            
            result = {
                "Ticker": ticker,
                "Probability (%)": round(probability, 2),
                "Entry Price": round(entry_price, 2),
                "Stop Loss": round(stop_loss, 2),
                "Profit Target": round(profit_target, 2)
            }
            results.append(result)
            progress_bar.progress((i + 1) / len(tickers))
        
        if results:
            results_df = pd.DataFrame(results)
            st.header("Analysis Results")
            st.dataframe(results_df)
        else:
            st.write("No valid results to display.")

if __name__ == "__main__":
    main()


