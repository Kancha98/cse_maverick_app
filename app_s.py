import streamlit as st
import pandas as pd
import psycopg2
import os
import urllib.parse as urlparse
import plotly.express as px
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv(dotenv_path=r"E:\Side Projects\CSE bot\myenv\streamlit\myenv\Scripts\.env")
load_dotenv()

# --- Database Connection ---
@st.cache_resource
def init_connection():
    # This line will now directly get the secret from Streamlit Cloud's environment
    db_url = os.environ.get("NEON_DB_URL")
    if not db_url:
        # This error will trigger if the secret is not set in Streamlit Cloud
        st.error("Database URL not found in environment variables! Please configure Streamlit Secrets.")
        return None
    url = urlparse.urlparse(db_url)
    return psycopg2.connect(
        database=url.path[1:],
        user=url.username,
        password=url.password,
        host=url.hostname,
        port=url.port or "5432"
    )

# --- Load Data ---
@st.cache_data(ttl=600)
def load_data():
    conn = init_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM stock_analysis_all_results;")
        colnames = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=colnames)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

# --- Streamlit App ---
st.title("📈 CSE Gem Finder by CSE Maverick")
st.markdown("💡An intelligent assistant to help you discover high-potential stocks..")

def get_mavericks_picks(results_df):
    """Filters stocks for Mavericks Picks based on Tier 1 and Tier 2 conditions."""
    # Ensure numeric columns are properly typed
    for col in ['turnover', 'volume', 'relative_strength']:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0)
    
    
    tier_1_conditions = (
        (results_df['rsi_divergence'] == "Bullish Divergence") &
        (results_df['volume_analysis'].isin(["Emerging Bullish Momentum", "Increase in weekly Volume Activity Detected"]))
    ) & (
        (results_df['turnover'] > 999999) &
        (results_df['volume'] > 9999) &
        (results_df['relative_strength'] >= 1)
    )

    tier_2_conditions = (
        (results_df['volume_analysis'].isin(["Emerging Bullish Momentum", "High Bullish Momentum"]))&
        (results_df['turnover'] > 999999) &
        (results_df['volume'] > 9999) &
        (results_df['relative_strength'] >= 1)
    ) | (
        (results_df['rsi_divergence'] == "Bullish Divergence")&
        (results_df['turnover'] > 999999) &
        (results_df['volume'] > 9999) 

    )

    tier_1_picks = results_df[tier_1_conditions]
    tier_2_picks = results_df[tier_2_conditions]

    return tier_1_picks, tier_2_picks

try:
    df = load_data()

    if df.empty:
        st.warning("No data found in the table.")
        st.stop()

    # Remove unwanted columns
    df = df.drop(columns=[col for col in ['id'] if col in df.columns])
    
    # Rename headers
    #df.columns = [col.replace('_', ' ').title() for col in df.columns]
    
    # Format numeric values with commas
    #for col in df.select_dtypes(include=['float64', 'int64']).columns:
    #    df[col] = df[col].apply(lambda x: f"{x:,.2f}" if isinstance(x, float) else f"{x:,}")


    # === Display Filtered Table ===
    #st.subheader("📄 Filtered Analysis Results")
    #st.dataframe(df, use_container_width=True)

    # === Filters Section ===
    st.markdown("### 🔍 Apply Filters")

    # Dropdown filters
    selected_symbol = st.selectbox("Select Symbol", options=["All"] + list(df['symbol'].unique()))
    selected_divergence = st.selectbox("Select Divergence Check", options=["All"] + list(df['rsi_divergence'].dropna().unique()))
    selected_volume_analysis = st.selectbox("Select Volume Analysis", options=["All"] + list(df['volume_analysis'].dropna().unique()))
    
    # Turnover ranges
    turnover_ranges = {
        "100K-1M": (100000, 1000000),
        "1M-10M": (1000000, 10000000),
        "10M-100M": (10000000, 100000000),
        "100M+": (100000000, float('inf'))
    }
    selected_turnover_ranges = st.multiselect(
        "Select Turnover Ranges",
        options=list(turnover_ranges.keys()),
        default=["100K-1M", "1M-10M"]
    )

    # Range sliders
    rsi_range = st.slider("RSI Range", float(df['rsi'].min()), float(df['rsi'].max()), (30.0, 70.0))
    date_range = st.slider(
        "Date Range",
        min_value=df['last_updated'].min().date(),
        max_value=df['last_updated'].max().date(),
        value=(df['last_updated'].min().date(), df['last_updated'].max().date())
    )
    
    # EMA Checker
    st.markdown("### EMA Checker")
    ema_20_check = st.checkbox("Price Above EMA 20")
    ema_50_check = st.checkbox("Price Above EMA 50")
    ema_100_check = st.checkbox("Price Above EMA 100")
    ema_200_check = st.checkbox("Price Above EMA 200")
    
    

    # Apply filters
    filtered_df = df.copy()
    if selected_symbol != "All":
        filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]
    if selected_divergence != "All":
        filtered_df = filtered_df[filtered_df['rsi_divergence'] == selected_divergence]
    if selected_volume_analysis != "All":
        filtered_df = filtered_df[filtered_df['volume_analysis'] == selected_volume_analysis]
    filtered_df = filtered_df[
        (filtered_df['rsi'].between(rsi_range[0], rsi_range[1])) &
        (filtered_df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
    ]
    
    # Apply turnover range filters
    if selected_turnover_ranges:
        turnover_conditions = []
        for range_key in selected_turnover_ranges:
            min_turnover, max_turnover = turnover_ranges[range_key]
            turnover_conditions.append(
                (filtered_df['turnover'] >= min_turnover) & (filtered_df['turnover'] < max_turnover)
            )
        filtered_df = filtered_df[pd.concat(turnover_conditions, axis=1).any(axis=1)]

    # Apply EMA filters
    if ema_20_check and 'ema_20' in df.columns:
        filtered_df = filtered_df[filtered_df['closing_price'] > filtered_df['ema_20']]
    if ema_50_check and 'ema_50' in df.columns:
        filtered_df = filtered_df[filtered_df['closing_price'] > filtered_df['ema_50']]
    if ema_100_check and 'ema_100' in df.columns:
        filtered_df = filtered_df[filtered_df['closing_price'] > filtered_df['ema_100']]
    if ema_200_check and 'ema_200' in df.columns:
        filtered_df = filtered_df[filtered_df['closing_price'] > filtered_df['ema_200']]

    
    # Rename headers
    filtered_df.columns = [col.replace('_', ' ').title() for col in filtered_df.columns]

    numeric_columns = [
    'Closing Price', 'Prev Close', 'Turnover'
    ]
    
    for col in numeric_columns:
        if col in filtered_df.columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
    
     # Sort the table by Turnover in descending order
    if 'Turnover' in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='Turnover', ascending=False)
        
    # Format numeric values with commas
    for col in filtered_df.select_dtypes(include=['float64', 'int64']).columns:
        filtered_df[col] = filtered_df[col].apply(lambda x: f"{x:,.2f}" if isinstance(x, float) else f"{x:,}")

    filtered_df = filtered_df.drop(columns=[col for col in ['Vol Avg 5D','Vol Avg 20D', 'Ema 20', 'Ema 50', 'Ema 100', 'Ema 200', 'Last Updated'] if col in filtered_df.columns])
        
    # Display the filtered table
    st.dataframe(filtered_df, use_container_width=True)
    
    if not filtered_df.empty:
        st.markdown("## 💎 Maverick's Potential Gems")
        
        # Add a date picker for filtering Maverick's Picks
        selected_maverick_date = st.date_input(
        "Select Start Date for Maverick's Picks",
        value=filtered_df['Date'].min().date(),  # Default to the earliest date in the filtered data
        min_value=filtered_df['Date'].min().date(),
        max_value=filtered_df['Date'].max().date()
        )
        
        # Ensure numeric columns are properly typed
        numeric_columns = [
        'Turnover', 'Volume', 'Relative Strength', 'Closing Price', 'Prev Close'
        ]
        for col in numeric_columns:
            if col in filtered_df.columns:
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

        
        # Filter data based on the selected date
        df['Date'] = pd.to_datetime(df['date'], errors='coerce')  # Ensure Date column is datetime
        maverick_filtered_df = df[df['date'] >= pd.to_datetime(selected_maverick_date)]
        columns_to_remove = ['Vol Avg 5D', 'Vol Avg 20D','Last Updated']
        maverick_filtered_df = maverick_filtered_df.drop(columns=[col for col in columns_to_remove if col in maverick_filtered_df.columns])       
        # Debugging: Display the filtered DataFrame
        #st.write("Filtered Maverick DataFrame:", maverick_filtered_df)
        
        # Get Tier 1 and Tier 2 picks
        tier_1_picks, tier_2_picks = get_mavericks_picks(maverick_filtered_df)

        # Display Tier 1 Picks
        st.markdown("### 🌟 Tier 1 Picks")
        if not tier_1_picks.empty:
            
            columns_to_remove = ['vol_avg_5d', 'vol_avg_20d']
            tier_1_picks = tier_1_picks.drop(columns=[col for col in columns_to_remove if col in tier_1_picks.columns])
            
            # Format numeric values with commas
            for col in ['Turnover', 'Volume']:
                if col in tier_1_picks.columns:
                    tier_1_picks[col] = tier_1_picks[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else x)
    
            # Sort by Date
            tier_1_picks = tier_1_picks.sort_values(by='date', ascending=False)  
            
            st.markdown("These are the counters identified by Maverick as having the highest potential for Gains.")
            st.dataframe(tier_1_picks, use_container_width=True)
            
        else:
            st.info("No stocks meet Tier 1 conditions.")
        # Display Tier 2 Picks
        st.markdown("### Tier 🔹 2 Picks")
        if not tier_2_picks.empty:
            
            columns_to_remove = ['vol_avg_5d', 'vol_avg_20d']
            tier_2_picks = tier_2_picks.drop(columns=[col for col in columns_to_remove if col in tier_1_picks.columns])
            
            # Format numeric values with commas
            for col in ['turnover', 'volume']:
                if col in tier_2_picks.columns:
                    tier_2_picks[col] = tier_2_picks[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else x)
    
            # Sort by Date
            tier_2_picks = tier_2_picks.sort_values(by='date', ascending=False)
            
            
            st.markdown("These stocks show moderate upside potential compared to the broader market. While not as strong as Tier 1 picks, they still present relatively favorable opportunities._")
            st.markdown("Pay attention to the stocks that have recurring mentions in the list, they have much better chances!")
            st.dataframe(tier_2_picks, use_container_width=True)
        else:
            st.info("No stocks meet Tier 2 conditions.")
            
    # === Chart Section ===
    if not filtered_df.empty:
        selected_chart_symbol = st.selectbox("📊 View Chart for Symbol", filtered_df['Symbol'].unique())
        chart_df = filtered_df[filtered_df['Symbol'] == selected_chart_symbol]
        
        # Ensure data is sorted by Date
        chart_df = chart_df.sort_values(by='Date')
        
        fig = px.line(chart_df, 
                      x='Date', 
                      y='Closing Price', 
                      title=f"📈 Closing Price Trend for {selected_chart_symbol}",
                      markers=True)
        
        fig.update_traces(line=dict(shape='linear'))  # Ensure smooth lines
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Closing Price",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No data matches the selected filters.")
        
    # === Legend Section ===
    st.markdown("## 📘 Legend: Understanding Key Terms")
    st.markdown("""
Here are some key terms to help you understand the analysis better:

- **📈 Relative Strength (RS)**:
  - A momentum indicator that compares the performance of a stock to the overall market or to the ASI.
  - **RS >= 1**: Indicates the stock is outperforming the market.
  - **RS < 1**: Indicates the stock is underperforming the market.

- **🔄 Bullish Divergence**:
  - Occurs when the stock's price is making lower lows, but the RSI (Relative Strength Index) is making higher lows.
  - This is a potential signal for a reversal to the upside.

- **📊 Volume Analysis Criteria**:
  - **Emerging Bullish Momentum**: Indicates a sudden increase in buying activity,compared to their weekly average volumes.Suggesting in start of interest shown to the stock.
  - **High Bullish Momentum**: Indicates break-out buying activity, higher volume than their weekly or monthly averages.Suggesting a strong,commited interest in the stock.
  - **Increase in Weekly Volume Activity Detected**: Highlights stocks with a gradual increase in trading volume compared to their weekly average.

- **📐 EMAs (Exponential Moving Averages)**:
  - A type of moving average that gives more weight to recent prices, making it more responsive to new information.
  - **EMA 20**: Short-term trend indicator.
  - **EMA 50**: Medium-term trend indicator.
  - **EMA 100**: Long-term trend indicator.
  - **EMA 200**: Very long-term trend indicator, often used to identify major support or resistance levels.

We hope this helps you better understand the analysis and make informed decisions! 🚀
""")

except Exception as e:
    st.error(f"An error occurred: {e}")
