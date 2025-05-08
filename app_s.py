import streamlit as st
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from streamlit.components.v1 import html
import psycopg2
import os
import urllib.parse as urlparse
import plotly.express as px
from dotenv import load_dotenv
import datetime

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
        port=url.port or "5432",
        sslmode="require",
    )

# --- Load Data ---
# Use `st.cache_data` for the dataframe result.
# It depends on the connection obtained from init_connection.
@st.cache_data(ttl=600)
def load_data():
    status_message = st.empty()

    status_message.text("Attempting to load data...") # Write initial status
    
    # Get a connection from the cache. It will be created only once by init_connection
    # until the cache is cleared or arguments change.
    conn = init_connection()

    if conn is None:
        status_message.error("Cannot load data: Database connection failed.")
        return pd.DataFrame() # Return empty DataFrame if connection wasn't established

    # --- Add a try-except block to handle stale connections ---
    try:
        # Use the connection within a 'with' block for safe cursor handling
        # Check if the connection is closed before using (optional but can help catch early)
        # Note: Checking conn.closed might not always catch a network-level closure immediately
        if conn.closed != 0:
             status_message.warning("Cached connection found but it was closed. Attempting to re-establish...")
             init_connection.clear() # Clear the broken cached connection
             conn = init_connection() # Try to get a new connection immediately
             if conn is None:
                 status_message.error("Failed to re-establish database connection.")
                 return pd.DataFrame() # Return empty if reconnect failed


        with conn.cursor() as cur:
            status_message.text("Executing SQL query...") # Update status
            cur.execute("SELECT * FROM stock_analysis_all_results;")
            colnames = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            
        status_message.success("Data loaded successfully.") # Debugging
        # DO NOT CLOSE THE CONNECTION HERE! @st.cache_resource manages its lifecycle.
        # conn.close() # <--- REMOVE THIS LINE! (You already did this, good!)

        df = pd.DataFrame(rows, columns=colnames)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Ensure 'date' is timezone-naive if it's not already, for consistent comparisons later
        if 'date' in df.columns and pd.api.types.is_datetime64tz_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Ensure 'date' is timezone-naive if it's not already, for consistent comparisons later
            # Use isinstance(dtype, pd.DatetimeTZDtype) as recommended by the warning
            if isinstance(df['date'].dtype, pd.DatetimeTZDtype): # <--- Corrected line
                df['date'] = df['date'].dt.tz_convert(None)
            # Drop rows where date conversion failed
            df.dropna(subset=['date'], inplace=True)


        return df

    except psycopg2.OperationalError as e:
        # This specific error often indicates connection problems (like being closed)
        status_message.error(f"Database Operational Error: {e}")
        st.info("Attempting to clear cached connection and reload.")
        # Clear the cached connection resource so init_connection will run again
        init_connection.clear()
        # Returning an empty DataFrame. Streamlit will rerun on user interaction,
        # or you could use st.rerun() but clearing cache and letting user interact
        # is often sufficient and less disruptive.
        return pd.DataFrame()

    except Exception as e:
        # Catch any other unexpected errors during data loading
        st.error(f"An unexpected error occurred while loading data: {e}")
        return pd.DataFrame()

from tvDatafeed import TvDatafeed, Interval
from streamlit.components.v1 import html

# Initialize TradingView datafeed
tv = TvDatafeed()



def get_mavericks_picks(results_df):
    """Filters stocks for Mavericks Picks based on Tier 1 and Tier 2 conditions."""
    # Ensure numeric columns are properly typed
    for col in ['turnover', 'volume', 'relative_strength']:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0)
    
    
    tier_1_conditions = (
        
        (results_df['volume_analysis'].isin(["High Bullish Momentum","Emerging Bullish Momentum", "Increase in weekly Volume Activity Detected"]))
    ) 

    tier_2_conditions = (
        
        (results_df['rsi_divergence'] == "Bearish Divergence")
    ) | (
        (results_df['rsi_divergence'] == "Bullish Divergence")

    )


    tier_3_conditions = (
        (results_df['volume_analysis'].isin(["Emerging Bullish Momentum", "High Bullish Momentum"]))&
        (results_df['turnover'] > 999999) &
        (results_df['volume'] > 9999) &
        (results_df['relative_strength'] >= 1)

    )
    
    tier_1_picks = results_df[tier_1_conditions]
    tier_2_picks = results_df[tier_2_conditions]
    
    tier_3_picks = results_df[tier_3_conditions]
    
    return tier_1_picks, tier_2_picks, tier_3_picks

# --- Streamlit App ---
st.title("📈 CSE Gem Finder by CSE Maverick")
st.markdown("💡An intelligent assistant to help you discover high-potential stocks by leveraging technical analysis tools!!")
st.markdown("Let's find Gems!")
st.markdown("")
st.markdown("Note: This app is for Research purposes only. Please do your own research before making any investment decisions.")


# Add a button to force a data reload (clears cache)
if st.button("Reload Data"):
    load_data.clear() # Clear the data cache
    init_connection.clear() # Clear the connection cache
    st.rerun() # Rerun the app immediately

try:
    df = load_data()

    if df.empty:
        st.warning("No data found in the table.")
        st.stop()
        
    # Check for duplicate columns and remove them
    if df.columns.duplicated().any():
        st.warning(f"Duplicate column names found: {df.columns[df.columns.duplicated()].tolist()}")
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns

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
    
    if not df.empty:
        # Add a date picker for filtering Maverick's Picks
        selected_maverick_date = st.date_input(
        "Select Start Date for Filtering Stocks",
        value=datetime.date(2025, 5, 1),  # Default to the earliest date in the filtered data
        min_value=df['date'].min().date(),
        max_value=df['date'].max().date()
        )
        
        # Ensure numeric columns are properly typed
        numeric_columns = [
        'turnover', 'volume', 'relative_strength', 'closing_price', 'prev_close'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        
        # Filter data based on the selected date
        df['Date'] = pd.to_datetime(df['date'], errors='coerce')  # Ensure Date column is datetime
        maverick_filtered_df = df[df['date'] >= pd.to_datetime(selected_maverick_date)]
        columns_to_remove = ['vol_avg_5d', 'vol_avg_20d','last_updated']
        maverick_filtered_df = maverick_filtered_df.drop(columns=[col for col in columns_to_remove if col in maverick_filtered_df.columns])       
        # Debugging: Display the filtered DataFrame
        #st.write("Filtered Maverick DataFrame:", maverick_filtered_df)
        
        # Get Tier 1 and Tier 2 picks
        tier_1_picks, tier_2_picks,tier_3_picks = get_mavericks_picks(maverick_filtered_df)

        # Display Tier 1 Picks
        st.markdown("### 🌟 Bullish Volumes!")
        st.markdown("These are the counters identified by Maverick to have interesting Volume Signatures.")
        if not tier_1_picks.empty:
            
            columns_to_remove = ['vol_avg_5d','vol_avg_20d',
                                 'ema_20', 'ema_50','ema_100', 'ema_200','Date','rsi','rsi_divergence',
                                 'relative_strength','last_updated',
                                 'prev_close'
                                                            ]
           # Reset the index to remove the index column
            tier_1_picks = tier_1_picks.reset_index(drop=True)
            
            # Format numeric values with commas
            for col in ['turnover', 'volume']:
                if col in tier_1_picks.columns:
                    tier_1_picks[col] = tier_1_picks[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else x)
    
            # Sort by Date
            tier_1_picks = tier_1_picks.sort_values(by='date', ascending=False)
            
            tier_1_picks = tier_1_picks.drop(columns=[col for col in columns_to_remove if col in tier_1_picks.columns])
            
            column_rename_map = {
                'change_pct': '% Change',
                'closing_price': "Today Close",
                }
            
            
            tier_1_picks_show = tier_1_picks.copy()
            
            tier_1_picks_show = tier_1_picks_show.rename(columns=column_rename_map)
            
            tier_1_picks_show.columns = [col.replace('_', ' ').title() for col in tier_1_picks_show.columns]
            
            # Format the Date column to remove the time component
            if 'Date' in tier_1_picks_show.columns:
                tier_1_picks_show['Date'] = pd.to_datetime(tier_1_picks_show['Date']).dt.date
            
            
            st.dataframe(tier_1_picks_show, use_container_width=True)
        else:
            st.info("")
            
        if not tier_1_picks.empty:
            
         # Find the most recurring stocks and their counts
            recurring_stocks_1 = tier_1_picks['symbol'].value_counts()
            recurring_stocks_1 = recurring_stocks_1[recurring_stocks_1 >= 4]  # Filter stocks with count >= 2

            if not recurring_stocks_1.empty:
                st.markdown("List of Stocks with Repeated Bullish Volume Signatures:")
                for stock, count in recurring_stocks_1.items():
                    st.markdown(f"- **{stock}**: {count} times")
            else:
                st.info("")
                
        # Display Tier 2 Picks
        
        st.markdown("### Imminent Reversal!")
        st.markdown("Stocks that are showing a potential reversal in price action due to divergence with RSI.")
        
        if tier_2_picks.columns.duplicated().any():
            st.warning(f"Duplicate column names found: {tier_2_picks.columns[tier_2_picks.columns.duplicated()].tolist()}")
            tier_2_picks = tier_2_picks.loc[:, ~tier_2_picks.columns.duplicated()]  # Remove duplicate columns
        
        if not tier_2_picks.empty:
            
            columns_to_remove = ['vol_avg_5d','vol_avg_20d',
                                 'ema_20', 'ema_50','ema_100', 'ema_200','Date',
                                 'Last Updated', 'volume','volume_analysis',
                                 'prev_close'
                                                            ]
            
            
            # Reset the index to remove the index column
            tier_2_picks = tier_2_picks.reset_index(drop=True)
            
            # Format numeric values with commas
            for col in ['turnover', 'volume']:
                if col in tier_2_picks.columns:
                    tier_2_picks[col] = tier_2_picks[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else x)
    
            # Sort by Date
            tier_2_picks = tier_2_picks.sort_values(by='date', ascending=False)
            
            tier_2_picks = tier_2_picks.drop(columns=[col for col in columns_to_remove if col in tier_1_picks.columns])
            
            column_rename_map = {
                'change_pct': '% Change',
                'closing_price': "Today Close",
                }
            
            
            tier_2_picks_show = tier_2_picks.copy()
            
            tier_2_picks_show = tier_2_picks_show.rename(columns=column_rename_map)
            
            tier_2_picks_show.columns = [col.replace('_', ' ').title() for col in tier_2_picks_show.columns]
            
            # Format the Date column to remove the time component
            if 'Date' in tier_2_picks_show.columns:
                tier_2_picks_show['Date'] = pd.to_datetime(tier_2_picks_show['Date']).dt.date
            
            
            st.dataframe(tier_2_picks_show, use_container_width=True)
        else:
            st.info("")
        
        st.markdown("### Top Performers!")
        st.markdown("These are rather liquid Stocks that has registered a Bullish Volume as well as price action stronger than the RSI.")
        
        if not tier_3_picks.empty:
            
            columns_to_remove = ['vol_avg_5d','vol_avg_20d',
                                 'ema_20', 'ema_50','ema_100', 'ema_200','Date',
                                 'Last Updated', 'Rsi','Rsi Divergence',
                                 'prev_close'
                                                            ]
            
            
            # Reset the index to remove the index column
            tier_3_picks = tier_3_picks.reset_index(drop=True)
            
            # Format numeric values with commas
            for col in ['turnover', 'volume']:
                if col in tier_3_picks.columns:
                    tier_3_picks[col] = tier_3_picks[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else x)
    
            # Sort by Date
            tier_3_picks = tier_3_picks.sort_values(by='date', ascending=False)
            
            tier_3_picks = tier_3_picks.drop(columns=[col for col in columns_to_remove if col in tier_1_picks.columns])
            
            column_rename_map = {
                'change_pct': '% Change',
                'closing_price': "Today Close",
                }
            
            
            tier_3_picks_show = tier_3_picks.copy()
            
            tier_3_picks_show = tier_3_picks_show.rename(columns=column_rename_map)
            
            tier_3_picks_show.columns = [col.replace('_', ' ').title() for col in tier_3_picks_show.columns]
            
            # Format the Date column to remove the time component
            if 'Date' in tier_3_picks_show.columns:
                tier_3_picks_show['Date'] = pd.to_datetime(tier_3_picks_show['Date']).dt.date
            
            
            st.dataframe(tier_3_picks_show, use_container_width=True)
        else:
            st.info("")
            

    # === Filters Section ===
    st.markdown("### 🔍 DIY & Take Control of Your Analysis")
    st.markdown("Use the filters below to invoke your selection criteria.")
    st.markdown("You can filter stocks based on RSI, Divergence, Volume Analysis, and more.")
    st.markdown("")
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
        default=["10M-100M", "100M+"]
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
    
    
    st.markdown("## Filtered Results")
    
    # Check for duplicate columns and remove them
    if df.columns.duplicated().any():
        st.warning(f"Duplicate column names found: {df.columns[df.columns.duplicated()].tolist()}")
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
    
    # Apply filters
    filtered_df = df.copy()
    
    #Apply symbol filter
    if selected_symbol != "All" and 'symbol' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]

    # Apply divergence filter
    if selected_divergence != "All" and 'rsi_divergence' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['rsi_divergence'] == selected_divergence]

    # Apply volume analysis filter
    if selected_volume_analysis != "All" and 'volume_analysis' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['volume_analysis'] == selected_volume_analysis]

    # Apply RSI and date range filters
    if 'rsi' in filtered_df.columns and 'date' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['rsi'].between(rsi_range[0], rsi_range[1])) &
            (filtered_df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
    ]

    # Handle empty DataFrame
    if filtered_df.empty:
        st.info("No results match the selected filters.")
    #else:
        #st.dataframe(filtered_df, use_container_width=True)
    
    
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
    

    #numeric_columns = [
    #'Closing Price', 'Prev Close', 'Turnover'
    #]
    
    for col in numeric_columns:
        if col in filtered_df.columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
    
     # Sort the table by Turnover in descending order
    if 'turnover' in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by='turnover', ascending=False)
        
    # Format numeric values with commas
    for col in filtered_df.select_dtypes(include=['float64', 'int64']).columns:
        filtered_df[col] = filtered_df[col].apply(lambda x: f"{x:,.2f}" if isinstance(x, float) else f"{x:,}")

    filtered_df = filtered_df.drop(columns=[col for col in ['Vol Avg 5D','Vol Avg 20D', 'Ema 20', 'Ema 50', 
                                                            'Ema 100', 'Ema 200', 'Last Updated','Date', 'Symbol', 
                                                            'Closing Price', 'Prev Close', 'Change Pct', 'Turnover', 
                                                            'Volume', 'Volume Analysis', 'Rsi', 'Rsi Divergence', 
                                                            'Relative Strength','vol_avg_5d','vol_avg_20d',
                                                            'ema_20', 'ema_50', 
                                                            'ema_100', 'ema_200', 'Last Updated'
                                                            ] if col in filtered_df.columns])    
    
    filtered_df_show = filtered_df.copy()
        
    # Sort by Date
    filtered_df_show = filtered_df_show.sort_values(by='date', ascending=False)
            
    
            
    column_rename_map = {
                'change_pct': '% Change',
                'closing_price': "Today Closing Price",
                'prev_close': "Previous Day Closing Price"
                }
            
    filtered_df_show = filtered_df_show.rename(columns=column_rename_map)
            
    filtered_df_show.columns = [col.replace('_', ' ').title() for col in filtered_df_show.columns]
            
    filtered_df_show = filtered_df_show.reset_index(drop=True)
    
    # Format the Date column to remove the time component
    if 'Date' in filtered_df_show.columns:
            filtered_df_show['Date'] = pd.to_datetime(filtered_df_show['Date']).dt.date
    
    # Display the filtered table
    st.dataframe(filtered_df_show, use_container_width=True)
    
    
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
