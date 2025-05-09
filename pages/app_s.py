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
# It's generally better to load environment variables once.
# On Streamlit Cloud, you'll set secrets directly in the app settings.
# If running locally, the .env file path might need adjustment or handling
# based on the execution environment.
# load_dotenv(dotenv_path=r"E:\Side Projects\CSE bot\myenv\streamlit\myenv\Scripts\.env")
load_dotenv() # This will load from a .env file in the current directory or parent directories

# --- Database Connection ---
@st.cache_resource
def init_connection():
    # This line will now directly get the secret from Streamlit Cloud's environment
    db_url = os.environ.get("NEON_DB_URL")
    if not db_url:
        # This error will trigger if the secret is not set in Streamlit Cloud
        st.error("Database URL not found in environment variables! Please configure Streamlit Secrets.")
        return None
    try:
        url = urlparse.urlparse(db_url)
        conn = psycopg2.connect(
            database=url.path[1:],
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port or "5432",
            sslmode="require",
        )
        return conn
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return None


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
        # Check if the connection is closed before using (optional but can help catch early)
        # Note: Checking conn.closed might not always catch a network-level closure immediately
        # A safer approach is to rely on the OperationalError and reconnect logic.
        # if conn.closed != 0:
        #      status_message.warning("Cached connection found but it was closed. Attempting to re-establish...")
        #      init_connection.clear() # Clear the broken cached connection
        #      conn = init_connection() # Try to get a new connection immediately
        #      if conn is None:
        #          status_message.error("Failed to re-establish database connection.")
        #          return pd.DataFrame() # Return empty if reconnect failed


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
            # Convert 'date' to datetime, coercing errors
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Ensure 'date' is timezone-naive if it's not already, for consistent comparisons later
            if pd.api.types.is_datetime64tz_dtype(df['date']):
                 df['date'] = df['date'].dt.tz_convert(None)
            # Drop rows where date conversion failed
            df.dropna(subset=['date'], inplace=True)

        if 'last_updated' in df.columns:
             # Convert 'last_updated' to datetime, coercing errors
             df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
             # Ensure 'last_updated' is timezone-naive
             if pd.api.types.is_datetime64tz_dtype(df['last_updated']):
                 df['last_updated'] = df['last_updated'].dt.tz_convert(None)
             # Drop rows where conversion failed
             df.dropna(subset=['last_updated'], inplace=True)


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

# Initialize TradingView datafeed (Consider if this is actually used in the current app logic,
# otherwise it can be removed or moved inside a function that uses it)
# tv = TvDatafeed()


def get_mavericks_picks(results_df):
    """Filters stocks for Mavericks Picks based on Tier 1, Tier 2, and Tier 3 conditions."""
    # Ensure numeric columns are properly typed before applying conditions
    numeric_cols_for_conditions = ['turnover', 'volume', 'relative_strength']
    for col in numeric_cols_for_conditions:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0) # Fill NaN with 0 for comparison

    # Ensure 'volume_analysis' and 'rsi_divergence' are strings to prevent errors in isin()
    if 'volume_analysis' in results_df.columns:
         results_df['volume_analysis'] = results_df['volume_analysis'].astype(str)
    if 'rsi_divergence' in results_df.columns:
         results_df['rsi_divergence'] = results_df['rsi_divergence'].astype(str)


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

    # Apply conditions and return the filtered DataFrames (return copies to prevent SettingWithCopyWarning)
    tier_1_picks = results_df[tier_1_conditions].copy()
    tier_2_picks = results_df[tier_2_conditions].copy()
    tier_3_picks = results_df[tier_3_conditions].copy()

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

    # Check for duplicate columns and remove them (should ideally be handled during data loading if possible)
    # But keeping this check as a safeguard
    if df.columns.duplicated().any():
        st.warning(f"Duplicate column names found: {df.columns[df.columns.duplicated()].tolist()}")
        df = df.loc[:, ~df.columns.duplicated()].copy()  # Remove duplicate columns and make a copy

    # Remove unwanted columns from the main dataframe early if they are never used
    # Based on the display logic, 'id' is dropped. Let's keep other columns needed for filtering and tier logic.
    if 'id' in df.columns:
        df = df.drop(columns=['id'])


    if not df.empty:
        # Add a date picker for filtering Maverick's Picks
        # Ensure min/max values for date picker are valid dates
        min_date_value = df['date'].min().date() if pd.notna(df['date'].min()) else datetime.date.today() - datetime.timedelta(days=365*5) # Default to 5 years ago
        max_date_value = df['date'].max().date() if pd.notna(df['date'].max()) else datetime.date.today()
        # Set a default value that is within the valid range
        default_date_value = max(min_date_value, datetime.date.today() - datetime.timedelta(days=5)) # Default to last 30 days or min date

        selected_maverick_date = st.date_input(
            "Select Start Date for Filtering Stocks",
            value=default_date_value,
            min_value=min_date_value,
            max_value=max_date_value
        )

        # Ensure numeric columns used in filtering and display are properly typed
        numeric_columns = [
            'turnover', 'volume', 'relative_strength', 'closing_price', 'prev_close',
            'rsi', 'ema_20', 'ema_50', 'ema_100', 'ema_200', 'vol_avg_5d', 'vol_avg_20d', 'change_pct'
        ]
        for col in numeric_columns:
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')


        # Filter data based on the selected date for Maverick's Picks
        # The filtering is done inside get_mavericks_picks on the maverick_filtered_df,
        # so we just need to create that filtered df here.
        # Ensure df['date'] is datetime before comparison
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        maverick_filtered_df = df[df['date'] >= pd.to_datetime(selected_maverick_date)].copy()


        # Get Tier 1, Tier 2, and Tier 3 picks
        tier_1_picks_full, tier_2_picks_full, tier_3_picks_full = get_mavericks_picks(maverick_filtered_df)

        # --- Display Tier 1 Picks ---
        st.markdown("### 🌟 Bullish Volumes!")
        st.markdown("These are the counters identified by Maverick to have interesting Volume Signatures.")
        if not tier_1_picks_full.empty:
            # Define columns to keep for Tier 1 display
            tier_1_cols_to_show = ['date', 'symbol', 'closing_price', 'change_pct', 'volume', 'turnover', 'volume_analysis']
            # Filter for columns that actually exist
            tier_1_cols_to_show = [col for col in tier_1_cols_to_show if col in tier_1_picks_full.columns]
            tier_1_picks_show = tier_1_picks_full[tier_1_cols_to_show].copy() # Select columns and make a copy

            # Format numeric values with commas
            for col in ['turnover', 'volume', 'closing_price', 'change_pct']:
                if col in tier_1_picks_show.columns:
                     tier_1_picks_show[col] = pd.to_numeric(tier_1_picks_show[col], errors='coerce')
                     if col in ['turnover', 'volume']:
                          tier_1_picks_show[col] = tier_1_picks_show[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else '')
                     elif col in ['closing_price', 'change_pct']:
                          tier_1_picks_show[col] = tier_1_picks_show[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else '')

            # Sort by Date
            if 'date' in tier_1_picks_show.columns:
                 tier_1_picks_show = tier_1_picks_show.sort_values(by='date', ascending=False)

            # Rename columns for display
            column_rename_map_tier1 = {
                'date': 'Date',
                'symbol': 'Symbol',
                'change_pct': '% Change',
                'closing_price': "Today Close",
                'volume': 'Volume',
                'turnover': 'Turnover',
                'volume_analysis': 'Volume Analysis'
            }
            column_rename_map_tier1 = {k: v for k, v in column_rename_map_tier1.items() if k in tier_1_picks_show.columns}
            tier_1_picks_show = tier_1_picks_show.rename(columns=column_rename_map_tier1)

            # Format the Date column to remove the time component
            if 'Date' in tier_1_picks_show.columns:
                 tier_1_picks_show['Date'] = pd.to_datetime(tier_1_picks_show['Date']).dt.date

            if not tier_1_picks_show.empty:
                tier_1_picks_show = tier_1_picks_show.reset_index(drop=True)  # Remove index

            st.dataframe(tier_1_picks_show, use_container_width=True)
        else:
            st.info("No stocks found with Bullish Volume signatures in the selected period.")

        # Find recurring stocks based on the full tier 1 picks DataFrame
        if not tier_1_picks_full.empty:
            recurring_stocks_1 = tier_1_picks_full['symbol'].value_counts()
            recurring_stocks_1 = recurring_stocks_1[recurring_stocks_1 >= 4]

            if not recurring_stocks_1.empty:
                st.markdown("List of Stocks with Repeated Bullish Volume Signatures:")
                for stock, count in recurring_stocks_1.items():
                    st.markdown(f"- **{stock}**: {count} times")
            else:
                 st.info("")


        # --- Display Tier 2 Picks ---
        st.markdown("### 🔄 Potential Reversal Ahead!")
        st.markdown("Stocks that are showing a potential reversal in price action due to divergence with RSI.")

        if not tier_2_picks_full.empty:
            # Define columns to keep for Tier 2 display
            tier_2_cols_to_show = ['date', 'symbol', 'closing_price', 'change_pct', 'turnover', 'rsi', 'rsi_divergence', 'relative_strength']
             # Filter for columns that actually exist
            tier_2_cols_to_show = [col for col in tier_2_cols_to_show if col in tier_2_picks_full.columns]
            tier_2_picks_show = tier_2_picks_full[tier_2_cols_to_show].copy() # Select columns and make a copy


            # Format numeric values with commas
            for col in ['turnover', 'closing_price', 'change_pct', 'rsi', 'relative_strength']:
                 if col in tier_2_picks_show.columns:
                      tier_2_picks_show[col] = pd.to_numeric(tier_2_picks_show[col], errors='coerce')
                      if col == 'turnover':
                           tier_2_picks_show[col] = tier_2_picks_show[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else '')
                      elif col in ['closing_price', 'change_pct', 'rsi', 'relative_strength']:
                            tier_2_picks_show[col] = tier_2_picks_show[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else '')


            # Sort by Date
            if 'date' in tier_2_picks_show.columns:
                 tier_2_picks_show = tier_2_picks_show.sort_values(by='date', ascending=False)

            # Rename columns for display
            column_rename_map_tier2 = {
                 'date': 'Date',
                 'symbol': 'Symbol',
                 'change_pct': '% Change',
                 'closing_price': "Today Close",
                 'turnover': 'Turnover',
                 'rsi': 'RSI',
                 'rsi_divergence': 'RSI Divergence',
                 'relative_strength': 'Relative Strength'
            }
            column_rename_map_tier2 = {k: v for k, v in column_rename_map_tier2.items() if k in tier_2_picks_show.columns}
            tier_2_picks_show = tier_2_picks_show.rename(columns=column_rename_map_tier2)

            # Format the Date column to remove the time component
            if 'Date' in tier_2_picks_show.columns:
                 tier_2_picks_show['Date'] = pd.to_datetime(tier_2_picks_show['Date']).dt.date

            st.dataframe(tier_2_picks_show, use_container_width=True)
        else:
            st.info("No stocks found with RSI Divergence in the selected period.")

        # --- Display Tier 3 Picks ---
        st.markdown("### 🏆 Top Performers!")
        st.markdown("These are rather liquid Stocks that has registered a Bullish Volume as well as price action stronger than the ASI.")

        if not tier_3_picks_full.empty:
            # Define columns to keep for Tier 3 display
            tier_3_cols_to_show = ['date', 'symbol', 'closing_price', 'change_pct', 'volume', 'turnover', 'volume_analysis', 'relative_strength']
            # Filter for columns that actually exist
            tier_3_cols_to_show = [col for col in tier_3_cols_to_show if col in tier_3_picks_full.columns]
            tier_3_picks_show = tier_3_picks_full[tier_3_cols_to_show].copy() # Select columns and make a copy

            # Format numeric values with commas
            for col in ['turnover', 'volume', 'closing_price', 'change_pct', 'relative_strength']:
                if col in tier_3_picks_show.columns:
                     tier_3_picks_show[col] = pd.to_numeric(tier_3_picks_show[col], errors='coerce')
                     if col in ['turnover', 'volume']:
                          tier_3_picks_show[col] = tier_3_picks_show[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else '')
                     elif col in ['closing_price', 'change_pct', 'relative_strength']:
                           tier_3_picks_show[col] = tier_3_picks_show[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else '')

            # Sort by Date
            if 'date' in tier_3_picks_show.columns:
                 tier_3_picks_show = tier_3_picks_show.sort_values(by='date', ascending=False)

            # Rename columns for display
            column_rename_map_tier3 = {
                 'date': 'Date',
                 'symbol': 'Symbol',
                 'change_pct': '% Change',
                 'closing_price': "Today Close",
                 'volume': 'Volume',
                 'turnover': 'Turnover',
                 'volume_analysis': 'Volume Analysis',
                 'relative_strength': 'Relative Strength'
            }
            column_rename_map_tier3 = {k: v for k, v in column_rename_map_tier3.items() if k in tier_3_picks_show.columns}
            tier_3_picks_show = tier_3_picks_show.rename(columns=column_rename_map_tier3)

            # Format the Date column to remove the time component
            if 'Date' in tier_3_picks_show.columns:
                 tier_3_picks_show['Date'] = pd.to_datetime(tier_3_picks_show['Date']).dt.date

            st.dataframe(tier_3_picks_show, use_container_width=True)
        else:
            st.info("No Top Performing stocks found in the selected period based on the criteria.")


    # === Filters Section ===
    st.markdown("### 🔍 DIY & Take Control of Your Analysis")
    st.markdown("Use the filters below to invoke your selection criteria.")
    st.markdown("You can filter stocks based on RSI, Divergence, Volume Analysis, and more.")
    st.markdown("")
    # Dropdown filters (Use df for options as it has all data)
    selected_symbol = st.selectbox("Select Symbol", options=["All"] + sorted(list(df['symbol'].unique()))) # Sort symbols
    # Handle potential NaNs in unique values for selectboxes
    selected_divergence = st.selectbox("Select Divergence Check", options=["All"] + sorted(list(df['rsi_divergence'].dropna().unique())))
    selected_volume_analysis = st.selectbox("Select Volume Analysis", options=["All"] + sorted(list(df['volume_analysis'].dropna().unique())))


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
    # Ensure min/max values are calculated on numeric columns, handle potential NaNs
    rsi_min = float(df['rsi'].min()) if 'rsi' in df.columns and pd.notna(df['rsi'].min()) else 0.0
    rsi_max = float(df['rsi'].max()) if 'rsi' in df.columns and pd.notna(df['rsi'].max()) else 100.0
    rsi_range = st.slider("RSI Range", rsi_min, rsi_max, (30.0, 70.0))

    # Ensure date min/max are valid datetime objects before converting to date
    date_min_val = df['last_updated'].min() if 'last_updated' in df.columns and pd.notna(df['last_updated'].min()) else datetime.datetime(2020, 1, 1)
    date_max_val = df['last_updated'].max() if 'last_updated' in df.columns and pd.notna(df['last_updated'].max()) else datetime.datetime.now()

    date_range = st.slider(
        "Date Range",
        min_value=date_min_val.date(),
        max_value=date_max_val.date(),
        value=(date_min_val.date(), date_max_val.date())
    )

    # EMA Checker
    st.markdown("### EMA Checker")
    # Check if EMA columns exist before showing checkboxes
    ema_20_exists = 'ema_20' in df.columns and 'closing_price' in df.columns
    ema_50_exists = 'ema_50' in df.columns and 'closing_price' in df.columns
    ema_100_exists = 'ema_100' in df.columns and 'closing_price' in df.columns
    ema_200_exists = 'ema_200' in df.columns and 'closing_price' in df.columns

    ema_20_check = st.checkbox("Price Above EMA 20", disabled=not ema_20_exists) if ema_20_exists else False
    ema_50_check = st.checkbox("Price Above EMA 50", disabled=not ema_50_exists) if ema_50_exists else False
    ema_100_check = st.checkbox("Price Above EMA 100", disabled=not ema_100_exists) if ema_100_exists else False
    ema_200_check = st.checkbox("Price Above EMA 200", disabled=not ema_200_exists) if ema_200_exists else False
    
    if not any([ema_20_exists, ema_50_exists, ema_100_exists, ema_200_exists]):
        st.info("EMA data is not available in the dataset.")


    st.markdown("## Filtered Results")

    # Apply filters to a copy of the original dataframe
    filtered_df = df.copy()

    # Apply symbol filter
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
         # Ensure 'rsi' is numeric before filtering
         filtered_df['rsi'] = pd.to_numeric(filtered_df['rsi'], errors='coerce')
         filtered_df = filtered_df.dropna(subset=['rsi']) # Drop rows where rsi couldn't be converted

         # Ensure 'date' is datetime and timezone-naive
         filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
         if pd.api.types.is_datetime64tz_dtype(filtered_df['date']):
             filtered_df['date'] = filtered_df['date'].dt.tz_convert(None)
         filtered_df.dropna(subset=['date'], inplace=True) # Drop rows where date conversion failed

         # Ensure date range values are timezone-naive datetimes for comparison
         start_date_dt = pd.to_datetime(date_range[0])
         end_date_dt = pd.to_datetime(date_range[1])

         filtered_df = filtered_df[
             (filtered_df['rsi'].between(rsi_range[0], rsi_range[1])) &
             (filtered_df['date'].between(start_date_dt, end_date_dt))
         ]


    # Apply turnover range filters
    if selected_turnover_ranges and 'turnover' in filtered_df.columns:
        # Ensure 'turnover' is numeric
        filtered_df['turnover'] = pd.to_numeric(filtered_df['turnover'], errors='coerce')
        filtered_df = filtered_df.dropna(subset=['turnover']) # Drop rows where turnover couldn't be converted

        turnover_conditions = []
        for range_key in selected_turnover_ranges:
            min_turnover, max_turnover = turnover_ranges[range_key]
            turnover_conditions.append(
                (filtered_df['turnover'] >= min_turnover) & (filtered_df['turnover'] < max_turnover)
            )
        # Use .any(axis=1) to check if the row matches *any* of the selected ranges
        if turnover_conditions:
            # Concatenate the boolean conditions and filter the DataFrame
            filtered_df = filtered_df[pd.concat(turnover_conditions, axis=1).any(axis=1)]


    # Apply EMA filters (Ensure EMA and closing_price are numeric)
    ema_cols = ['ema_20', 'ema_50', 'ema_100', 'ema_200']
    cols_for_ema_check = [col for col in ema_cols + ['closing_price'] if col in filtered_df.columns]

    if cols_for_ema_check: # Only proceed if relevant EMA columns or closing_price exist
         for col in cols_for_ema_check:
              filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
         # Drop rows with NaNs in relevant EMA or closing_price columns if needed for the filter
         # filtered_df.dropna(subset=cols_for_ema_check, inplace=True) # This might remove too many rows

         # Apply filters only if the column exists and checkbox is checked
         if ema_20_check and 'ema_20' in filtered_df.columns and 'closing_price' in filtered_df.columns:
             filtered_df = filtered_df[filtered_df['closing_price'] > filtered_df['ema_20']]
         if ema_50_check and 'ema_50' in filtered_df.columns and 'closing_price' in filtered_df.columns:
             filtered_df = filtered_df[filtered_df['closing_price'] > filtered_df['ema_50']]
         if ema_100_check and 'ema_100' in filtered_df.columns and 'closing_price' in filtered_df.columns:
             filtered_df = filtered_df[filtered_df['closing_price'] > filtered_df['ema_100']]
         if ema_200_check and 'ema_200' in filtered_df.columns and 'closing_price' in filtered_df.columns:
             filtered_df = filtered_df[filtered_df['closing_price'] > filtered_df['ema_200']]


    # Prepare filtered_df for display
    if not filtered_df.empty:
        # Define columns to keep for the main filtered table display
        filtered_cols_to_show = [
            'date', 'symbol', 'closing_price', 'change_pct', 'turnover', 'volume',
            'volume_analysis', 'rsi', 'rsi_divergence', 'relative_strength',
            'ema_20', 'ema_50', 'ema_100', 'ema_200', 'vol_avg_5d', 'vol_avg_20d'
        ]
        # Filter for columns that actually exist in filtered_df
        filtered_cols_to_show = [col for col in filtered_cols_to_show if col in filtered_df.columns]

        # Ensure filtered_df has at least the columns to show before selection
        # This prevents errors if applying filters resulted in a DataFrame with fewer columns unexpectedly
        existing_cols_to_show = [col for col in filtered_cols_to_show if col in filtered_df.columns]
        if not existing_cols_to_show:
             st.info("Filtering resulted in no data with the required columns for display.")
             st.dataframe(pd.DataFrame()) # Display empty dataframe to avoid errors
        else:
             filtered_df_show = filtered_df[existing_cols_to_show].copy() # Select columns and make a copy

             # Sort the table by Date and then Turnover in descending order
             sort_cols = []
             if 'date' in filtered_df_show.columns:
                  sort_cols.append('date')
             if 'turnover' in filtered_df_show.columns:
                  # Ensure turnover is numeric before sorting
                  filtered_df_show['turnover'] = pd.to_numeric(filtered_df_show['turnover'], errors='coerce')
                  sort_cols.append('turnover')

             if sort_cols:
                  filtered_df_show = filtered_df_show.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))


             # Format numeric values with commas
             for col in filtered_df_show.select_dtypes(include=['float64', 'int64']).columns:
                 if col in ['turnover', 'volume', 'vol_avg_5d', 'vol_avg_20d']:
                     filtered_df_show[col] = filtered_df_show[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else '')
                 elif col in ['closing_price', 'change_pct', 'rsi', 'relative_strength', 'ema_20', 'ema_50', 'ema_100', 'ema_200']:
                      filtered_df_show[col] = filtered_df_show[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else '')

             # Rename columns for display
             column_rename_map_filtered = {
                 'date': 'Date',
                 'symbol': 'Symbol',
                 'closing_price': "Today Closing Price",
                 'change_pct': '% Change',
                 'turnover': 'Turnover',
                 'volume': 'Volume',
                 'volume_analysis': 'Volume Analysis',
                 'rsi': 'RSI',
                 'rsi_divergence': 'RSI Divergence',
                 'relative_strength': 'Relative Strength',
                 'ema_20': 'EMA 20',
                 'ema_50': 'EMA 50',
                 'ema_100': 'EMA 100',
                 'ema_200': 'EMA 200',
                 'vol_avg_5d': 'Vol Avg 5D',
                 'vol_avg_20d': 'Vol Avg 20D'
             }
             # Only rename columns that are actually present in filtered_df_show
             column_rename_map_filtered = {k: v for k, v in column_rename_map_filtered.items() if k in filtered_df_show.columns}
             filtered_df_show = filtered_df_show.rename(columns=column_rename_map_filtered)

             filtered_df_show = filtered_df_show.reset_index(drop=True)

             # Format the Date column to remove the time component
             if 'Date' in filtered_df_show.columns:
                  filtered_df_show['Date'] = pd.to_datetime(filtered_df_show['Date']).dt.date


             # Display the filtered table
             st.dataframe(filtered_df_show, use_container_width=True)
    else:
        st.info("No results match the selected filters.")


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