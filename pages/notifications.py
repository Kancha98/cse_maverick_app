# pages/notifications.py
import streamlit as st
from auth_utils import get_authenticated_user # Import the authentication helper
from redis import Redis # Import Redis client
import os
from datetime import datetime
import pytz # For timezone handling

# --- Authentication Check ---
# This check is crucial for every page that requires authentication.
user_info = get_authenticated_user()
if not user_info:
    st.warning("Please log in to view this page.")
    st.stop() # This stops the script execution for this specific page

# User is logged in, get their email for checking notification status
user_email = user_info.get('email', 'N/A')

st.title("Notification Status")

# --- Redis Setup (Consistent across app files that interact with Redis) ---
# Ensure these connection details match your Redis server and task.py
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))

# Attempt to connect to Redis. If it fails, disable status check.
try:
    redis_conn_app = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    redis_conn_app.ping() # Check connection by sending a PING command
    redis_available = True
except Exception as e:
    st.warning(f"Could not connect to Redis: {e}. Cannot check notification status.")
    redis_available = False

# --- Notification Status Display ---

st.subheader("Daily Email Notification Status")

if redis_available:
    # Construct the Redis key used by the worker to mark notifications sent
    today_str = datetime.now().strftime('%Y-%m-%d')
    redis_key = f"notification_sent:{user_email}:{today_str}"

    # Check if the key exists in Redis
    if redis_conn_app.exists(redis_key):
        st.success("Your daily picks notification has been sent today.")
        # Optional: If your worker stored a timestamp, you could retrieve and display it
        # sent_timestamp = redis_conn_app.get(redis_key)
        # if sent_timestamp:
        #    # Convert timestamp (bytes) to string, then int, then datetime
        #    try:
        #        sent_time = datetime.fromtimestamp(int(sent_timestamp.decode('utf-8')))
        #        st.write(f"Sent at: {sent_time.strftime('%Y-%m-%d %H:%M:%S')}")
        #    except (ValueError, TypeError) as ts_error:
        #        st.warning(f"Could not parse timestamp from Redis: {ts_error}")
    else:
        st.info("Your daily picks notification has not been sent today.")
        st.write("Go to the Dashboard page that generates picks (e.g., Dashboard 1) and click 'Send My Picks via Email' to trigger it.")

    st.subheader("Notification History (Requires Database Storage)")
    st.write("To see a history of notifications sent over time (beyond just today's status), you would need to store this information in a persistent database.")
    st.write("The current setup uses Redis only to enforce the daily sending limit.")

else:
    st.warning("Notification Functionality is coming Soon!.")

