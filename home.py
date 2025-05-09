# home.py
import streamlit as st
from auth_utils import get_oauth_client, do_login, do_logout, get_authenticated_user

import auth_utils
import sys
print(sys.path)
print(dir(auth_utils))

# Set the page configuration for the entire app
st.set_page_config(page_title="Analytics Dashboard", layout="wide")

st.title("Welcome to the Analytics Platform")

# --- Authentication Check ---
oauth_client = get_oauth_client()
user_info = get_authenticated_user()

if not user_info:
    # User is not logged in, show login button
    st.info("Please log in to access the dashboards.")
    do_login(oauth_client) # This function handles the OAuth flow and reruns

else:
    # User is logged in, display content for authenticated users
    user_email = user_info.get('email', 'N/A')
    user_name = user_info.get('name', 'User')

    # Display user info and logout button in the sidebar
    st.sidebar.write(f"Logged in as: {user_name} ({user_email})")
    if st.sidebar.button("Logout"):
        do_logout() # This function handles logout and reruns

    st.header("Main Dashboard Overview")
    st.write("Select a page from the sidebar to navigate:")

    # You can add any introductory content or a summary here.
    # Streamlit automatically creates sidebar navigation based on files in the 'pages/' directory.

    st.subheader("Your Google Account Details")
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** {user_email}")
    # Add more user info display if needed

    st.info("Use the sidebar on the left to access different sections like Dashboards, User Account, and Notifications.")

