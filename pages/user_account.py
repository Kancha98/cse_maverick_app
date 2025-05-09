# pages/user_account.py
import streamlit as st
from auth_utils import get_authenticated_user # Import the authentication helper

# --- Authentication Check ---
# This check is crucial for every page that requires authentication.
user_info = get_authenticated_user()
if not user_info:
    # If user is not logged in, show a warning and stop execution of this page
    st.warning("Please log in to view this page.")
    st.stop() # This stops the script execution for this specific page

# User is logged in, proceed with displaying user account details
st.title("User Account Details")

st.subheader("Your Google Account Information")

# Retrieve user details from the user_info dictionary
user_email = user_info.get('email', 'N/A')
user_name = user_info.get('name', 'User')
user_picture = user_info.get('picture') # URL to the user's profile picture
user_id = user_info.get('sub') # Google User ID, a unique identifier for the user

# Display the user's information
st.write(f"**Name:** {user_name}")
st.write(f"**Email:** {user_email}")
st.write(f"**Google User ID:** {user_id}")

# Display the profile picture if available
if user_picture:
    st.image(user_picture, caption="Profile Picture", width=100)

st.write("This information is retrieved from your Google Account.")
st.write("You can add functionality here to link other profile details if needed,")
st.write("storing them in your database associated with the Google User ID or email.")

