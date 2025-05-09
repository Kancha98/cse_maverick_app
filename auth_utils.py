import streamlit as st
from streamlit_oauth import OAuth2
import yaml
import os
import requests

def get_oauth_client():
    """
    Initializes and returns the OAuth2 client using configuration from config.yaml
    and secrets from environment variables or Streamlit's secrets.toml.
    """
    CONFIG_FILE = 'config.yaml'

    # Load config.yaml
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        oauth_config = config.get('oauth')
        if not oauth_config:
            st.error("OAuth configuration not found under the 'oauth' key.")
            st.stop()
    else:
        st.error(f"Configuration file '{CONFIG_FILE}' not found.")
        st.stop()

    # Load client secret
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", st.secrets.get("GOOGLE_CLIENT_SECRET"))
    if not client_secret:
        st.error("Google Client Secret not found.")
        st.stop()

    # Create OAuth2 client (do NOT pass redirect_uri here)
    return OAuth2(
        client_id=oauth_config['client_id'],
        client_secret=client_secret,
        authorize_endpoint=oauth_config['auth_endpoint'],  # Map to authorize_endpoint
        access_token_endpoint=oauth_config['token_endpoint'],  # Map to access_token_endpoint]
    )

# Initialize the OAuth client
oauth_client = get_oauth_client()


def do_login(oauth_client):
    """
    Handles the login process using Google OAuth.
    Returns user info if logged in.
    """
    if 'user_info' in st.session_state and st.session_state.get('logged_in', False):
        return st.session_state['user_info']

    # Retrieve scope and user_info_endpoint from the configuration
    CONFIG_FILE = 'config.yaml'
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    oauth_config = config.get('oauth')
    scope = oauth_config['scope']
    user_info_endpoint = oauth_config['user_info_endpoint']  # Retrieve user_info_endpoint

    # Construct the authorization URL
    redirect_uri = "http://localhost:8501"
    auth_url = f"{oauth_client.authorize_endpoint}?response_type=code&client_id={oauth_client.client_id}&redirect_uri={redirect_uri}&scope={scope}"

    # Display the login button
    if st.button("Login with Google"):
        # Redirect the user to the authorization URL
        st.experimental_set_query_params()  # Clear any existing query params
        st.write(f'<meta http-equiv="refresh" content="0; url={auth_url}">', unsafe_allow_html=True)

    # Handle the authorization response
    if "code" in st.experimental_get_query_params():
        code = st.experimental_get_query_params()["code"][0]
        try:
            # Debugging: Print the token request data
            st.write("Token Request Data:", {
                "code": code,
                "client_id": oauth_client.client_id,
                "client_secret": oauth_client.client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            })

            # Exchange the authorization code for an access token
            token_response = requests.post(
                oauth_client.access_token_endpoint,
                data={
                    "code": code,
                    "client_id": oauth_client.client_id,
                    "client_secret": oauth_client.client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            token_response.raise_for_status()
            token = token_response.json().get("access_token")

            if not token:
                st.error("Failed to retrieve access token.")
                return None

            # Fetch user info
            user_info_response = requests.get(
                user_info_endpoint,  # Use the endpoint from config.yaml
                headers={"Authorization": f"Bearer {token}"}
            )
            user_info_response.raise_for_status()
            user_info = user_info_response.json()

            # Store user info in session state
            st.session_state['user_info'] = user_info
            st.session_state['logged_in'] = True
            st.rerun()
            return user_info
        except Exception as e:
            st.error(f"An error occurred: {e}")
    return None

def do_logout():
    """
    Clears the login session.
    """
    st.session_state.pop('user_info', None)
    st.session_state.pop('logged_in', None)
    st.rerun()

def get_authenticated_user():
    """
    Retrieves authenticated user info from session.
    """
    return st.session_state.get('user_info') if st.session_state.get('logged_in') else None

# === Main execution ===

user_info = get_authenticated_user()

if not user_info:
    st.info("Please log in to access the dashboards.")
    do_login(oauth_client)
else:
    user_email = user_info.get('email', 'N/A')
    user_name = user_info.get('name', 'User')

    st.sidebar.write(f"Logged in as: {user_name} ({user_email})")
    if st.sidebar.button("Logout"):
        do_logout()

    st.header("Main Dashboard Overview")
    st.write("Welcome to your secure dashboard, powered by Google OAuth.")
