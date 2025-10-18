from scipy import spatial
import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import time
from statsmodels.tsa.arima.model import ARIMA
import folium
from folium.plugins import HeatMap, MarkerCluster
from rtree import index
import os
from datetime import datetime, timedelta
import plotly.express as px
import matplotlib.colors as mcolors
import streamlit.components.v1 as components
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import pydeck as pdk
import warnings
from streamlit_option_menu import option_menu

warnings.filterwarnings("ignore")

# Configuration
CITY_MUNICIPALITY = "Echague"
PROVINCE = "Isabela" 
COUNTRY = "Philippines"
# NOTE: The file path below is local and will not work on a public deployment.
BACKUP_FILE_PATH = "C:\\Users\\jakeq\\OneDrive - isu.edu.ph\\BSCS\\4TH YEAR\\THESIS WRITING 2\\DATASETS\\Barangay.csv"
BACKUP_DATA = None 
GEOLOCATOR = Nominatim(user_agent=f"PNP_{CITY_MUNICIPALITY}_Crime_App")

# Clustering parameters
eps_spatial = 0.0002
eps_temporal = 432000.0
min_samples = 5

# Define PNP colors as Python variables for Plotly charts
PNP_BLUE_HEX = "#0033A0"
PNP_GOLD_HEX = "#FFC72C"

# Page config
st.set_page_config(page_title="PNP Crime Mapping System", layout="wide", initial_sidebar_state="collapsed")

def inject_login_css():
    """
    Merges all global, landing page, and login form CSS rules into a single function,
    now including animations for a more dynamic feel.
    """
    # PNP Logo URL for background
    PNP_BACKGROUND_URL = 'https://media.philstar.com/photos/2019/10/20/pnp-12019-10-1623-33-56_2019-10-20_17-30-33.jpg'
    
    st.markdown(f"""
    <style>
    /* 🎨 --- KEYFRAME ANIMATIONS --- 🎨 */
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}

    @keyframes slideInUp {{
        from {{
            transform: translateY(50px);
            opacity: 0;
        }}
        to {{
            transform: translateY(0);
            opacity: 1;
        }}
    }}

    @keyframes pulse {{
        0% {{ transform: scale(1); box-shadow: 0 6px 20px rgba(0, 51, 160, 0.4); }}
        50% {{ transform: scale(1.03); box-shadow: 0 8px 25px rgba(0, 51, 160, 0.6); }}
        100% {{ transform: scale(1); box-shadow: 0 6px 20px rgba(0, 51, 160, 0.4); }}
    }}

    /* Base class for elements to be animated */
    .animated {{
        animation-duration: 0.8s;
        animation-fill-mode: both;
    }}

    .fade-in {{
        animation-name: fadeIn;
    }}

    .slide-in-up {{
        animation-name: slideInUp;
    }}
    
    /* 🎨 PNP Brand Colors (Unified) */
    :root {{
        --pnp-blue: #0033A0; /* Royal Blue - Primary Action Color */
        --pnp-gold: #FFC72C; /* Gold/Yellow - Secondary Accent */
        --pnp-white: #FFFFFF;
        --pnp-light-gray: #f0f4f7; /* Very light background (Landing) */
        --pnp-dark-text: #212529;
        --dropbox-dark-nav: #272727; /* Dark header color */
        --dropbox-text-dark: #1F1F1F; /* Very dark text for main content */
    }}
    
    /* 1. --- GLOBAL STYLING: MIMIC DROPBOX FONT & PNP BACKGROUND --- */
    [data-testid="stAppViewContainer"] {{
        background-color: var(--pnp-white); 
        background-image: linear-gradient(rgba(0, 51, 160, 0.85), rgba(255, 199, 44, 0.65)), url('{PNP_BACKGROUND_URL}');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        filter: none !important;
    }}
    
    .stApp {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}

    .main {{
        background: transparent;
        padding-top: 5rem;
        max-width: 1400px;
        margin: 0 auto;
    }}
    
    h1, h2, h3, h4, h5, h6, p, span, label, li {{
        color: var(--dropbox-text-dark) !important;
        text-shadow: none !important;
    }}
    
    div[data-testid="stToolbar"] {{
        display: none;
    }}

    /* 2. --- CUSTOM HEADER/NAV BAR STYLING (Fixed Dark Bar) --- */
    div[data-testid="stHeader"] {{
        background-color: var(--dropbox-dark-nav);
        height: 60px;
        position: fixed;
        width: 100%;
        top: 0;
        z-index: 1000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        animation: fadeIn 0.5s ease-in-out;
    }}
    
    .header-logo {{
        color: var(--pnp-white) !important;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        padding: 0 20px;
        line-height: 60px;
        text-shadow: none !important;
    }}
    .header-logo strong {{
        color: var(--pnp-blue) !important;
    }}

    /* 3. --- LANDING PAGE: MAIN MARKETING TITLES --- */
    .landing-main-title, .landing-subtitle {{
        color: var(--pnp-white) !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5) !important;
    }}

    .landing-main-title {{
        font-size: 4.5rem;
        font-weight: 900;
        line-height: 1.1;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        max-width: 600px;
    }}
    .landing-subtitle {{
        font-size: 1.3rem;
        line-height: 1.6;
    }}
    
    /* 4. --- LOGIN FORM: CLEAN & CENTERED --- */
    /* MODIFIED: Changed to semi-transparent white with a blur filter for a modern "frosted glass" look */
    div[data-testid="stForm"] {{
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 3rem;
        border-radius: 15px;
        box-shadow: 0 12px 40px rgba(0, 51, 160, 0.5);
        border: 2px solid var(--pnp-gold);
        animation: slideInUp 0.6s ease-out;
    }}
    
    /* MODIFIED: Make form title and subtitle white as they are outside the form box */
    .form-title, .subtitle {{
        text-align: center;
        color: var(--pnp-white) !important;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.6) !important;
    }}

    .form-title {{
        font-weight: 900;
        margin-bottom: 0.5rem;
        font-size: 2.8rem;
    }}

    .subtitle {{
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }}

    /* 5. --- INPUT FIELD STYLING (PNP Style) --- */
    /* This is inside the form, so it needs to contrast with the light background */
    .stTextInput > label {{
        color: var(--pnp-blue) !important;
        font-weight: 700;
        font-size: 1.1rem;
        text-shadow: none !important; /* Ensure no shadow inside the form */
    }}

    .stTextInput > div > div > input {{
        border: 2px solid var(--pnp-blue);
        border-radius: 8px;
        padding: 12px;
        font-size: 1rem;
        color: var(--pnp-dark-text);
        transition: all 0.3s ease;
    }}

    .stTextInput > div > div > input:focus {{
        border-color: var(--pnp-gold);
        box-shadow: 0 0 0 0.2rem rgba(0, 51, 160, 0.25);
    }}

    /* 6. --- PRIMARY BUTTONS (Login/PNP Style - Gradient & Gold Border) --- */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, var(--pnp-blue) 0%, #0056b3 100%);
        color: white !important;
        border: 2px solid var(--pnp-gold);
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(0, 51, 160, 0.4);
    }}

    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #0056b3 0%, var(--pnp-blue) 100%);
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 51, 160, 0.6);
        border-color: var(--pnp-white);
    }}
                    
    /* 6a. --- Animate the main call to action button on the landing page --- */
    div[data-testid*="portal_access_btn_unique"] button {{ 
        color: white !important;
        animation: pulse 2.5s infinite;
    }}
    
    /* 7. --- SECONDARY BUTTONS (PNP Style) --- */
    .stButton > button:not([kind="primary"]) {{
        background: var(--pnp-light-gray);
        color: var(--pnp-blue);
        border: 1px solid var(--pnp-blue);
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }}

    .stButton > button:not([kind="primary"]):hover {{
        background: var(--pnp-white);
        color: var(--pnp-blue);
        border-color: var(--pnp-gold);
        transform: translateY(-2px);
    }}
    
    /* 8. --- FOOTERS --- */
    .login-footer {{
        text-align: center;
        color: #e0e0e0 !important; /* Lighter color for visibility */
        margin-top: 1.5rem;
        font-size: 0.85rem;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7) !important;
    }}
    
    /* 9. --- LANDING PAGE INFO CARDS --- */
    .card-container {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 25px;
        margin-top: 3rem;
        max-width: 1400px;
        margin-left: auto;
        margin-right: auto;
        padding: 0 20px;
    }}

    .info-card {{
        background: var(--pnp-white);
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 5px 20px rgba(0, 51, 160, 0.15);
        border-top: 5px solid var(--pnp-gold);
        display: flex;
        flex-direction: column;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        opacity: 0;
        animation: slideInUp 0.8s ease-out forwards;
    }}
    .info-card:nth-child(1) {{ animation-delay: 0.2s; }}
    .info-card:nth-child(2) {{ animation-delay: 0.4s; }}
    .info-card:nth-child(3) {{ animation-delay: 0.6s; }}

    .info-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 10px 30px rgba(0, 51, 160, 0.25);
    }}

    .info-card h3 {{
        color: var(--pnp-blue) !important;
        font-weight: 800;
        margin-top: 0;
        margin-bottom: 15px;
        font-size: 1.4rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 10px;
    }}

    .info-card p, .info-card ul {{
        color: var(--pnp-dark-text) !important;
        font-size: 0.95rem;
        line-height: 1.7;
        flex-grow: 1;
    }}

    .info-card ul {{
        list-style-type: '✓ ';
        padding-left: 20px;
        margin-bottom: 1rem;
    }}
    
    .info-card li {{
        margin-bottom: 8px;
    }}
    
    .info-card-button {{
        display: inline-block;
        background: transparent;
        border: 2px solid var(--pnp-blue);
        color: var(--pnp-blue) !important;
        font-weight: 700;
        padding: 10px 20px;
        border-radius: 8px;
        text-align: center;
        transition: all 0.3s ease;
        width: 100%;
        text-decoration: none;
    }}

    .info-card-button:hover {{
        background: var(--pnp-blue);
        color: var(--pnp-white) !important;
        transform: scale(1.05);
    }}
    </style>
    """, unsafe_allow_html=True)

# (Python helper functions like render_kpi_card, authenticate_user etc. are unchanged)
def render_kpi_card(title, value, description):
    html_card = f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-desc">{description}</div>
    </div>
    """
    st.markdown(html_card, unsafe_allow_html=True)

# Initialize session state for authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'show_login' not in st.session_state:
    st.session_state.show_login = False
if 'system_info' not in st.session_state:
    st.session_state.system_info = False

# Initialize session state for data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_clustered' not in st.session_state:
    st.session_state.df_clustered = None
if 'clustering_complete' not in st.session_state:
    st.session_state.clustering_complete = False
if 'geocoding_complete' not in st.session_state:
    st.session_state.geocoding_complete = False
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'clustering_ran' not in st.session_state:
    st.session_state.clustering_ran = False
if 'selected_barangays' not in st.session_state:
    st.session_state.selected_barangays = []
if 'avg_query_time' not in st.session_state: # <-- ADDED
    st.session_state.avg_query_time = 0 # <-- ADDED

# Authentication function
def authenticate_user(username, password):
    valid_credentials = {
        "admin": "admin",
        "user": "user"
    }
    return username in valid_credentials and valid_credentials[username] == password

def system_info():
    inject_login_css()
    
    # Custom Header/Navbar - Title in the top left
    st.markdown("""
        <div data-testid="stHeader">
            <p class="header-logo">Echague <strong>IntelliCrime</strong></p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main title and subtitle with animations
    st.markdown('<div class="animated fade-in"><h1 class="landing-main-title" style="font-size: 3rem; margin-top: 1rem;">🛡️Crime Mapping System</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="animated fade-in" style="animation-delay: 0.3s;"><p class="landing-subtitle" style="font-size: 1.2rem; margin-bottom: 2rem; color: #555555 !important;">A Data-Driven Tool for Safer Communities in Echague</p></div>', unsafe_allow_html=True)
    
    # Two column layout for system info
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Layman's Terms for Core Capabilities (All About the System)
        st.markdown("""
        <div class="info-box">
            <h3>📈 The System's Power: Your Data, Analyzed</h3>
            <ul>
                <li><strong>See Where Crime Happens:</strong> Instantly map out crime locations and visualize high-density areas (hotspots) with interactive maps and heatmaps.</li>
                <li><strong>Find True Hotspots:</strong> Uses a powerful method (**CD-DBSCAN**) to accurately group crimes that happen close together in both **location and time**.</li>
                <li><strong>Predict the Future:</strong> Utilizes statistical modeling (**ARIMA**) to forecast when and where crime incidents are likely to occur next.</li>
                <li><strong>Prioritize Patrols:</strong> Automatically flags the **highest-risk barangays** based on predictions, helping you quickly decide where to send resources.</li>
                <li><strong>Actionable Reports:</strong> Generates clear, easy-to-read reports showing the most common crime types, busiest times, and high-risk areas.</li>
                <li><strong>Smart Deployment:</strong> Helps police command deploy personnel more effectively and prevent future incidents by knowing the risk areas.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        # Direct-to-the-point Workflow Guide
        st.markdown("""
        <div class="info-box" style="animation-delay: 0.2s;">
            <h3>⚙️ Quick Start Workflow</h3>
            <p style="color: var(--pnp-dark-text); font-style: italic; margin-bottom: 1.5rem;">Follow these essential steps to run a complete analysis:</p>
            <ol style="color: var(--pnp-dark-text); line-height: 1.8; font-size: 1rem; padding-left: 20px;">
                <li><strong>Load Data:</strong> Upload your raw crime incident CSV file.</li>
                <li><strong>Get Coordinates:</strong> Run the Geocoding tool to convert barangay names into map coordinates (Latitude/Longitude).</li>
                <li><strong>Identify Hotspots:</strong> Execute the CD-DBSCAN Cluster Analysis to find crime clusters.</li>
                <li><strong>View Maps & Reports:</strong> Go to 'Visualizations' and 'Reports' to see the historical analysis.</li>
                <li><strong>Predict Risk:</strong> Use the 'Forecasting' tools to generate the future risk map for proactive deployment.</li>
                <li><strong>Save Results:</strong> Download the final, analyzed dataset for your records.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Back to home button (Secondary look from CSS)
    if st.button("← Back to Home", use_container_width=True):
        st.session_state.show_login = False
        st.session_state.system_info = False
        st.rerun()

    st.markdown("""
    <div class="footer-text">
        <span style="color: var(--pnp-gold); font-weight: 700;">"Service, Honor, Justice"</span><br>
        © 2025 - Philippine National Police Crime Mapping System for Echague, Isabela
    </div>
    """, unsafe_allow_html=True)

def landing_page():
    inject_login_css()
    
    # Custom Header/Navbar - Title in the top left
    st.markdown("""
        <div data-testid="stHeader">
            <p class="header-logo">Echague <strong>IntelliCrime</strong></p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main Content Layout: Two Columns (Left for Marketing, Right for Info/Action)
    col_left, col_right = st.columns([6, 5])
    
    with col_left:
        # Title and Subtitle - The "Find anything. Protect everything." section
        st.markdown('<div class="animated slide-in-up"><header class="landing-main-title">Echague IntelliCrime</header></div>', unsafe_allow_html=True)
        st.markdown('<div class="animated slide-in-up" style="animation-delay: 0.2s;"><h2 class="landing-main-title">Predict. Analyze. Protect. —IntelliCrime Way.</h2></div>', unsafe_allow_html=True)
        
        # Adding animation delay to the button and description box
        st.markdown('<div class="animated slide-in-up" style="animation-delay: 0.6s;">', unsafe_allow_html=True)
        if st.button("Access the IntelliCrime Portal: Click Here ➡️", use_container_width=True, key="portal_access_btn_unique", type="secondary"):
            st.session_state.show_login = True
            st.session_state.system_info = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="animated slide-in-up" style="animation-delay: 0.4s; background-color: rgba(0, 51, 160, 0.85); padding: 20px; border-radius: 10px; margin-bottom: 2.5rem; max-width: 600px;">
            <p class="landing-subtitle" style="margin: 0; max-width: 100%;">
                A smart system that helps police officers track, understand, and predict crimes. 
                It shows crime locations on a map, highlights areas with many incidents, and helps 
                plan patrols and use resources more effectively.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Info Box / Workflow with animation
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="animated slide-in-up" style="animation-delay: 0.5s; background: var(--pnp-light-gray); padding: 30px; border-radius: 12px; border-left: 5px solid var(--pnp-blue); box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);">
            <h3 style="color: var(--pnp-blue) !important; font-weight: 700; margin-top: 0;">Quick Guide</h3>
            <ol style="color: var(--dropbox-text-dark); line-height: 1.8; font-size: 1rem; padding-left: 20px;">
                <li><strong>Upload Data:</strong> Upload the crime incident file (CSV format).</li>
                <li><strong>Location Mapping:</strong> Convert barangay names to map coordinates (Latitude/Longitude).</li>
                <li><strong>Hotspot Detection:</strong> Identify crime-prone areas using clustering analysis.</li>
                <li><strong>Forecasting:</strong> Predict future crime risks based on past incidents.</li>
                <li><strong>Action Plan:</strong> Focus patrols on barangays with higher predicted crime risks.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# --- NEW INFO CARDS SECTION ---
    st.markdown("""
    <div class="card-container">
        <div class="info-card">
            <h3>About IntelliCrime</h3>
            <p>
                IntelliCrime is an intelligent crime analysis and prediction system designed to support the Philippine National Police in data-driven decision-making. It transforms raw crime reports into actionable insights—showing where, when, and what crimes occur most frequently across Echague, Isabela. Through interactive maps, IntelliCrime visualizes crime hotspots and identifies high-risk areas per barangay.
                <br><br>
                Beyond mapping, IntelliCrime predicts possible crime trends using machine learning, helping the PNP anticipate and prevent incidents before they happen. It also highlights the top 10 most reported crimes, providing a quick overview for resource allocation and patrol planning. With its smart, user-friendly interface and real-time analytics, IntelliCrime empowers law enforcement to see patterns, strengthen strategies, and enhance community safety—one insight at a time.
            </p>
        </div>
        <div class="info-card">
            <h3>Key Features</h3>
            <ul>
                <li><strong>Crime Mapping:</strong> Displays reported crimes on an interactive map and shows crime density with heatmaps.</li>
                <li><strong>Real-Time Dashboard:</strong> Provides instant access to crime data summaries and analytics.</li>
                <li><strong>Time-Based Analysis:</strong> Identifies specific times and days when crimes most frequently occur.</li>
                <li><strong>Top 10 Monitoring:</strong> Lists the most commonly reported crimes in the municipality and per barangay.</li>
                <li><strong>Predictive Analysis:</strong> Uses data patterns to forecast possible crime trends and locations.</li>
                <li><strong>Secure Data Access:</strong> Provides role-based access for authorized PNP personnel.</li>
            </ul>
        </div>
        <div class="info-card">
            <h3>How It Works</h3>
            <p>
                IntelliCrime is designed to make crime analysis simple, smart, and efficient. PNP personnel can upload a CSV file containing crime records. Once uploaded, IntelliCrime automatically updates its database and refreshes all related modules in real time.
                <br><br>
                Using built-in data analytics and forecasting algorithms, the system processes the information to:
            </p>
            <ul>
                <li>Identify the most committed crimes.</li>
                <li>Show the time and day crimes occur most frequently.</li>
                <li>Generate crime heatmaps that visualize risk levels.</li>
                <li>Predict potential crime trends for proactive policing.</li>
            </ul>
            <p>
                With just one upload, IntelliCrime transforms raw police records into powerful, data-driven insights.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div style="text-align: center; margin-top: 6rem; padding-top: 2rem; border-top: 1px solid #eee;">
        <p style="color: #666; font-size: 0.9rem; font-weight: 500;">
            <span style="color: var(--pnp-blue); font-weight: 700;">"Service, Honor, Justice"</span><br>
            "© 2025 - Philippine National Police Crime Mapping System for Echague, Isabela"</span><br>
            Developed by Jake Ronian Quiming
        </p>
    </div>
    """, unsafe_allow_html=True)
    

def login_page():
    """Enhanced login page with improved readability and smooth animations"""
    inject_login_css()
    
    st.markdown("""
    <style>
    /* Hide default header */
    div[data-testid="stHeader"] { 
        display: none !important; 
    }
    
    /* ========== KEYFRAME ANIMATIONS ========== */
    
    /* Slide down from top */
    @keyframes slideInDown {
        from {
            transform: translateY(-100px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    /* Fade in with scale effect */
    @keyframes fadeInScale {
        from {
            transform: scale(0.9);
            opacity: 0;
        }
        to {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    /* Pulsing glow effect for login button */
    @keyframes glowPulse {
        0%, 100% {
            box-shadow: 0 12px 40px rgba(0, 51, 160, 0.5), 
                        0 0 20px rgba(255, 199, 44, 0.3);
        }
        50% {
            box-shadow: 0 15px 50px rgba(0, 51, 160, 0.7), 
                        0 0 30px rgba(255, 199, 44, 0.5);
        }
    }
    
    /* Shimmer effect for title */
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }
    
    /* ========== LOGIN FORM CONTAINER ========== */
    
    div[data-testid="stForm"] {
        animation: fadeInScale 0.7s ease-out !important;
        animation-delay: 0.4s !important;
        opacity: 0;
        animation-fill-mode: forwards !important;
        background: rgba(255, 255, 255, 0.98) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 3px solid var(--pnp-gold) !important;
        box-shadow: 0 15px 50px rgba(0, 51, 160, 0.6), 
                    0 0 30px rgba(255, 199, 44, 0.4) !important;
    }
    
    /* ========== TITLE & SUBTITLE STYLING ========== */
    
    .form-title {
        animation: slideInDown 0.6s ease-out !important;
        text-align: center;
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        letter-spacing: 2px !important;
        margin-bottom: 0.5rem !important;
        
        /* Enhanced white color with strong shadow */
        color: #ffffff !important;
        text-shadow: 
            3px 3px 10px rgba(0, 0, 0, 1),
            0 0 25px rgba(255, 199, 44, 0.8),
            0 0 40px rgba(0, 51, 160, 0.6) !important;
        
        /* Add shimmer effect */
        background: linear-gradient(
            90deg,
            #ffffff 0%,
            #ffd700 50%,
            #ffffff 100%
        );
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: slideInDown 0.6s ease-out, shimmer 3s linear infinite !important;
    }
    
    .subtitle {
        animation: slideInDown 0.6s ease-out !important;
        animation-delay: 0.2s !important;
        opacity: 0;
        animation-fill-mode: forwards !important;
        text-align: center;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 2.5rem !important;
        
        /* Pure white with very strong shadow */
        color: #ffffff !important;
        text-shadow: 
            2px 2px 8px rgba(0, 0, 0, 1),
            0 0 20px rgba(0, 0, 0, 0.8),
            0 0 10px rgba(255, 199, 44, 0.6) !important;
    }
    
    /* ========== INPUT FIELDS ========== */
    
    .stTextInput > label {
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        color: var(--pnp-blue) !important;
        margin-bottom: 8px !important;
    }
    
    .stTextInput > div > div > input {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 2px solid var(--pnp-blue) !important;
        border-radius: 10px !important;
        padding: 14px !important;
        font-size: 1.05rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        transform: translateY(-3px) !important;
        border-color: var(--pnp-gold) !important;
        box-shadow: 
            0 0 0 0.2rem rgba(255, 199, 44, 0.3),
            0 10px 25px rgba(0, 51, 160, 0.3) !important;
    }
    
    .stTextInput > div > div > input:hover {
        border-color: var(--pnp-gold) !important;
    }
    
    /* ========== BUTTONS ========== */
    
    /* Primary Login Button */
    .stButton > button[kind="primary"] {
        animation: glowPulse 2.5s infinite !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        letter-spacing: 1px !important;
        padding: 14px 32px !important;
        margin-top: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-4px) scale(1.03) !important;
        animation: none !important;
        box-shadow: 0 15px 40px rgba(0, 51, 160, 0.7) !important;
    }
    
    .stButton > button[kind="primary"]:active {
        transform: translateY(-1px) scale(0.98) !important;
    }
    
    /* Back Button */
    .stButton > button:not([kind="primary"]) {
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        margin-top: 15px !important;
    }
    
    .stButton > button:not([kind="primary"]):hover {
        transform: translateX(-8px) !important;
        background: var(--pnp-blue) !important;
        color: white !important;
        border-color: var(--pnp-gold) !important;
    }
    
    /* ========== FOOTER ========== */
    
    .login-footer {
        animation: fadeIn 0.8s ease-out !important;
        animation-delay: 0.7s !important;
        opacity: 0;
        animation-fill-mode: forwards !important;
        
        text-align: center;
        font-size: 1rem !important;
        font-weight: 700 !important;
        margin-top: 2rem !important;
        padding: 18px 20px !important;
        border-radius: 10px !important;
        
        /* Pure white text with strong dark background */
        color: #ffffff !important;
        background: rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(8px) !important;
        border: 2px solid rgba(255, 199, 44, 0.3) !important;
        
        text-shadow: 
            2px 2px 8px rgba(0, 0, 0, 1),
            0 0 10px rgba(0, 0, 0, 0.8) !important;
        
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* ========== ERROR MESSAGE STYLING ========== */
    
    div[data-testid="stAlert"] {
        animation: fadeInScale 0.4s ease-out !important;
        border-left: 5px solid #dc3545 !important;
        font-weight: 600 !important;
    }
    
    /* ========== RESPONSIVE ADJUSTMENTS ========== */
    
    @media (max-width: 768px) {
        .form-title {
            font-size: 2.5rem !important;
        }
        .subtitle {
            font-size: 1.1rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add spacing from top
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Center the login form
    _, col2, _ = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<p class="animated slide-in-up" style="animation-delay: 0.2s;">'
                    '<h2 class="landing-main-title" style="text-align: center;">Echague IntelliCrime System</h2></p>',
                    unsafe_allow_html=True)
        
        # Subtitle with enhanced styling
        st.markdown(
            '<p class="subtitle">Secure Access to Crime Analytics System</p>', 
            unsafe_allow_html=True
        )
        
        # Login Form
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input(
                "👤 Username", 
                placeholder="Enter your username",
                help="Enter your authorized PNP credentials"
            )
            password = st.text_input(
                "🔑 Password", 
                type="password", 
                placeholder="Enter your password",
                help="Your password is encrypted and secure"
            )
            
            submit = st.form_submit_button(
                "🔓 Login to System", 
                use_container_width=True, 
                type="primary"
            )
            
            if submit:
                if authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("✅ Login successful! Redirecting...")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error('❌ Invalid credentials. Access denied.')
        
        # Back to Home Button
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.show_login = False
            st.rerun()
        
        # Footer with security notice
        st.markdown(
            '''<div class="login-footer">
                🛡️ <strong>Authorized Personnel Only</strong><br>
                All access attempts are logged and monitored for security purposes.
            </div>''', 
            unsafe_allow_html=True
        )

# ============================================================================
# MAIN APPLICATION FUNCTIONS
# =================================s===========================================

def run_disaggregated_arima(df, selected_barangays, forecast_days):
    """ARIMA forecasting function"""
    forecast_results = []
    forecast_weeks = int(np.ceil(forecast_days / 7))

    df_filtered = df[df['Barangay'].isin(selected_barangays)].dropna(subset=['timestamp']).copy()
    
    if df_filtered.empty:
        return pd.DataFrame(columns=['Incident_Date', 'Predicted_Count', 'Barangay'])
        
    last_historical_date = df_filtered['timestamp'].max().normalize()
    forecast_dates_daily = pd.date_range(
        start=last_historical_date + timedelta(days=1), 
        periods=forecast_days, 
        freq='D'
    )

    for barangay in selected_barangays:
        brgy_df = df_filtered[df_filtered['Barangay'] == barangay].copy()

        if 'timestamp' in brgy_df.columns:
            brgy_df.set_index('timestamp', inplace=True)

        if len(brgy_df.index.normalize().unique()) < 30: 
            continue

        brgy_weekly = brgy_df.resample('W-SUN').size().rename('Count')
        brgy_weekly.index = pd.to_datetime(brgy_weekly.index)
        brgy_weekly = brgy_weekly.asfreq('W-SUN', fill_value=0)

        try:
            model_brgy = ARIMA(brgy_weekly, order=(1, 0, 1)) 
            results_brgy = model_brgy.fit()
            forecast_result = results_brgy.get_forecast(steps=forecast_weeks)
            forecast_series_weekly = forecast_result.predicted_mean
            total_predicted_weekly = forecast_series_weekly.sum()
            total_predicted_incidents = np.maximum(total_predicted_weekly, 0).round().astype(int)

            daily_baseline = total_predicted_incidents / forecast_days
            daily_forecasts = []
            cumulative_prediction = 0
            
            for date in forecast_dates_daily:
                cumulative_prediction += daily_baseline
                
                if cumulative_prediction >= 1:
                    predicted_count_today = int(np.floor(cumulative_prediction))
                    cumulative_prediction -= predicted_count_today
                else:
                    predicted_count_today = 0
                
                daily_forecasts.append({
                    'Incident_Date': date,
                    'Predicted_Count': predicted_count_today,
                    'Barangay': barangay
                })

            df_forecast = pd.DataFrame(daily_forecasts)
            
            if not df_forecast.empty:
                forecast_results.append(df_forecast)

        except Exception as e:
            continue

    if forecast_results:
        return pd.concat(forecast_results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['Incident_Date', 'Predicted_Count', 'Barangay'])

def load_backup_data(file_path):
    """Loads the backup CSV file for geocoding fallback."""
    global BACKUP_DATA
    try:
        df_backup = pd.read_csv(file_path)
        df_backup['Barangay'] = df_backup['Barangay'].astype(str).str.strip()
        df_backup = df_backup[['Barangay', 'Latitude', 'Longitude']].rename(
            columns={'Latitude': 'LAT_BACKUP', 'Longitude': 'LON_BACKUP'}
        )
        BACKUP_DATA = df_backup.set_index('Barangay').to_dict('index')
        return True
    except FileNotFoundError:
        st.warning(f"Backup file not found at: {file_path}")
        return False
    except Exception as e:
        st.warning(f"Backup file unavailable: {e}")
        return False

def geocode_barangay(barangay_name, city, province, country="Philippines"):
    """Enhanced geocoding with backup file priority."""
    brgy_name_clean = str(barangay_name).strip()

    if BACKUP_DATA and brgy_name_clean in BACKUP_DATA:
        coords = BACKUP_DATA[brgy_name_clean]
        if pd.notna(coords['LAT_BACKUP']) and pd.notna(coords['LON_BACKUP']):
            return coords['LAT_BACKUP'], coords['LON_BACKUP']

    time.sleep(1.2)
    
    queries = [
        f"Barangay {brgy_name_clean}, {city}, {province}, {country}",
        f"{brgy_name_clean}, {city}, {province}, {country}",
        f"{brgy_name_clean}, {city}, {province}",
        f"{brgy_name_clean}, Echague",
    ]

    for query in queries: 
        try:
            location = GEOLOCATOR.geocode(query, timeout=10)
            if location:
                lat = float(location.latitude)
                lon = float(location.longitude)
                
                if 16.5 <= lat <= 16.9 and 121.5 <= lon <= 121.9:
                    return lat, lon
        except (ValueError, Exception):
            continue
            
    return None, None

def batch_geocode_dataset(df):
    """Applies geocoding to unique Barangay names."""
    load_backup_data(BACKUP_FILE_PATH)

    unique_barangays = df['Barangay'].dropna().unique()
    total_barangays = len(unique_barangays)

    geo_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, brgy in enumerate(unique_barangays):
        brgy_name = str(brgy).strip()
        progress = (idx + 1) / total_barangays
        progress_bar.progress(progress)
        status_text.text(f"Geocoding: {brgy_name} ({idx + 1}/{total_barangays})")

        lat, lon = geocode_barangay(brgy_name, CITY_MUNICIPALITY, PROVINCE, COUNTRY)
        geo_results.append({'Barangay': brgy, 'LAT': lat, 'LON': lon})

    progress_bar.empty()
    status_text.empty()

    geo_df = pd.DataFrame(geo_results)
    failed_barangays = geo_df[geo_df['LAT'].isna()]['Barangay'].tolist()
    failed_count = len(failed_barangays)
    
    if failed_barangays:
        st.warning(f"⚠️ Could not geocode {failed_count} barangays: {', '.join(map(str, failed_barangays[:10]))}")

    df = df.drop(columns=['LAT', 'LON'], errors='ignore')
    df_geocoded = pd.merge(df, geo_df, on='Barangay', how='left')
    
    initial_size = len(df_geocoded)
    df_geocoded.dropna(subset=['LAT', 'LON'], inplace=True)
    removed_rows = initial_size - len(df_geocoded)

    return df_geocoded, removed_rows, total_barangays, failed_count

def run_cd_dbscan(df):
    """CD-DBSCAN clustering with R-tree spatial indexing."""
    df = df.reset_index(drop=True)
    
    df['LAT'] = df['LAT'].astype(str).str.replace('[^0-9\.\-]', '', regex=True)
    df['LON'] = df['LON'].astype(str).str.replace('[^0-9\.\-]', '', regex=True)
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df.dropna(subset=['LAT', 'LON'], inplace=True)
    
    coords = df[['LAT', 'LON']].values
    times = df['seconds'].values
    visited = set()
    labels = [-1] * len(df)
    cluster_id = 0

    spatial_index = index.Index()
    for i, row in df.iterrows():
        spatial_index.insert(i, (row['LON'], row['LAT'], row['LON'], row['LAT']))

    query_times = []

    def get_neighbors(i):
        start_q = time.perf_counter()
        lat, lon, t = coords[i][0], coords[i][1], times[i]
        box = (lon - eps_spatial, lat - eps_spatial, lon + eps_spatial, lat + eps_spatial)
        spatial_candidates = list(spatial_index.intersection(box))
        neighbors = [j for j in spatial_candidates if abs(times[j] - t) <= eps_temporal]
        end_q = time.perf_counter()
        query_times.append((end_q - start_q) * 1000)
        return neighbors

    for i in range(len(df)):
        if i in visited:
            continue
        neighbors = get_neighbors(i)
        if len(neighbors) < min_samples:
            visited.add(i)
            continue
        labels[i] = cluster_id
        seeds = set(neighbors)
        while seeds:
            current = seeds.pop()
            if current not in visited:
                visited.add(current)
                current_neighbors = get_neighbors(current)
                if len(current_neighbors) >= min_samples:
                    seeds.update(current_neighbors)
            if labels[current] == -1:
                labels[current] = cluster_id
        cluster_id += 1

    avg_query_time = np.mean(query_times) if query_times else 0
    df['cluster'] = labels
    return df, cluster_id, avg_query_time

def save_clustered_csv(df):
    """Saves clustered data to history folder."""
    SAVE_FOLDER = 'dbscan_history' 
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dbscan_clustered_output_{timestamp}.csv"
    save_path = os.path.join(SAVE_FOLDER, filename)

    try:
        df.to_csv(save_path, index=False)
        return {"status": "success", "path": save_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def main_application():
    """Main crime mapping application with a new dashboard-style UI"""

    # Initialize session state for view navigation
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'dashboard'
    if 'username' not in st.session_state:
        st.session_state.username = 'Guest'


    PNP_BACKGROUND_URL = 'https://media.philstar.com/photos/2019/10/20/pnp-12019-10-1623-33-56_2019-10-20_17-30-33.jpg'
    st.markdown(f"""
    <style>
    /* 🎨 --- KEYFRAME ANIMATIONS --- */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}


    /* 🎯 MAIN APP BACKGROUND (Clean) --- */
    [data-testid="stAppViewContainer"] {{
        background-color: #f0f2f6; /* A clean, light gray background */
        background-image: none; /* Ensure no background image on main app */
    }}
    [data-testid="stAppViewContainer"] > .main {{
        background-color: transparent;
        background-image: none; /* Ensure no background image on main content */
    }}

    /* 🎯 NEW HEADER BACKGROUND CONTAINER --- */
    /* This rule applies the background image to the header area */
    .header-background-container {{
        background-image: linear-gradient(rgba(130, 163, 248, 0.8), rgba(0, 51, 60, 0.85)), url('{PNP_BACKGROUND_URL}');
        background-size: cover;
        background-position: center;
        padding: 2rem 1.5rem 1.5rem 1.5rem; /* Added padding */
        border-radius: 10px; /* Added rounded corners */
        margin-bottom: 2rem; /* Added space below header */
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}

    /* 🎯 HEADER TEXT (Title & Caption) --- */
    /* This new rule makes *only* the text inside the header white */
    .header-background-container .main-app-title,
    .header-background-container .stCaption,
    .header-background-container div[data-testid="stCaptionContainer"] p
    {{
        color: white !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7) !important;
    }}
    
    .main-app-title {{
        font-weight: 900; font-size: 3rem; text-shadow: 2px 2px 5px rgba(0,0,0,0.8);
        border-bottom: 4px solid #FFC72C; padding-bottom: 10px; margin-bottom: 0;
    }}

    /* 🎨 --- STYLED CONTAINERS FOR CONTENT (White Boxes) --- */
    .styled-container {{
        background-color: rgba(255, 255, 255, 0.98);
        border-radius: 15px; padding: 2rem; margin-bottom: 2rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.2);
        border: 1px solid #ddd;
        animation: fadeIn 0.6s ease-out;
    }}
    
    /* * ✅ --- COMPREHENSIVE OVERRIDE BLOCK ---
     * This block resets text color to dark for ALL elements inside styled-container.
     */
    .styled-container div,
    .styled-container p,
    .styled-container span,
    .styled-container li,
    .styled-container label,
    .styled-container h1, .styled-container h2, .styled-container h3, 
    .styled-container h4, .styled-container h5, .styled-container h6,
    .styled-container .stMarkdown p, .styled-container .stMarkdown li,
    .styled-container div[data-testid="stMarkdownContainer"] p,
    .styled-container div[data-testid="stText"],
    .styled-container .element-container p,
    .styled-container .stAlert p, .styled-container .stWarning p,
    .styled-container .stSuccess p, .styled-container .stInfo p,
    .styled-container .stRadio > label,
    .styled-container .stRadio > div > label > div,
    .styled-container .stMetric,
    .styled-container .stMetric div,
    .styled-container .stFileUploader label,
    .styled-container .stSlider > label,
    .styled-container .stSelectbox > label
    {{
        color: #212529 !important; /* Dark text for readability */
        text-shadow: none !important;
    }}

    /* 🎨 --- ACTION CARD STYLES (Menu Boxes) --- */
    .action-card {{
        background-color: #f8f9fa; color: white; padding: 2rem 1.5rem;
        border-radius: 15px; text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease; 
        min-height: 280px;
        display: flex; flex-direction: column; justify-content: space-between;
        align-items: center; box-shadow: 0 5px 20px rgba(0,0,0,0.4);
        border: 2px solid rgba(255, 255, 255, 0.8); margin-bottom: 1rem;
    }}
    .action-card:hover {{
        transform: translateY(-10px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.5);
    }}
    .action-card h3 {{
        font-size: 1.8rem; font-weight: 700; color: white !important;
    }}
    .action-card p {{
        font-size: 1rem; color: white !important; opacity: 0.95;
    }}
    .card-viz {{ background: linear-gradient(135deg, #663399, #8A2BE2); }}
    .card-forecast {{ background: linear-gradient(135deg, #FFC72C, #FFA500); }}
    .card-reports {{ background: linear-gradient(135deg, #DC143C, #FF4500); }}
    .card-download {{ background: linear-gradient(135deg, #0033A0, #0056b3); }}

    /* 🎨 --- BUTTON STYLES --- */
    .stButton > button {{ text-shadow: none !important; }}
    
    .stButton > button[kind="primary"] {{
        background: rgba(255, 255, 255, 0.2) !important; color: white !important;
        border: 2px solid white !important; font-weight: bold !important; 
        border-radius: 8px !important; transition: all 0.3s ease !important;
    }}
    .stButton > button[kind="primary"]:hover {{ 
        background: white !important; 
        color: #333 !important; 
    }}
    
    .stButton > button[kind="primary"]:active {{
        background: #0056b3 !important; 
        color: white !important;
        border: 2px solid #004085 !important;
        transform: scale(0.98);
    }}
    
    /* Tooltip text color - comprehensive targeting */
    [data-testid="stTooltipIcon"],
    div[role="tooltip"],
    div[role="tooltip"] *,
    .stTooltip,
    .stTooltip *,
    [data-testid="stTooltipHoverTarget"] + div,
    [data-testid="stTooltipHoverTarget"] + div *,
    [data-baseweb="tooltip"],
    [data-baseweb="tooltip"] *,
    button[title],
    button[disabled]:hover::after {{
        color: #212529 !important;
        text-shadow: none !important;
    }}
    
    div[role="tooltip"] div,
    div[role="tooltip"] p,
    div[role="tooltip"] span,
    [data-baseweb="tooltip"] div,
    [data-baseweb="tooltip"] p,
    [data-baseweb="tooltip"] span {{
        color: #212529 !important;
        text-shadow: none !important;
    }}
    
    div[role="tooltip"],
    [data-baseweb="tooltip"] {{
        background-color: rgba(255, 255, 255, 0.98) !important;
        border: 1px solid #ccc !important;
    }}
    
    .back-button-container,
    .logout-container {{
        margin-bottom: 0px !important; /* Adjusted margin */
    }}

    /* Back to Dashboard & Logout Buttons */
    .back-button-container div[data-testid="stButton"] > button,
    .logout-container div[data-testid="stButton"] > button {{
        background: linear-gradient(135deg, #0033A0 0%, #0056b3 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 8px rgba(0, 51, 160, 0.2) !important;
        text-shadow: none !important;
    }}
    
    .back-button-container div[data-testid="stButton"] > button:hover,
    .logout-container div[data-testid="stButton"] > button:hover {{
        background: linear-gradient(135deg, #0056b3 0%, #0033A0 100%) !important;
        transform: translateY(-3px) scale(1.03) !important;
        box-shadow: 0 6px 15px rgba(0, 51, 160, 0.4) !important;
    }}
    
    .back-button-container div[data-testid="stButton"] > button *,
    .logout-container div[data-testid="stButton"] > button * {{
        color: white !important;
        text-shadow: none !important;
    }}
    
    </style>
    """, unsafe_allow_html=True)

    # --- CSS STYLES ---
    st.markdown("""
    <style>
    /* Style for the new blue container for the uploader */
    .upload-container {
        background-color: #0d6efd; /* Bootstrap primary blue */
        padding: 2rem 2rem 1rem 2rem; /* top, right, bottom, left */
        border-radius: 0.5rem;
        margin-bottom: 2rem; /* Add space below the container */
    }
    /* Style for the header text inside the blue container */
    .upload-container h3 {
        color: white;
        margin-top: 0;
    }
    /* Make the file uploader label white */
    .upload-container .st-emotion-cache-1v0mbdj > p {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


    # --- HEADER SECTION ---
    # Create header with background using container
    st.markdown(f"""
    <div class="header-background-container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="main-app-title">Echague IntelliCrime System</h1>
                <p style="color: white; text-shadow: 1px 1px 3px rgba(0,0,0,0.7); margin-top: 0.5rem;">
                    ⚪ Logged in as: <strong>{st.session_state.username}</strong> | A Data-Driven Tool for Safer Communities
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # NOTE: The old fixed-position logout button and its style have been removed from here.

    # --- VIEW ROUTING ---
    if st.session_state.current_view == 'dashboard':


        # Use columns to position the header on the left and logout button on the right
        col_header, col_logout = st.columns([0.8, 0.2])
        
        with col_header:
            st.header("📂 Upload & Process Records")

        with col_logout:
            # This button is now inside the blue container at the top right
            if st.button("➜] Logout", use_container_width=True):
                # Add your logout logic here. For example, resetting session state:
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()

        # --- Uploader and Processor Logic ---
        file = st.file_uploader("Upload your crime records CSV file.", type=["csv"], help="...")
        
        
        if file:
            df = pd.read_csv(file)
            
            original_size = len(df)
            required_cols = ['Date_Occ', 'Crime_Type', 'Barangay']
            
            if not all(col in df.columns for col in required_cols):
                st.error(f"❌ **Missing Required Columns:** Please ensure your file contains {', '.join(required_cols)}.")
                st.stop()
            
            df['timestamp'] = pd.to_datetime(df['Date_Occ'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            df['seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
            
            cleaned_size = len(df)
            removed_rows = original_size - cleaned_size
            
            st.success(f"✅ **File Processed:** {cleaned_size} valid records loaded. ({removed_rows} rows removed due to missing/invalid dates)")
            
            st.session_state.df = df
            needs_geocoding = 'LAT' not in df.columns or 'LON' not in df.columns
            
            st.markdown("---")
            action_cols = st.columns(2)
            
            with action_cols[0]:
                if needs_geocoding:
                    st.info("Your dataset requires geocoding to map crime locations.")
                    if st.button("🌍 **Step 1: Start Geocoding & Clustering**", use_container_width=True):
                        # GEOCODING
                        with st.spinner("Geocoding barangays... This may take a moment."):
                            df_geocoded, removed, total_brgy, failed = batch_geocode_dataset(df)
                        if df_geocoded.empty:
                            st.error("Geocoding failed for all barangays. Cannot proceed.")
                            st.stop()
                        st.success(f"Geocoding complete: {total_brgy - failed}/{total_brgy} successful.")
                        
                        # CLUSTERING
                        with st.spinner("Running CD-DBSCAN clustering..."):
                            df_clustered, clusters_found, avg_query = run_cd_dbscan(df_geocoded)
                        st.session_state.df_clustered = df_clustered
                        st.session_state.clustering_complete = True
                        st.session_state.avg_query_time = avg_query
                        st.success("Clustering complete!")
                        st.toast("✅ Analysis Finished!")
                        st.rerun()
                else:
                    st.info("Dataset already contains LAT/LON columns.")
                    if st.button("🚀 **Step 1: Run CD-DBSCAN Clustering**", use_container_width=True):
                        with st.spinner("Running clustering analysis..."):
                            df_clustered, clusters_found, avg_query = run_cd_dbscan(df)
                        st.session_state.df_clustered = df_clustered
                        st.session_state.clustering_complete = True
                        st.session_state.avg_query_time = avg_query
                        st.success("Clustering complete!")
                        st.toast("✅ Analysis Finished!")
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # --- MENU CARDS (1 ROW of 4 COLUMNS) ---
        st.header("Main Menu")
        is_disabled = not st.session_state.get('clustering_complete', False)
        disabled_tooltip = "Please upload and process a dataset first to enable this feature."

        col1, col2, col3, col4 = st.columns(4, gap="medium")

        with col1:
            st.markdown('<div class="action-card card-viz"><h3>🗺️ Visualizations</h3><p>Explore crime data through interactive maps, and heatmaps.</p>', unsafe_allow_html=True)
            st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
            if st.button("Open Visualizations", key="viz_btn", use_container_width=True, disabled=is_disabled, help=disabled_tooltip if is_disabled else "Go to the maps page", type="primary"):
                st.session_state.current_view = 'visualizations'; st.rerun()
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="action-card card-forecast"><h3>📈 Forecasting</h3><p>Forecast future crime trends and use the predictive risk map.</p>', unsafe_allow_html=True)
            st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
            if st.button("Go to Forecasting", key="forecast_btn", use_container_width=True, disabled=is_disabled, help=disabled_tooltip if is_disabled else "Access predictive tools", type="primary"):
                st.session_state.current_view = 'forecasting'; st.rerun()
            st.markdown('</div></div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="action-card card-reports"><h3>📊 Reports</h3><p>Generate detailed statistical reports on crime types and high-risk areas.</p>', unsafe_allow_html=True)
            st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
            if st.button("View Reports", key="reports_btn", use_container_width=True, disabled=is_disabled, help=disabled_tooltip if is_disabled else "Generate statistical reports", type="primary"):
                st.session_state.current_view = 'reports'; st.rerun()
            st.markdown('</div></div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="action-card card-download"><h3>💾 Download</h3><p>Download data analysis.</p>', unsafe_allow_html=True)
            st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
            if st.button("Download Data", key="download_btn", use_container_width=True, disabled=is_disabled, help=disabled_tooltip if is_disabled else "Download the processed CSV", type="primary"):
                st.session_state.current_view = 'download'; st.rerun()
            st.markdown('</div></div>', unsafe_allow_html=True)
        
    # --- RENDER CONTENT FOR OTHER VIEWS ---
    else:
        if st.button("← Back to Dashboard", use_container_width=True, help="Return to the main dashboard", key="back_to_dashboard", type="primary"):
            st.session_state.current_view = 'dashboard'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        df = st.session_state.get('df_clustered')
        if df is None:
            st.warning("No data processed. Please return to the dashboard and upload a file.")
            st.stop()
            
        # --- Visualizations Section ---

        if st.session_state.current_view == 'visualizations':
            st.header("🗺️ Visualizations")
            
            # 1. Prepare data for Cluster Visualizations (Valid LAT/LON and assigned cluster != -1)
            df_clustered = df[(df['LAT'] != 0) & (df['LON'] != 0) & (df['cluster'] != -1)].copy()

            if df_clustered.empty:
                st.warning("No valid, clustered data available for visualization.")
            else:
                st.subheader("Crime Cluster Visualization")

                # User Choice for Map Type
                map_type = st.radio(
                    "Select Cluster Map Type:",
                    ('Density Map', 'Scatter Plot'),
                    horizontal=True
                )

                # Use a sample for better performance, common to both Plotly maps
                df_sample = df_clustered.sample(min(25000, len(df_clustered)))

                # 2. Map Rendering based on User Choice

                if map_type == 'Density Map':
                    st.caption("Shows the concentration of crime clusters.")

                    fig = px.density_mapbox(
                        df_sample.sample(min(10000, len(df_sample))), # Smaller sample is often better for Density
                        lat='LAT',
                        lon='LON',
                        z='cluster', # Use cluster as the density value
                        radius=15,
                        center=dict(lat=df_clustered['LAT'].mean(), lon=df_clustered['LON'].mean()),
                        zoom=10,
                        mapbox_style="open-street-map",
                        title="Crime Cluster Density Map",
                        color_continuous_scale='Viridis'
                    )

                elif map_type == 'Scatter Plot':
                    # Scatter Plot is configured for a single color
                    st.caption("Shows individual crime points in a single color.")

                    fig = px.scatter_mapbox(
                        df_sample,
                        lat='LAT',
                        lon='LON',
                        hover_data=['cluster'],
                        center=dict(lat=df_clustered['LAT'].mean(), lon=df_clustered['LON'].mean()),
                        zoom=10,
                        mapbox_style="open-street-map",
                        title="Crime Cluster Scatter Plot (Single Color)",
                        opacity=0.8
                    )

                    # Set a fixed color and size for all markers
                    fig.update_traces(marker=dict(color='blue', size=5)) 

                # Display the Plotly figure (common to both map types)
                fig.update_layout(height=600, margin={"r":0,"t":50,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)

            # 2. INTERACTIVE HEAT MAP (All Valid Crimes)

            st.subheader("Interactive Heat Map (Overall Crime Density)")
            st.caption("Visualize the overall spatial density of all reported incidents.")
            
            # Filter for all valid crime locations (requires only valid LAT/LON)
            df_all_valid = df[(df['LAT'] != 0) & (df['LON'] != 0)].copy()

            if not df_all_valid.empty:
                map_center = [df_all_valid['LAT'].mean(), df_all_valid['LON'].mean()]
                m = folium.Map(location=map_center, zoom_start=12)
                # Use the data frame containing all valid crime locations
                heat_data = [[row['LAT'], row['LON'], 0.5] for _, row in df_all_valid.iterrows()] # Adding intensity value
                HeatMap(heat_data, radius=15).add_to(m)
                
                # Add a marker cluster for noise/individual points
                marker_cluster = MarkerCluster().add_to(m)
                df_noise = df[(df['LAT'] != 0) & (df['LON'] != 0) & (df['cluster'] == -1)].copy().sample(min(2000, len(df[(df['cluster'] == -1)])))

                for _, row in df_noise.iterrows():
                    folium.Marker(
                        location=[row['LAT'], row['LON']],
                        popup=f"Crime Type: {row.get('Crime_Type', 'N/A')}<br>Barangay: {row['Barangay']}",
                        icon=folium.Icon(color='blue', icon='info-sign')
                    ).add_to(marker_cluster)

                components.html(m._repr_html_(), height=600)
            else:
                st.write("No valid data available for the Heat Map.")
                
            st.markdown("---")

            st.subheader("🎯 Predictive Risk Assessment")

            required_cols = ['LAT', 'LON', 'Barangay', 'timestamp']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Missing one or more required columns for risk assessment: {required_cols}.")
                st.stop()
            else:
                
                all_barangays = sorted(df['Barangay'].dropna().unique().tolist())
                selected_barangays = all_barangays
                
                st.info(f"Analysis will automatically cover **all {len(selected_barangays)} barangays** of Echague.")

                forecast_days = st.slider(
                    "Forecast Horizon (Days)", 
                    min_value=7,
                    max_value=60, 
                    value=30, 
                    step=1
                )

                if not selected_barangays:
                    st.error("No barangay data available to run the predictive analysis.")

                else:
                    if st.button(f"Generate Municipality-Wide Forecast 📈", type="secondary"):
                        with st.spinner(f"Running disaggregated ARIMA forecast for {len(selected_barangays)} barangays over {forecast_days} days..."):

                            # --- SIMULATED FORECASTING (Placeholder) ---
                            # Replace this block with your actual forecasting function call
                            if 'run_disaggregated_arima' in globals():
                                forecast_data = run_disaggregated_arima(df, selected_barangays, forecast_days)
                            else:
                                # Placeholder/simulated data generation
                                dates = pd.to_datetime(pd.date_range(start=pd.Timestamp.now(), periods=forecast_days, freq='D'))
                                np.random.seed(42)
                                all_data = []
                                for b in selected_barangays:
                                    # Artificially increase counts for certain barangays for demonstration
                                    counts = np.random.randint(0, 5, size=forecast_days) + (5 if 'San Isidro' in b or 'Victoria' in b else 0)
                                    counts[counts < 0] = 0

                                    for i, date in enumerate(dates):
                                        all_data.append({
                                            'Incident_Date': date,
                                            'Predicted_Count': counts[i],
                                            'Barangay': b
                                        })
                                forecast_data = pd.DataFrame(all_data)
                            # --- END SIMULATED FORECASTING ---

                            st.session_state.forecast_data = forecast_data
                            st.session_state.selected_barangays = selected_barangays

                            if not forecast_data.empty:
                                st.success(f"Forecast complete for all {len(selected_barangays)} Barangays.")
                            else:
                                st.warning("Forecast completed, but no data was generated. Check data sufficiency.")
                            st.rerun()

                    # --- DISPLAY AND RISK VISUALIZATION ---
                    if 'forecast_data' in st.session_state and st.session_state.forecast_data is not None and not st.session_state.forecast_data.empty:

                        df_f = st.session_state.forecast_data.copy()
                        df_f['Incident_Date'] = pd.to_datetime(df_f['Incident_Date'])

                        # Data Aggregation for Risk Scoring
                        df_risk = df_f.groupby('Barangay')['Predicted_Count'].sum().reset_index()

                        # Get mean coordinates for each barangay (for map)
                        df_coords = df[['Barangay', 'LAT', 'LON']].groupby('Barangay').mean().reset_index()
                        df_risk = pd.merge(df_risk, df_coords, on='Barangay', how='inner')

                        # --- Risk Alert Logic ---
                        st.markdown("<h3 style='color: #cc0000 !important;'>🚨 High-Risk Assessment Alert 🚨</h3>", unsafe_allow_html=True)
                        N = 4 # Number of top high-risk barangays to display
                        df_high_risk = df_risk.sort_values(by='Predicted_Count', ascending=False).head(N)

                        # Threshold for High Risk: Any barangay in the top N with a count > 0
                        high_risk_barangays = df_high_risk[df_high_risk['Predicted_Count'] > 0]['Barangay'].tolist()

                        if high_risk_barangays:
                            barangay_list_html = ''.join([f"<div class='barangay-item'>🔴 {b} (Predicted Incidents: {int(df_high_risk.loc[df_high_risk['Barangay'] == b, 'Predicted_Count'].iloc[0])})</div>" for b in high_risk_barangays])
                            alert_html = f"""
                            <style>
                                .risk-alert-box {{
                                    /* Changed the background to a lightsalmon gradient */
                                    background: linear-gradient(135deg, #FFA07A 0%, #FA8072 100%); 
                                    /* Updated border to a darker salmon color */
                                    border: 3px solid #E9967A; 
                                    border-radius: 15px; padding: 30px; margin: 20px 0;
                                    box-shadow: 0 8px 16px rgba(204, 0, 0, 0.3); color: white;
                                }}
                                .risk-alert-box h2 {{
                                    color: white; text-align: center; font-size: 28px; margin-bottom: 20px;
                                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); border-bottom: 2px solid rgba(255, 255, 255, 0.3);
                                    padding-bottom: 15px;
                                }}
                                .risk-alert-box p {{ font-size: 16px; line-height: 1.6; }}
                                .risk-alert-box ul {{ margin: 15px 0; }}
                                .risk-alert-box li {{ padding: 8px 0; font-size: 16px; }}
                                .barangay-item {{
                                    background: rgba(255, 255, 255, 0.1); padding: 10px; margin: 8px 0;
                                    border-radius: 8px; font-weight: bold; font-size: 18px;
                                }}
                            </style>
                            <div class="risk-alert-box">
                                <h2>🚨 High Accident Risk Detected ({forecast_days} Days) 🚨</h2>
                                <p style="text-align: center; font-size: 18px; font-weight: bold;">Immediate action required in the following <strong>Top {N}</strong> barangays based on predicted incidents:</p>
                                <div style="margin: 20px 0;">{barangay_list_html}</div>
                                <p style="font-weight: bold; text-align: center; font-size: 18px; margin-top: 25px; margin-bottom: 15px;">Recommended Actions:</p>
                                <ul style="list-style-type: none; padding: 0; margin: 0 auto; max-width: 600px;">
                                    <li style="background: rgba(255, 255, 255, 0.1); padding: 12px; margin: 8px 0; border-radius: 8px;">✓ Increase patrol frequency and visibility.</li>
                                    <li style="background: rgba(255, 255, 255, 0.1); padding: 12px; margin: 8px 0; border-radius: 8px;">✓ Conduct focused community engagement in these areas.</li>
                                    <li style="background: rgba(255, 255, 255, 0.1); padding: 12px; margin: 8px 0; border-radius: 8px;">✓ Review existing security measures and infrastructure.</li>
                                </ul>
                            </div>
                            """
                            st.markdown(alert_html, unsafe_allow_html=True)
                        else:
                            st.info(f"The forecast did not identify any barangays with a high enough predicted incident count to trigger a high-risk alert for the next {forecast_days} days.")

                        st.markdown("---")

                        # --- Filter to show ONLY Top N High Risk Barangays ---
                        df_risk_map = df_risk[df_risk['Barangay'].isin(high_risk_barangays)].copy()
                        
                        # Set color to red for all displayed barangays
                        if not df_risk_map.empty:
                            df_risk_map['Color_R'] = 255
                            df_risk_map['Color_G'] = 0
                            df_risk_map['Color_B'] = 0
                            df_risk_map['Risk_Level'] = 'High Risk'

                        # --- Pydeck Visualization (Top N Only) ---
                        st.subheader(f"Predictive Risk Map - Top {N} High-Risk Barangays ({forecast_days} Days)")

                        if not df_risk_map.empty and df_risk_map['Predicted_Count'].sum() > 0:
                            max_count = df_risk_map['Predicted_Count'].max()
                            ELEVATION_SCALE = 5000 / (max_count if max_count > 0 else 1)

                            view_state = pdk.ViewState(
                                latitude=df_risk_map['LAT'].mean(),
                                longitude=df_risk_map['LON'].mean(),
                                zoom=11.5,
                                pitch=50,
                                bearing=-20
                            )

                            column_layer = pdk.Layer(
                                'ColumnLayer',
                                data=df_risk_map,
                                get_position='[LON, LAT]',
                                get_elevation=f'Predicted_Count * {ELEVATION_SCALE}',
                                elevation_scale=1,
                                radius=300,
                                get_fill_color='[Color_R, Color_G, Color_B, 200]',
                                pickable=True,
                                extruded=True,
                                auto_highlight=True,
                            )

                            text_layer = pdk.Layer(
                                'TextLayer',
                                data=df_risk_map,
                                get_position='[LON, LAT]',
                                get_text='Barangay',
                                get_size=12,
                                get_color='[255, 255, 255, 255]',
                                get_elevation=f'Predicted_Count * {ELEVATION_SCALE}',
                                get_text_anchor='"middle"',
                                get_alignment_baseline='"center"',
                                get_pixel_offset=[0, 30],
                                billboard=True,
                                background=True,
                                get_background_color='[0, 0, 0, 150]',
                            )
                            
                            tooltip = {
                                "html": "<b>Barangay:</b> {Barangay}<br/><b>Risk Level:</b> {Risk_Level}<br/><b>Predicted Incidents:</b> {Predicted_Count}",
                                "style": {"backgroundColor": "#333", "color": "white", "border-radius": "5px", "padding": "5px"}
                            }

                            r = pdk.Deck(
                                layers=[column_layer, text_layer],
                                initial_view_state=view_state,
                                tooltip=tooltip,
                                map_style='mapbox://styles/mapbox/streets-v11',
                            )

                            st.pydeck_chart(r)

                            # Legend and Explanation
                            st.markdown(f"""
                            <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
                                <p style='font-weight: bold;'>Risk Legend for the Next {forecast_days} Days:</p>
                                <div style="display: flex; align-items: center; gap: 10px; font-weight: 600;">
                                    <span style="color: #ff4444; font-size: 24px;">&#9632;</span>
                                    <span>High Risk (Top {N} Barangays Only)</span>
                                </div>
                                <p style='font-style: italic; margin-top: 10px;'>
                                    The <b>height</b> of each column represents the <b>total predicted incidents</b>.
                                    Taller columns indicate higher risk.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("No high-risk barangays identified to display on the 3D Risk Map.")
            
        # --- Forecasting Section ---
        elif st.session_state.current_view == 'forecasting':
            st.header("📈 Time Series Forecasting")
            st.caption("Monthly ARIMA forecast for the total crime incidents across the municipality.")
            
            df['month'] = df['timestamp'].dt.to_period('M')
            cluster_counts_monthly = df.groupby('month').size().rename('Count')
            cluster_counts_monthly.index = cluster_counts_monthly.index.to_timestamp()
            
            if not cluster_counts_monthly.empty:
                try:
                    # Forecast for 3 months (steps=3)
                    model = ARIMA(cluster_counts_monthly, order=(1, 1, 1))
                    results = model.fit()
                    forecast = results.forecast(steps=3) # Forecast 3 steps (months)

                    df_observed = cluster_counts_monthly.reset_index()
                    df_observed.columns = ['Date', 'Count']
                    df_observed['Type'] = 'Observed'

                    df_forecast = forecast.reset_index()
                    df_forecast.columns = ['Date', 'Count']
                    df_forecast['Type'] = 'Forecast'

                    df_plot = pd.concat([df_observed, df_forecast])

                    col_chart, col_summary = st.columns([3, 1])
                    
                    with col_chart:
                        chart_type = st.radio("Select Chart Type:", ('Line', 'Bar'), horizontal=True)
                        
                        # FIX: Replaced CSS variables with Python hex variables in color_discrete_map
                        color_map = {'Observed': PNP_BLUE_HEX, 'Forecast': PNP_GOLD_HEX}
                        
                        if chart_type == 'Line':
                            fig2 = px.line(df_plot, x='Date', y='Count', color='Type',
                                        title="Monthly Crime Incident Forecast (ARIMA)", markers=True,
                                        color_discrete_map=color_map)
                        else:
                            fig2 = px.bar(df_plot, x='Date', y='Count', color='Type',
                                        title="Monthly Crime Incident Forecast (ARIMA)", barmode='group',
                                        color_discrete_map=color_map)
                        
                        st.plotly_chart(fig2, use_container_width=True)

                    with col_summary:
                        st.subheader("Summary")
                        st.markdown(f"**Forecast for Next 3 Months:**")
                        for index, row in df_forecast.iterrows():
                            st.metric(label=row['Date'].strftime("%b %Y"), value=f"{int(row['Count']):,}")
                        
                except Exception as e:
                    st.error(f"Forecasting error: Insufficient historical data (need at least two months) or model fitting failed. Error: {e}")
            else:
                st.warning("No time-series data available for forecasting.")
            
        # --- Reports Section ---
        elif st.session_state.current_view == 'reports':
            st.header("📊 Statistical Reports")
            
           # 1. TOP 10 CRIME TYPES
            st.subheader("🔴 Top 10 Crime Types")

            if 'Crime_Type' in df.columns:
                filter_col1, filter_col2 = st.columns([1, 3])
                
                with filter_col1:
                    if 'Barangay' in df.columns:
                        barangay_list = ['All'] + sorted(df['Barangay'].dropna().unique().tolist())
                        selected_barangay = st.selectbox(
                            "Filter by Barangay:",
                            options=barangay_list,
                            key="crime_type_barangay_filter"
                        )
                    else:
                        selected_barangay = 'All'
                
                if selected_barangay == 'All':
                    filtered_df = df
                    title_suffix = " (All Barangays)"
                else:
                    filtered_df = df[df['Barangay'] == selected_barangay]
                    title_suffix = f" ({selected_barangay})"
                
                crime_type_counts = filtered_df.groupby('Crime_Type').size().reset_index(name='Count')
                crime_type_counts = crime_type_counts.sort_values('Count', ascending=False).head(10)
                crime_type_counts.insert(0, 'No.', range(1, len(crime_type_counts) + 1))
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write(f"**Crime Type Statistics{title_suffix}**")
                    st.dataframe(
                        crime_type_counts.style.set_properties(**{'text-align': 'left'}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    fig_crime = px.bar(
                        crime_type_counts, 
                        x='Count', 
                        y='Crime_Type',
                        orientation='h',
                        title=f"Top 10 Crime Types{title_suffix}",
                        labels={'Crime_Type': 'Crime Type', 'Count': 'Number of Incidents'},
                        color='Count',
                        color_continuous_scale='Sunset'
                    )
                    fig_crime.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                    st.plotly_chart(fig_crime, use_container_width=True)
            else:
                st.warning("Crime_Type column not found in dataset.")

            st.markdown("---")
            
            # 2. TOP 10 BARANGAYS
            st.subheader("📍 Top 10 Barangays with Most Crimes")

            if 'Barangay' in df.columns:
                filter_col1, filter_col2 = st.columns([1, 3])
                
                with filter_col1:
                    if 'Crime_Type' in df.columns:
                        crime_type_list = ['All'] + sorted(df['Crime_Type'].dropna().unique().tolist())
                        selected_crime_type = st.selectbox(
                            "Filter by Crime Type:",
                            options=crime_type_list,
                            key="barangay_crime_type_filter"
                        )
                    else:
                        selected_crime_type = 'All'
                
                if selected_crime_type == 'All':
                    filtered_df = df
                    title_suffix = " (All Crime Types)"
                else:
                    filtered_df = df[df['Crime_Type'] == selected_crime_type]
                    title_suffix = f" ({selected_crime_type})"
                
                barangay_counts = filtered_df.groupby('Barangay').size().reset_index(name='Count')
                barangay_counts = barangay_counts.sort_values('Count', ascending=False).head(10)
                barangay_counts.insert(0, 'No.', range(1, len(barangay_counts) + 1))
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write(f"**Barangay Crime Statistics{title_suffix}**")
                    st.dataframe(
                        barangay_counts.style.set_properties(**{'text-align': 'left'}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    fig_barangay = px.bar(
                        barangay_counts, 
                        x='Count', 
                        y='Barangay',
                        orientation='h',
                        title=f"Top 10 Barangays by Crime Count{title_suffix}",
                        labels={'Barangay': 'Barangay', 'Count': 'Number of Incidents'},
                        color='Count',
                        color_continuous_scale='Teal'
                    )
                    fig_barangay.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                    st.plotly_chart(fig_barangay, use_container_width=True)
            else:
                st.warning("Barangay column not found in dataset.")

            st.markdown("---")
                        
            # 3. TOP 10 TIME PERIODS
            st.subheader("🕐 Top 10 Time Periods When Crimes Occur Most")

            if 'Time_Occ' in df.columns:
                filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
                
                with filter_col1:
                    if 'Barangay' in df.columns:
                        barangay_list = ['All'] + sorted(df['Barangay'].dropna().unique().tolist())
                        selected_barangay_time = st.selectbox(
                            "Filter by Barangay:",
                            options=barangay_list,
                            key="time_barangay_filter"
                        )
                    else:
                        selected_barangay_time = 'All'
                
                with filter_col2:
                    if 'Crime_Type' in df.columns:
                        crime_type_list = ['All'] + sorted(df['Crime_Type'].dropna().unique().tolist())
                        selected_crime_time = st.selectbox(
                            "Filter by Crime Type:",
                            options=crime_type_list,
                            key="time_crime_filter"
                        )
                    else:
                        selected_crime_time = 'All'
                
                filtered_df = df.copy()
                title_parts = []
                
                if selected_barangay_time != 'All':
                    filtered_df = filtered_df[filtered_df['Barangay'] == selected_barangay_time]
                    title_parts.append(selected_barangay_time)
                
                if selected_crime_time != 'All':
                    filtered_df = filtered_df[filtered_df['Crime_Type'] == selected_crime_time]
                    title_parts.append(selected_crime_time)
                
                title_suffix = f" ({', '.join(title_parts)})" if title_parts else " (All)"
                
                df_temp = filtered_df.copy()
                df_temp['Time_Occ'] = pd.to_datetime(df_temp['Time_Occ'], format='%H:%M:%S', errors='coerce').dt.time
                df_temp['Hour'] = pd.to_datetime(df_temp['Time_Occ'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour
                
                time_counts = df_temp.groupby('Hour').size().reset_index(name='Count')
                time_counts = time_counts.sort_values('Count', ascending=False).head(10)
                time_counts['Time'] = time_counts['Hour'].apply(lambda x: f"{int(x):02d}:00")
                time_counts = time_counts[['Time', 'Count']].reset_index(drop=True)
                time_counts.insert(0, 'No.', range(1, len(time_counts) + 1))
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write(f"**Time Period Crime Statistics{title_suffix}**")
                    st.dataframe(
                        time_counts.style.set_properties(**{'text-align': 'left'}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    fig_time = px.bar(
                        time_counts, 
                        x='Count', 
                        y='Time',
                        orientation='h',
                        title=f"Top 10 Time Periods by Crime Count{title_suffix}",
                        labels={'Time': 'Time Period', 'Count': 'Number of Incidents'},
                        color='Count',
                        color_continuous_scale='Viridis'
                    )
                    fig_time.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                    st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.warning("Time_Occ column not found in dataset. Showing 'Time of Day' analysis requires a 'Time_Occ' column in HH:MM:SS format.")

            st.markdown("---")
            
            # 4. DISTRIBUTION OVERVIEW (CRIME TYPE & BARANGAY PIE CHARTS)
# 4. DISTRIBUTION OVERVIEW (CRIME TYPE & BARANGAY PIE CHARTS)
            st.subheader("📊 Distribution Overview")

            # Crime Type Pie Chart (always shows all data)
            if 'Crime_Type' in df.columns:
                crime_pie_data = df.groupby('Crime_Type').size().reset_index(name='Count')
                crime_pie_data = crime_pie_data.sort_values('Count', ascending=False).head(10)
                
                fig_crime_pie = px.pie(
                    crime_pie_data,
                    values='Count',
                    names='Crime_Type',
                    title="Crime Type Distribution (Top 10)",
                    hole=0.3
                )
                fig_crime_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_crime_pie.update_layout(height=700)
                st.plotly_chart(fig_crime_pie, use_container_width=True)

            # Barangay Pie Chart with filter
            if 'Barangay' in df.columns:
                st.markdown("---")  # Optional: add separator between the two charts
                
                # Create filter for crime type (now appears between charts)
                pie_filter_col1, pie_filter_col2 = st.columns([1, 3])
                
                with pie_filter_col1:
                    if 'Crime_Type' in df.columns:
                        crime_type_list_pie = ['All'] + sorted(df['Crime_Type'].dropna().unique().tolist())
                        selected_crime_pie = st.selectbox(
                            "Filter Barangay Chart by Crime Type:",
                            options=crime_type_list_pie,
                            key="pie_crime_filter"
                        )
                    else:
                        selected_crime_pie = 'All'
                
                # Filter dataframe based on selection
                if selected_crime_pie == 'All':
                    filtered_df_pie = df
                    pie_title_suffix = " (All Crime Types)"
                else:
                    filtered_df_pie = df[df['Crime_Type'] == selected_crime_pie]
                    pie_title_suffix = f" ({selected_crime_pie})"
                
                barangay_pie_data = filtered_df_pie.groupby('Barangay').size().reset_index(name='Count')
                barangay_pie_data = barangay_pie_data.sort_values('Count', ascending=False).head(10)
                
                fig_barangay_pie = px.pie(
                    barangay_pie_data,
                    values='Count',
                    names='Barangay',
                    title=f"Barangay Distribution (Top 10){pie_title_suffix}",
                    hole=0.3
                )
                fig_barangay_pie.update_traces(textposition='auto', textinfo='percent+label')
                fig_barangay_pie.update_layout(height=700)
                st.plotly_chart(fig_barangay_pie, use_container_width=True)

            st.markdown("---")
            
            # 5. ALL BARANGAYS OVERVIEW
            st.subheader("🏘️ Overall Crime Incidents by Barangay")
            st.markdown("This bar chart displays the **total number of reported incidents** for **all** available barangays, sorted alphabetically. Scroll to view all.")
            
            if 'Barangay' not in df.columns:
                st.error("Missing 'Barangay' column in the dataset.")
            else:
                barangay_counts_all = df.dropna(subset=['Barangay']).groupby('Barangay').size().reset_index(name='Total Incidents')
                barangay_counts_all = barangay_counts_all.sort_values(by='Barangay', ascending=True).reset_index(drop=True)
                
                if barangay_counts_all.empty:
                    st.warning("No valid 'Barangay' data found.")
                else:
                    num_barangays = len(barangay_counts_all)
                    st.info(f"✅ Found **{num_barangays}** unique barangays.")
                    
                    # Dynamic width based on number of barangays
                    bar_width = 50
                    chart_width = max(1000, num_barangays * bar_width)
                    chart_height = 500
                    
                    fig_barangay_all = px.bar(
                        barangay_counts_all,
                        x='Barangay',
                        y='Total Incidents',
                        color='Total Incidents',
                        title="Total Crime Counts per Barangay (Alphabetical)",
                        labels={'Barangay': 'Barangay Name', 'Total Incidents': 'Total Crime Incidents'},
                        height=chart_height,
                        width=chart_width,
                        color_continuous_scale=px.colors.sequential.Plotly3 # New color scale
                    )
                    
                    fig_barangay_all.update_layout(
                        xaxis={
                            'tickangle': -45,
                            'title': 'Barangay Name'
                        },
                        yaxis={
                            'title': 'Total Incidents',
                            'tickformat': ',d'
                        },
                        showlegend=False,
                        margin=dict(l=80, r=40, t=80, b=150),
                        yaxis_gridcolor='lightgray'
                    )
                    st.plotly_chart(fig_barangay_all, use_container_width=False)
            
            st.markdown("---")
            
        # --- Download Section ---
        elif st.session_state.current_view == 'download':
            st.header("💾 Download Output")
            st.info("The clustered data includes the original incident data along with the calculated **LAT, LON, seconds, and cluster** columns.")
            
            st.write("Preview of clustered data (first 100 rows):")
            st.dataframe(df.head(100))
            
            # Prepare data for download
            csv = df.to_csv(index=False).encode('utf-8')
            
            col_save, col_download = st.columns([1, 2])
            
            with col_save:
                if st.button("💾 Save to Output History", use_container_width=True):
                    result = save_clustered_csv(df)
                    if result['status'] == 'success':
                        st.success(f"✅ File saved: {result['path']}")
                    else:
                        st.error(f"❌ Error: {result['message']}")
            
            with col_download:
                st.download_button(
                    label="⬇️ Download Clustered CSV",
                    data=csv,
                    file_name=f"pnp_clustered_crime_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    # Check authentication state and show appropriate page
    if st.session_state.logged_in:
        # User is logged in - show main application
        main_application()
    elif st.session_state.show_login:
        # User clicked "Go to Login" - show login page
        login_page()

    elif st.session_state.system_info:
        # Show system info page
        system_info()

    else:
        # Default - show landing page
        landing_page()

if __name__ == "__main__":
    main()