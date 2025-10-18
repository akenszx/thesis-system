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

# Page config
st.set_page_config(page_title="PNP Crime Mapping System", layout="wide", initial_sidebar_state="collapsed")

# ============================================================================
# LOGIN SYSTEM INTEGRATION
# ============================================================================

    # Custom CSS for login page with NEW PNP theme (Blue & Gold)
def inject_login_css():
    st.markdown("""
    <style>
    /* PNP Brand Colors */
    :root {
        --pnp-blue: #0033A0; /* Royal Blue */
        --pnp-gold: #FFC72C; /* Gold/Yellow */
        --pnp-white: #FFFFFF;
        --pnp-light-gray: #f8f9fa;
        --pnp-dark-text: #212529;
    }
    
    /* MINIMALIST FONT STYLE (Affects all text, but main-title is specific) */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
        background-image: linear-gradient(rgba(0, 51, 160, 0.85), rgba(255, 199, 44, 0.65)), 
                        url('https://media.philstar.com/photos/2019/10/20/pnp-12019-10-1623-33-56_2019-10-20_17-30-33.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        filter: blur(0px);
    }

    .main {
        background: transparent;
    }

    /* All text white by default on landing page */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: var(--pnp-white) !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7); 
    }

    /* Login form styling - White/Blue/Gold */
    div[data-testid="stForm"] {
        background: var(--pnp-white);
        padding: 3rem;
        border-radius: 15px;
        box-shadow: 0 12px 40px rgba(0, 51, 160, 0.4);
        border: 4px solid var(--pnp-blue);
    }

    .form-title {
        color: var(--pnp-blue) !important;
        text-align: center;
        font-weight: 900;
        margin-bottom: 0.5rem;
        text-shadow: none !important;
        font-size: 2.8rem;
    }

    .subtitle {
        text-align: center;
        color: var(--pnp-dark-text);
        margin-bottom: 2rem;
        font-size: 1.1rem;
        text-shadow: none !important;
    }

    /* Input labels - Blue */
    .stTextInput > label {
        color: var(--pnp-blue) !important;
        font-weight: 700;
        text-shadow: none !important;
        font-size: 1.1rem;
    }

    /* Input fields styling */
    .stTextInput > div > div > input {
        border: 2px solid var(--pnp-blue);
        border-radius: 8px;
        padding: 12px;
        font-size: 1rem;
        color: var(--pnp-dark-text);
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--pnp-gold);
        box-shadow: 0 0 0 0.2rem rgba(0, 51, 160, 0.2);
    }

    /* Primary Buttons (Login) - Blue/Gold theme */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--pnp-blue) 0%, #0056b3 100%);
        color: white;
        border: 2px solid var(--pnp-gold);
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(0, 51, 160, 0.4);
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0056b3 0%, var(--pnp-blue) 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 51, 160, 0.6);
        border-color: var(--pnp-white);
    }
    
    /* Secondary Buttons (Back to Home) - White/Blue theme */
    .stButton > button:not([kind="primary"]) {
        background: var(--pnp-light-gray);
        color: var(--pnp-blue);
        border: 1px solid var(--pnp-blue);
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        margin-top: 10px;
    }

    .stButton > button:not([kind="primary"]):hover {
        background: var(--pnp-white);
        color: var(--pnp-blue);
        border-color: var(--pnp-gold);
    }

    /* Error box - Blue/Gold theme */
    .error-box {
        background-color: #fff3cd; /* Light Yellow */
        border: 2px solid var(--pnp-gold);
        color: var(--pnp-dark-text);
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
        font-weight: 600;
    }

    /* Footer text */
    .footer {
        text-align: center;
        color: var(--pnp-white);
        margin-top: 2rem;
        font-size: 0.9rem;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
        font-weight: 600;
    }

    /* Info boxes for landing page - Clean White Look */
    .info-box {
        background: var(--pnp-white);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        margin-bottom: 2rem;
        border-top: 5px solid var(--pnp-gold);
        height: 100%; /* Ensure columns are same height */
    }

    .info-box h3 {
        color: var(--pnp-blue) !important;
        text-shadow: none !important;
        margin-bottom: 1rem;
        font-weight: 800;
        font-size: 1.6rem;
        border-bottom: 2px solid var(--pnp-light-gray);
        padding-bottom: 10px;
    }

    .info-box ul {
        color: var(--pnp-dark-text);
        line-height: 1.8;
        font-size: 1rem;
        padding-left: 20px;
    }
    
    .info-box li {
        margin-bottom: 0.6rem;
        text-shadow: none !important;
    }

    .info-box strong {
        color: var(--pnp-blue) !important;
        text-shadow: none !important;
    }

    /* Main title styling - MODIFIED FOR WHITE COLOR ONLY (no gradient) */
    .main-title {
        text-align: center;
        color: var(--pnp-white) !important; /* Set to pure white */
        font-size: 4rem;
        font-weight: 900;
        text-shadow: 3px 3px 10px rgba(0, 0, 0, 0.95);
        margin-top: 3rem;
        margin-bottom: 1rem;
        /* Remove text-gradient for a cleaner, minimalist white look */
        -webkit-background-clip: unset; 
        -webkit-text-fill-color: var(--pnp-white) !important;
        background-clip: unset;
        background: unset;
    }
    
    .main-subtitle {
        text-align: center; 
        color: var(--pnp-light-gray); 
        font-size: 1.5rem; 
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.9); 
        font-weight: 600; 
        margin-bottom: 3rem;
    }

    /* Footer text styling */
    .footer-text {
        text-align: center;
        color: white;
        font-size: 1rem;
        margin-top: 4rem;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.9);
        padding: 1.5rem;
        background: rgba(0, 51, 160, 0.4);
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid var(--pnp-gold);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'show_login' not in st.session_state:
    st.session_state.show_login = False

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

# Authentication function
def authenticate(username, password):
    valid_credentials = {
        "admin": "admin",
        "user": "user"
    }
    return username in valid_credentials and valid_credentials[username] == password

# Landing/Welcome page (Revised for better design and focus)
def landing_page():
    inject_login_css()
    
    st.markdown('<h1 class="main-title">🛡️ PNP Crime Mapping System</h1>', unsafe_allow_html=True)
    
    st.markdown('<p class="main-subtitle">A Data-Driven Tool for Safer Communities in Echague</p>', unsafe_allow_html=True)
    
    # Center the button - Use a primary button for high visibility
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    with col2:
        if st.button("🚀 ACCESS THE LAW ENFORCEMENT PORTAL", use_container_width=True, type="primary"):
            st.session_state.show_login = True
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
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
        <div class="info-box">
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
    
    st.markdown("""
    <div class="footer-text">
        <span style="color: var(--pnp-gold); font-weight: 700;">"Service, Honor, Justice"</span><br>
        © 2025 - Philippine National Police Crime Mapping System for Echague, Isabela
    </div>
    """, unsafe_allow_html=True)

# Login page (Revised for new theme)
def login_page():
    inject_login_css()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<h1 class="form-title">🔒 Law Enforcement Portal</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Secure Access to Crime Analytics System</p>', unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            
            # The submit button is set as 'primary' in the CSS above for the blue/gold look
            submit = st.form_submit_button("🔓 Login to System", use_container_width=True, type="primary")
            
            if submit:
                if username and password:
                    if authenticate(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.markdown('<div class="error-box">❌ Invalid credentials. Access denied.</div>', 
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-box">⚠️ Please enter both username and password.</div>', 
                            unsafe_allow_html=True)
        
        # Back to home button (Secondary look from CSS)
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.show_login = False
            st.rerun()
        
        st.markdown('<div class="footer">🛡️ Authorized Personnel Only<br>All access attempts are logged and monitored for security purposes</div>', 
                unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION FUNCTIONS (No changes needed, kept for completeness)
# ============================================================================

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

# ============================================================================
# MAIN APPLICATION INTERFACE (New BLUE & GOLD THEME)
# ============================================================================

def main_application():
    """Main crime mapping application with BLUE & GOLD theme"""
    
    # Custom CSS for main application
    st.markdown("""
    <style>
    /* PNP Brand Colors */
    :root {
        --pnp-blue: #0033A0; /* Royal Blue */
        --pnp-gold: #FFC72C; /* Gold/Yellow */
        --pnp-white: #FFFFFF;
        --pnp-light-gray: #f8f9fa;
        --pnp-dark-text: #212529;
    }

    /* Main app background */
    .stApp {
        background-color: var(--pnp-white);
        background-image: none; /* Remove background image from main app */
    }
    
    /* Headers styling - Blue/Gold */
    h1, h2, h3 {
        color: var(--pnp-blue) !important;
        font-weight: 800;
        border-bottom: 2px solid var(--pnp-gold);
        padding-bottom: 5px;
        margin-top: 1.5rem;
    }
    
    /* Buttons - Blue theme */
    .stButton > button {
        background: linear-gradient(135deg, var(--pnp-blue) 0%, #0056b3 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 51, 160, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0056b3 0%, var(--pnp-blue) 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 51, 160, 0.4);
    }
    
    /* Metrics styling - Blue accents */
    div[data-testid="stMetricValue"] {
        color: var(--pnp-blue) !important;
        font-weight: 800;
        font-size: 1.8rem;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        border-left: 5px solid var(--pnp-blue);
        border-radius: 8px;
        padding: 1rem;
        background-color: var(--pnp-light-gray);
    }
    
    /* Menu styling (streamlit_option_menu) */
    .css-1544g2n { /* Target for the menu container */
        background-color: var(--pnp-white) !important;
        border: 1px solid var(--pnp-light-gray);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Dataframe headers */
    .stDataFrame th {
        background-color: var(--pnp-blue) !important;
        color: white !important;
    }
    
    /* Risk Alert Box - Blue/Gold/Red Theme */
    .risk-alert-box {
        border: 4px solid #cc0000; /* Darker Red for high risk */
        padding: 20px;
        background-color: #fcebeb; /* Very Light Red */
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(204, 0, 0, 0.2);
    }
    
    .risk-alert-box h2 {
        color: #cc0000 !important;
        text-align: center;
        border-bottom: 2px solid #cc0000;
        padding-bottom: 10px;
    }
    
    .risk-alert-box p, .risk-alert-box ul {
        color: var(--pnp-dark-text);
        text-shadow: none !important;
    }
    
    .risk-alert-box li {
        color: #cc0000;
        font-weight: 600;
    }
    
    /* Main application title */
    .main-app-title {
        color: var(--pnp-blue);
        font-weight: 900;
        font-size: 2.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with logout button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown('<div class="main-app-title">🗺️ PNP Crime Mapping System</div>', unsafe_allow_html=True)
        st.caption(f"👤 Logged in as: **{st.session_state.username}** | Echague, Isabela")
    with col2:
        # Use secondary style for logout button
        st.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.show_login = False
            st.session_state.df = None
            st.session_state.df_clustered = None
            st.session_state.clustering_complete = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📂 Data Management")
    file = st.file_uploader("Upload your CSV file", type=["csv"])
    st.info("📌 Required columns: **Date_Occ, Crime_Type, Barangay**, and optionally **Time_Occ** for detailed reports.")

    if file:
        df = pd.read_csv(file)
        
        original_size = len(df)
        required_cols = ['Date_Occ', 'Crime_Type', 'Barangay']
        
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Missing required columns: {', '.join(required_cols)}")
            st.stop()
        
        df['timestamp'] = pd.to_datetime(df['Date_Occ'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df['seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
        
        cleaned_size = len(df)
        removed_rows = original_size - cleaned_size
        
        st.success("✅ Data preprocessed successfully!")
        st.info(f"📊 **Original:** {original_size} rows | **Cleaned:** {cleaned_size} rows ({removed_rows} removed due to missing/invalid dates)")
        
        needs_geocoding = 'LAT' not in df.columns or 'LON' not in df.columns
        
        # Storing the DataFrame in session state after initial cleaning
        st.session_state.df = df

        col_left_action, col_right_action = st.columns(2)
        
        with col_left_action:
            if needs_geocoding:
                st.warning("⚠️ Latitude/Longitude columns not found. Geocoding required.")
                
                if st.button("🌍 Start Geocoding & Clustering", use_container_width=True):
                    # --- GEOCODING ---
                    with st.spinner("Step 1/2: Geocoding barangays..."):
                        df_geocoded, removed, total_brgy, failed = batch_geocode_dataset(df)
                        
                        if df_geocoded.empty:
                            st.error("❌ Geocoding failed for all barangays. Cannot proceed.")
                            st.stop()
                        
                        st.success(f"✅ Geocoding complete: {total_brgy - failed}/{total_brgy} successful. ({removed} rows dropped)")
                        st.session_state.geocoding_complete = True
                    
                    # --- CLUSTERING ---
                    with st.spinner("Step 2/2: Running CD-DBSCAN clustering..."):
                        df_clustered, clusters_found, avg_query = run_cd_dbscan(df_geocoded)
                        
                        st.session_state.df_clustered = df_clustered
                        st.session_state.clustering_complete = True
                        
                        noise_points = (df_clustered['cluster'] == -1).sum()
                        clustered_points = len(df_clustered) - noise_points
                        
                        st.success("✅ Clustering complete!")
                        st.toast("Clustering finished!")
                        st.rerun() # Rerun to update the complete application state
            else:
                st.info("✅ Dataset already contains LAT/LON columns. Ready for clustering.")
                
                if st.button("🚀 Run CD-DBSCAN Clustering", use_container_width=True):
                    # --- CLUSTERING ---
                    with st.spinner("Running clustering..."):
                        df_clustered, clusters_found, avg_query = run_cd_dbscan(df)
                        
                        st.session_state.df_clustered = df_clustered
                        st.session_state.clustering_complete = True
                        
                        noise_points = (df_clustered['cluster'] == -1).sum()
                        clustered_points = len(df_clustered) - noise_points
                        
                        st.success("✅ Clustering complete!")
                        st.toast("Clustering finished!")
                        st.rerun() # Rerun to update the complete application state

        if st.session_state.clustering_complete:
            with col_right_action:
                df_clustered = st.session_state.df_clustered
                clusters_found = df_clustered['cluster'].nunique() - (1 if -1 in df_clustered['cluster'].unique() else 0)
                noise_points = (df_clustered['cluster'] == -1).sum()
                clustered_points = len(df_clustered) - noise_points
                
                # Simple placeholder for avg_query time if not recalculated on rerun
                # Note: To get the actual avg_query, you'd need to store it in session_state, 
                # but for this review, we'll focus on the core metrics.
                avg_query = 0 
                
                st.markdown("### Clustering Results Overview")
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Clusters Found", clusters_found)
                col_m2.metric("Clustered Points", f"{clustered_points:,}")
                
                col_m3, col_m4 = st.columns(2)
                col_m3.metric("Noise Points", noise_points)
                col_m4.metric("Total Data Points", len(df_clustered))
                
                st.session_state.df = df_clustered # Ensure the main df is the clustered one
                
    
    # Menu (only show if clustering is complete)
    if st.session_state.clustering_complete and st.session_state.df_clustered is not None:
        st.markdown("---")
        
        # Ensure df is the clustered version for menu use
        df = st.session_state.df_clustered

        menu = option_menu(
            "Navigation Menu",
            ["Visualizations", "Forecasting", "Reports", "Download Output"],
            icons=['map', 'graph-up', 'file-text', 'download'],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "5px", "background-color": "#ffffff", "border-bottom": "3px solid var(--pnp-gold)"},
                "icon": {"color": "var(--pnp-gold)", "font-size": "20px"}, 
                "nav-link": {
                    "font-size": "18px",
                    "text-align": "center",
                    "margin":"5px",
                    "background-color": "var(--pnp-light-gray)",
                    "border": "1px solid var(--pnp-blue)",
                    "border-radius": "8px",
                    "color": "var(--pnp-blue)",
                    "font-weight": "600",
                },
                "nav-link-selected": {
                    "background-color": "var(--pnp-blue)",
                    "color": "var(--pnp-white)",
                    "font-weight": "700",
                    "border": "1px solid var(--pnp-gold)"
                },
            }
        )
        
        # --- Visualizations Section ---
        if menu == "Visualizations":
            st.header("🗺️ Geospatial Visualizations")
            
            # 1. Cluster Visualization (Pydeck/Plotly for modern look)
            
            # 1. Prepare data for Cluster Visualizations (Valid LAT/LON and assigned cluster != -1)
            df_clustered = df[(df['LAT'] != 0) & (df['LON'] != 0) & (df['cluster'] != -1)].copy()

            if df_clustered.empty:
                st.warning("No valid, clustered data available for visualization.")
            else:
                st.subheader("Crime Hotspot Cluster Map")

                # Use Plotly Scatter Mapbox for better cluster distinction
                st.caption("Each color represents a distinct Spatio-Temporal Crime Cluster (Hotspot). Noise points are excluded.")

                # Use a smaller sample for performance in Plotly
                df_sample = df_clustered.sample(min(20000, len(df_clustered)))
                
                # Check for unique clusters in the sample
                unique_clusters = df_sample['cluster'].nunique()
                color_scale_name = 'Plasma' if unique_clusters > 1 else 'Blues'

                fig = px.scatter_mapbox(
                    df_sample,
                    lat='LAT',
                    lon='LON',
                    color='cluster',
                    hover_data=['Barangay', 'Crime_Type', 'cluster'],
                    center=dict(lat=df_clustered['LAT'].mean(), lon=df_clustered['LON'].mean()),
                    zoom=10,
                    mapbox_style="open-street-map",
                    title="CD-DBSCAN Crime Clusters",
                    opacity=0.8,
                    color_continuous_scale=color_scale_name
                )
                
                # Update layout for cleaner presentation
                fig.update_layout(
                    height=600, 
                    margin={"r":0,"t":50,"l":0,"b":0},
                    coloraxis_showscale=False # Hide color scale for cluster ID which isn't meaningful
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

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

            # 3. RISK ASSESSMENT ANALYSIS (Forecast and Pydeck)

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
                    if st.button(f"Generate Municipality-Wide Forecast ({forecast_days} Days) 📈", type="primary"):
                        with st.spinner(f"Running disaggregated ARIMA forecast for {len(selected_barangays)} barangays over {forecast_days} days..."):

                            # --- SIMULATED FORECASTING (Placeholder) ---
                            # In a real app, you'd use the run_disaggregated_arima function
                            if 'run_disaggregated_arima' in globals():
                                forecast_data = run_disaggregated_arima(df, selected_barangays, forecast_days)
                            else:
                                # Placeholder/simulated data generation 
                                dates = pd.to_datetime(pd.date_range(start=pd.Timestamp.now(), periods=forecast_days, freq='D'))
                                np.random.seed(42)
                                all_data = []
                                for b in selected_barangays:
                                    # Higher risk for two barangays
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
                        st.rerun() # Rerun to display the forecast results clearly

                    # --- DISPLAY AND RISK VISUALIZATION ---
                    if st.session_state.forecast_data is not None and not st.session_state.forecast_data.empty:
                        
                        df_f = st.session_state.forecast_data.copy()
                        df_f['Incident_Date'] = pd.to_datetime(df_f['Incident_Date']) 
                        
                        # Data Aggregation for Risk Scoring
                        df_risk = df_f.groupby('Barangay')['Predicted_Count'].sum().reset_index()
                        
                        # Get mean coordinates for each barangay (for map)
                        df_coords = df[['Barangay', 'LAT', 'LON']].groupby('Barangay').mean().reset_index()
                        df_risk = pd.merge(df_risk, df_coords, on='Barangay', how='inner')

                        # --- Risk Alert Logic ---
                        st.markdown("<h3 style='color: #cc0000 !important; border-bottom-color: #cc0000 !important;'>🚨 High-Risk Assessment Alert 🚨</h3>", unsafe_allow_html=True)
                        N = 4
                        df_high_risk = df_risk.sort_values(by='Predicted_Count', ascending=False).head(N)
                        
                        # Threshold for High Risk: Any barangay in the top N with a count > 0
                        high_risk_barangays = df_high_risk[df_high_risk['Predicted_Count'] > 0]['Barangay'].tolist()

                        if high_risk_barangays:
                            alert_html = f"""
                            <div class="risk-alert-box">
                                <h2>High Accident Risk Detected ({forecast_days} Days)</h2>
                                <p style="text-align: center;">Immediate action required in the following **Top {N}** barangays based on predicted incidents:</p>
                                <ul style="list-style-type: none; text-align: center; padding: 0;">
                                    {''.join([f"<li>🔴 **{b}** (Predicted Incidents: {df_high_risk[df_high_risk['Barangay'] == b]['Predicted_Count'].iloc[0]})</li>" for b in high_risk_barangays])}
                                </ul>
                                <br>
                                <p style="font-weight: bold; text-align: center;">Recommended Actions:</p>
                                <ul style="list-style-type: disc; margin: 0 auto; width: fit-content;">
                                    <li>Increase patrol frequency and visibility.</li>
                                    <li>Conduct focused community engagement in these areas.</li>
                                    <li>Review existing security measures and infrastructure.</li>
                                </ul>
                            </div>
                            """
                            st.markdown(alert_html, unsafe_allow_html=True)
                        else:
                            st.info(f"The forecast did not identify any barangays with a high enough predicted incident count to trigger a high-risk alert for the next {forecast_days} days.")
                        
                        st.markdown("---")

                        # --- Risk Level Coloring ---
                        # Logic to color ALL barangays, not just high risk, for Pydeck
                        df_risk['Color_R'] = 0
                        df_risk['Color_G'] = 0
                        df_risk['Color_B'] = 0
                        df_risk['Risk_Level'] = 'Low Risk' 

                        # Low Risk (Green/Blue): Default or very low prediction
                        low_mask = df_risk['Predicted_Count'] <= 1
                        df_risk.loc[low_mask, ['Color_R', 'Color_G', 'Color_B']] = [0, 150, 0] # Darker Green
                        df_risk.loc[low_mask, 'Risk_Level'] = 'Low Risk'
                        
                        # Moderate Risk (Orange)
                        moderate_mask = (~df_risk['Barangay'].isin(high_risk_barangays)) & (df_risk['Predicted_Count'] > 1)
                        df_risk.loc[moderate_mask, ['Color_R', 'Color_G', 'Color_B']] = [255, 140, 0] # Dark Orange
                        df_risk.loc[moderate_mask, 'Risk_Level'] = 'Moderate Risk'

                        # High Risk (Red)
                        high_mask = df_risk['Barangay'].isin(high_risk_barangays)
                        df_risk.loc[high_mask, ['Color_R', 'Color_G', 'Color_B']] = [255, 0, 0]
                        df_risk.loc[high_mask, 'Risk_Level'] = 'High Risk'

                        # --- Pydeck Visualization (All Barangays) ---
                        st.subheader(f"3D Predictive Risk Map ({forecast_days} Days)")

                        if not df_risk.empty and df_risk['Predicted_Count'].sum() > 0:

                            max_count = df_risk['Predicted_Count'].max()
                            # Elevation scale to ensure height is visible but not excessive
                            ELEVATION_SCALE = 5000 / (max_count if max_count > 0 else 1) 

                            view_state = pdk.ViewState(
                                latitude=df_risk['LAT'].mean(),
                                longitude=df_risk['LON'].mean(),
                                zoom=11.5,
                                pitch=50,
                                bearing=-20 
                            )
                            
                            # Column Layer (Cylinders) - ALL filtered barangays
                            column_layer = pdk.Layer(
                                'ColumnLayer',
                                data=df_risk, 
                                get_position='[LON, LAT]',
                                get_elevation='Predicted_Count * ' + str(ELEVATION_SCALE), 
                                elevation_scale=1, 
                                radius=300, 
                                radius_scale=1,
                                get_fill_color='[Color_R, Color_G, Color_B, 200]', 
                                pickable=True,
                                extruded=True,
                                auto_highlight=True,
                            )
                            
                            # Text Layer (Barangay Labels) - Only for High-Risk to reduce clutter
                            df_text = df_risk[df_risk['Risk_Level'] == 'High Risk'].copy()
                            text_layer = pdk.Layer(
                                'TextLayer',
                                data=df_text,
                                get_position='[LON, LAT]',
                                get_text='Barangay',
                                get_size=12,
                                get_color='[255, 255, 255, 255]',
                                get_elevation='Predicted_Count * ' + str(ELEVATION_SCALE), 
                                get_angle=0,
                                get_text_anchor='"middle"',
                                get_alignment_baseline='"center"',
                                get_pixel_offset=[0, 30],
                                pickable=False,
                                billboard=True,
                                background=True, 
                                get_background_color='[0, 0, 0, 150]',
                            )

                            tooltip = {
                                "html": "<b>Barangay:</b> {Barangay}<br/><b>Predicted Incidents:</b> {Predicted_Count:.0f}<br/><b>Risk Level:</b> {Risk_Level}",
                                "style": {"backgroundColor": "var(--pnp-blue)", "color": "white"}
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
                            <p style='font-weight: bold; color: var(--pnp-dark-text);'>Risk Legend for the Next {forecast_days} Days:</p>
                            <div style="display: flex; gap: 20px; font-weight: 600;">
                                <span style="color: red;">&#9679; High Risk (Top {N})</span>
                                <span style="color: orange;">&#9679; Moderate Risk</span>
                                <span style="color: green;">&#9679; Low Risk</span>
                            </div>
                            <p style='color: var(--pnp-dark-text); font-style: italic; margin-top: 10px;'>The **height** of the columns represents the **sum of predicted incidents** for each barangay. Taller columns indicate higher predicted volume/risk.</p>
                            """, unsafe_allow_html=True)
                            
                        else:
                            st.info("No predictive data available to generate the 3D Risk Map.")
            
        # --- Forecasting Section ---
        elif menu == "Forecasting":
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
                        
                        if chart_type == 'Line':
                            fig2 = px.line(df_plot, x='Date', y='Count', color='Type',
                                        title="Monthly Crime Incident Forecast (ARIMA)", markers=True,
                                        color_discrete_map={'Observed': 'var(--pnp-blue)', 'Forecast': 'var(--pnp-gold)'})
                        else:
                            fig2 = px.bar(df_plot, x='Date', y='Count', color='Type',
                                        title="Monthly Crime Incident Forecast (ARIMA)", barmode='group',
                                        color_discrete_map={'Observed': 'var(--pnp-blue)', 'Forecast': 'var(--pnp-gold)'})
                        
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
        elif menu == "Reports":
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
        elif menu == "Download Output":
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
    else:
        # Default - show landing page
        landing_page()

if __name__ == "__main__":
    main()