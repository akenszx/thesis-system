
import streamlit as st
import pandas as pd
import numpy as np #pandas
import matplotlib.pyplot as plt
import seaborn as sns
import time
from statsmodels.tsa.arima.model import ARIMA
import folium
from folium.plugins import HeatMap
from streamlit.components.v1 import html
from rtree import index  # For CD-DBSCAN
import os
from datetime import datetime
import plotly.express as px
import streamlit_option_menu
from streamlit_option_menu import option_menu

st.set_page_config(page_title="DBSCAN Simulator", layout="wide")

st.title("📍 Spatiotemporal DBSCAN Simulator")

menu_options = [
    "Upload & Preprocess",
    "Run Clustering",
    "Visualizations",
    "Evaluation",
    "Forecasting",
    "Comparison",
    "Download Output"
]
if 'menu' not in st.session_state:
    st.session_state.menu = "Upload & Preprocess"

with st.sidebar:
    menu = option_menu(
        "Menu",
        menu_options,
        default_index=menu_options.index(st.session_state.menu),
        orientation="vertical",
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "icon": {"color": "orange", "font-size": "20px"}, 
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin":"0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#521888"},
        }
    )
    st.session_state.menu = menu
menu = st.session_state.menu    

rec_eps_spatial = 0.05   
rec_eps_temporal = 3600     
rec_min_samples = 5         

eps_spatial = 0.05 
eps_temporal = 3600
min_samples = 5

st.sidebar.markdown("---")
st.sidebar.markdown(f"Created by **Jake**  \n📧 dbscan@isu.edu.ph")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'labels' not in st.session_state: #else if
    st.session_state.labels = None
if 'spatial_index' not in st.session_state:
    st.session_state.spatial_index = None

def go_to_step_2():
    """Sets the menu state to 'Run Clustering'."""
    st.session_state.menu = "Run Clustering"

def go_to_step_3():
    """Sets the menu state to 'Visualizations'."""
    st.session_state.menu = "Visualizations"

def go_to_step_4():
    """Sets the menu state to 'Evaluation'."""
    st.session_state.menu = "Evaluation"

def go_to_step_5():
    """Sets the menu state to 'Forecasting'."""
    st.session_state.menu = "Forecasting"

def go_to_step_6():
    """Sets the menu state to 'Comparison'."""
    st.session_state.menu = "Comparison"

def go_to_step_7():
    """Sets the menu state to 'Download Output'."""
    st.session_state.menu = "Download Output"

def run_st_dbscan(df):
    coords = df[['LAT', 'LON']].values
    times = df['seconds'].values
    visited = set()
    labels = [-1] * len(df)
    cluster_id = 0

    query_times = []  # Store query times

    def get_neighbors(i):
        start_q = time.perf_counter()
        lat, lon, t = coords[i][0], coords[i][1], times[i]
        neighbors = []
        for j in range(len(df)):
            if i == j:#i
                continue
            if abs(times[j] - t) <= eps_temporal:
                dist = np.sqrt((lat - coords[j][0])**2 + (lon - coords[j][1])**2)
                if dist <= eps_spatial:
                    neighbors.append(j)
        end_q = time.perf_counter()
        query_times.append((end_q - start_q) * 1000)  # ms
        return neighbors

    for i in range(len(df)):
        if i in visited:
            continue
        neighbors = get_neighbors(i)
        if len(neighbors) < min_samples:
            visited.add(i)
            continue
        labels[i] = cluster_id #j
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

# Enhancement
def run_cd_dbscan(df):
    coords = df[['LAT', 'LON']].values
    times = df['seconds'].values
    visited = set()
    labels = [-1] * len(df)
    cluster_id = 0

    # Build spatial index
    spatial_index = index.Index()
    for i, row in df.iterrows():
        spatial_index.insert(i, (row['LON'], row['LAT'], row['LON'], row['LAT']))

    query_times = [] 

    def get_neighbors(i):

        start_q = time.perf_counter()

        # Ensures neighbors are close in both space and time.
        lat, lon, t = coords[i][0], coords[i][1], times[i]
        box = (lon - eps_spatial, lat - eps_spatial, lon + eps_spatial, lat + eps_spatial)
        spatial_candidates = list(spatial_index.intersection(box))
        neighbors = [j for j in spatial_candidates if abs(times[j] - t) <= eps_temporal]
        end_q = time.perf_counter()

        query_times.append((end_q - start_q) * 1000)  # ms
        return neighbors

    # Main Clustering Loop
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

    # Output 
    avg_query_time = np.mean(query_times) if query_times else 0
    df['cluster'] = labels
    return df, cluster_id, avg_query_time

def save_clustered_csv_to_folder(df):
    """Saves the clustered DataFrame to a local folder with a timestamp."""
    # Define the local folder for history
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

# Step 1: Upload & Preprocess
if menu == "Upload & Preprocess":
    st.subheader("📁 Upload & Preprocess Data")
    file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if file:

        df = pd.read_csv(file)
        original_size = len(df)

        df.dropna(subset=['DATE OCC', 'LAT', 'LON'], inplace=True)
        df.drop_duplicates(inplace=True)
        cleaned_size = len(df)
        removed_rows = original_size - cleaned_size

        if 'DATE OCC' in df.columns:
            df['timestamp'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
        else:
            st.error("Missing 'DATE OCC' column.")
            st.stop()

        df['seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

        st.success("✅ Data preprocessed successfully!")
        st.write(f"Original dataset size: {original_size} rows") #write not read
        st.write(f"After cleaning: {cleaned_size} rows ({removed_rows} removed)")

        col1, col2, col3 = st.columns(3)
        time_span_days = (df['timestamp'].max() - df['timestamp'].min()).days
        st.subheader(f"📊 Data Preview")
        col1.metric("**Total Records**", cleaned_size)
        col2.metric("**Time Span (days)**", time_span_days)

        if 'LAT' in df.columns and 'LON' in df.columns:
            lat_range = df['LAT'].max() - df['LAT'].min()
            lon_range = df['LON'].max() - df['LON'].min()
            # Rough km² approximation (1° ~ 111 km)
            area_km2 = (lat_range * 111) * (lon_range * 111)
            col3.metric("**Area Coverage (km²)**", int(area_km2))

        st.dataframe(df.head())

        st.session_state.df = df

        if st.button("Proceed to Run Clustering 🚀", on_click=go_to_step_2):
            pass 

# Step 2: Run Clustering
elif menu == "Run Clustering":
    st.subheader("🔄 Run Clustering")

    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please preprocess the data first.")
    else:
        df = st.session_state.df.copy()
        dataset_size = len(df)

        algorithm = st.radio("Select Algorithm", ["ST-DBSCAN (Standard)", "CD-DBSCAN (Enhanced)"])

        st.write(f"**Dataset Size**: {dataset_size} rows")
        st.write(f"**Algorithm**: {algorithm}")
        st.write(f"**Parameters**: ε-distance={eps_spatial}, ε-temporal={eps_temporal}, min_samples={min_samples}")

        if st.button("🚀 Start Clustering"):
            with st.spinner(f"Running {algorithm}..."):
                # Simulated progress bar
                progress_bar = st.progress(0)
                for percent_complete in range(0, 101, 20):
                    time.sleep(0.2)  # short delay for effect
                    progress_bar.progress(percent_complete)

                start_time = time.time()
                if algorithm == "ST-DBSCAN (Standard)":
                    df, clusters_found, avg_query_time = run_st_dbscan(df)
                else:
                    df, clusters_found, avg_query_time = run_cd_dbscan(df)
                end_time = time.time()

            execution_time = end_time - start_time
            noise_points = (df['cluster'] == -1).sum()
            clustered_points = dataset_size - noise_points
            clustered_percentage = (clustered_points / dataset_size) * 100
            noise_percentage = 100 - clustered_percentage
            average_cluster_size = clustered_points / clusters_found if clusters_found > 0 else 0
            processing_rate = dataset_size / execution_time if execution_time > 0 else 0

            st.session_state.df = df
            st.session_state.labels = df['cluster']
            st.session_state.avg_query_time = avg_query_time

            st.success("✅ Clustering complete!")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Clusters Found", clusters_found)
            col2.metric("Noise Points", noise_points)
            col3.metric("Execution Time (s)", round(execution_time, 2))
            col4.metric("Query Responsiveness (ms/query)", f"{avg_query_time:.4f}")

            st.subheader("📊 Performance Summary")
            col1, col2 = st.columns(2)
            col1.write(f"Clustered Data: {clustered_points} points ({clustered_percentage:.1f}%)")
            col1.write(f"Average Cluster Size: {average_cluster_size:.1f} points")
            col2.write(f"Noise Ratio: {noise_points} points ({noise_percentage:.1f}%)")
            col2.write(f"Processing Rate: {int(processing_rate)} points/second")

            st.dataframe(df.head())

            if 'labels' in st.session_state:

                if st.button("Proceed to Step 3: Visualizations 🗺️", on_click=go_to_step_3):
                    pass

# Step 3: Visualizations
elif menu == "Visualizations":
    st.subheader("🗺️ Cluster Visualizations")
    if st.session_state.df is None:
        st.warning("Please run clustering first.")
    else:
        df = st.session_state.df

        df_filtered = df[(df['LAT'] != 0) & (df['LON'] != 0) & (df['cluster'] != -1)].copy()

        if df_filtered.empty:
            st.warning("No distinct clusters found after removing noise (cluster -1).")
        else:
            lat_bins = 50
            lon_bins = 50

            df_filtered['lat_bin'] = pd.cut(df_filtered['LAT'], bins=lat_bins)
            df_filtered['lon_bin'] = pd.cut(df_filtered['LON'], bins=lon_bins)

            heatmap_data = df_filtered.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='count')

            heatmap_data['lat_mid'] = heatmap_data['lat_bin'].apply(lambda x: x.mid)
            heatmap_data['lon_mid'] = heatmap_data['lon_bin'].apply(lambda x: x.mid)

            fig = px.density_mapbox(
                df_filtered.sample(min(10000, len(df_filtered))),
                lat='LAT',
                lon='LON',
                z='cluster',
                radius=15,
                center=dict(lat=df_filtered['LAT'].mean(), lon=df_filtered['LON'].mean()),
                zoom=10,
                mapbox_style="open-street-map",
                title="🗺️ Cluster Density Choropleth Map",
                labels={'cluster': 'Cluster ID'},
                color_continuous_scale='Viridis',
                hover_data={'cluster': True, 'LAT': ':.4f', 'LON': ':.4f'}
            )
            
            fig.update_layout(
                height=600,
                margin={"r":0,"t":50,"l":0,"b":0}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"📍 Displaying {len(df_filtered):,} clustered points across {df_filtered['cluster'].nunique()} distinct clusters. Colors represent cluster density and distribution.")

        st.subheader("🌍 Interactive Folium Map")
        if st.session_state.df is None:
            st.warning("Please run clustering first.")
        else:
            df = st.session_state.df

            df_filtered = df[(df['LAT'] != 0) & (df['LON'] != 0) & (df['cluster'] != -1)].copy()

            if df_filtered.empty:
                st.warning("No distinct clusters found to plot after removing noise.")
            else:
                map_center = [df_filtered['LAT'].mean(), df_filtered['LON'].mean()]
                m = folium.Map(location=map_center, zoom_start=12)
        
                for _, row in df_filtered.iterrows():

                    marker_color = 'blue' 

                    folium.CircleMarker(
                        location=[row['LAT'], row['LON']],
                        radius=3,
                        color=marker_color,
                        fill=True,
                        fill_opacity=0.7,  
                        popup=f"Cluster: {row['cluster']}"
                    ).add_to(m)

            html(m._repr_html_(), height=600)

        st.subheader("🔥 Crime Density Heatmap")
        m2 = folium.Map(location=map_center, zoom_start=12)
        heat_data = [[row['LAT'], row['LON']] for _, row in df.iterrows()]
        HeatMap(heat_data).add_to(m2)
        html(m2._repr_html_(), height=600)

        if st.button("Proceed to Step 4: Evaluation 📈", on_click=go_to_step_4):
            pass

# Step 4: Evaluation
elif menu == "Evaluation":
    st.subheader("📈 Cluster Evaluation")

    start_time = time.perf_counter()

    if st.session_state.df is None or 'cluster' not in st.session_state.df.columns:
        st.warning("Please run clustering first.")
    else:
        df = st.session_state.df
        labels = df['cluster'].values

        total_points = len(labels)
        noise_points = np.sum(labels == -1)
        clustered_points = total_points - noise_points
        unique_clusters = len(np.unique(labels[labels != -1]))
        noise_ratio = noise_points / total_points

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000  

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.markdown("**Total Points**")
        col1.metric("", f"{total_points:,}")

        col2.markdown("**Clusters Found**")
        col2.metric("", unique_clusters)

        col3.markdown("**Clustered Points**")
        col3.metric("", f"{clustered_points:,}")

        col4.markdown("**Noise Ratio**")
        col4.metric("", f"{noise_ratio:.1%}")

        # New latency metric
        col5.markdown("**Latency (ms)**")
        col5.metric("", f"{latency_ms:.2f}")

        avg_query_time = st.session_state.get('avg_query_time', 0)
        col6.markdown("**Query Responsiveness (ms/query)**")
        col6.metric("", f"{avg_query_time:.4f}")

        st.write("### 🔍 Detailed Cluster Analysis")
        if unique_clusters > 0:
            cluster_metrics = []
            for cluster_id in np.unique(labels[labels != -1]):
                cluster_data = df[df['cluster'] == cluster_id]

                centroid_lat = cluster_data['LAT'].mean()
                centroid_lon = cluster_data['LON'].mean()

                distances = np.sqrt((cluster_data['LAT'] - centroid_lat)**2 +
                                    (cluster_data['LON'] - centroid_lon)**2)
                avg_distance = distances.mean()
                max_distance = distances.max()

                time_span = (cluster_data['seconds'].max() - cluster_data['seconds'].min()) / 3600

                spatial_area = np.pi * (max_distance * 111)**2  # rough km²
                density = len(cluster_data) / (spatial_area * max(time_span, 1)) if spatial_area > 0 else 0

                cluster_metrics.append({
                    'Cluster ID': cluster_id,
                    'Size': len(cluster_data),
                    'Centroid (Lat, Lon)': f"({centroid_lat:.4f}, {centroid_lon:.4f})",
                    'Avg Radius (km)': f"{avg_distance * 111:.2f}",
                    'Max Radius (km)': f"{max_distance * 111:.2f}",
                    'Time Span (hrs)': f"{time_span:.1f}",
                    'Density (pts/km²/hr)': f"{density:.2f}"
                })

            cluster_df = pd.DataFrame(cluster_metrics)
            st.dataframe(cluster_df, use_container_width=True)
        else:
            st.info("No clusters found. All points may be noise.")

        st.write("### 🏆 Cluster Quality Assessment")
        quality_msgs = []

        if 0.10 <= noise_ratio <= 0.25:
            quality_msgs.append(("✅ Optimal noise ratio (10-25%)", "green"))
        elif 0.05 <= noise_ratio <= 0.35:
            quality_msgs.append(("🟡 Acceptable noise ratio", "orange"))
        else:
            if noise_ratio < 0.05:
                quality_msgs.append(("🔴 Very low noise - possible over-clustering", "red"))
            else:
                quality_msgs.append(("🔴 High noise - consider parameter adjustment", "red"))

        if quality_msgs:
            cols = st.columns(len(quality_msgs))
            for i, (msg, color) in enumerate(quality_msgs):
                cols[i].markdown(f"<div style='color:{color}; font-weight:bold;'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.write("No quality messages available.")

        if st.button("Proceed to Step 5: Forecasting 📅", on_click=go_to_step_5):
            pass 

# Step 5: Forecasting
elif menu == "Forecasting":
    st.subheader("📅 Forecasting with ARIMA")

    if st.session_state.df is None:
        st.warning("Please run clustering first.")
    else:
        df = st.session_state.df

        df['month'] = df['timestamp'].dt.to_period('M')
        cluster_counts_monthly = df.groupby('month')['cluster'].count()
        cluster_counts_monthly.index = cluster_counts_monthly.index.to_timestamp()
        if cluster_counts_monthly.empty:
            st.warning("No data available for monthly forecasting.")
        else:
            model = ARIMA(cluster_counts_monthly, order=(1, 1, 1))
            results = model.fit()

            forecast = results.forecast(steps=7)

            df_observed = cluster_counts_monthly.reset_index()
            df_observed.columns = ['Date', 'Count']
            df_observed['Type'] = 'Observed'

            df_forecast = forecast.reset_index()
            df_forecast.columns = ['Date', 'Count']
            df_forecast['Type'] = 'Forecast'

            df_plot = pd.concat([df_observed, df_forecast])

            st.write("---")

            chart_type = st.radio(
                "Select Chart Type:",
                ('Line', 'Bar'),
                horizontal=True,
                index=0
            )
            if chart_type == 'Line':

                fig2 = px.line(
                    df_plot, 
                    x='Date', 
                    y='Count', 
                    color='Type',            
                    title="Crime Incident Forecast (Monthly)",
                    markers=True,            
                    labels={'Count': 'Monthly Incident Count', 'Date': 'Month'},
                    line_dash='Type',         
                    hover_data={
                        'Date': '|%Y-%m',    
                        'Count': True,        
                        'Type': False         
                    }
                )

                fig2.update_traces(line=dict(shape='spline'))

                fig2.update_traces(selector=dict(name='Observed'), line=dict(color='blue'))
                fig2.update_traces(selector=dict(name='Forecast'), line=dict(color='orange'))
                
                st.plotly_chart(fig2, use_container_width=True)

            elif chart_type == 'Bar':

                fig2 = px.bar(
                    df_plot, 
                    x='Date', 
                    y='Count', 
                    color='Type', 
                    title="Crime Incident Forecast (Monthly)",
                    labels={'Count': 'Monthly Incident Count', 'Date': 'Month'},
                    barmode='group' 
                )
                fig2.update_traces(selector=dict(name='Observed'), marker_color='blue')
                fig2.update_traces(selector=dict(name='Forecast'), marker_color='orange')
                st.plotly_chart(fig2, use_container_width=True)

            if st.button("Proceed to Step 6: Comparison ⚖️", on_click=go_to_step_6):
                pass

if menu == "Comparison":
    st.subheader("⚖️ Performance Comparison: ST-DBSCAN vs CD-DBSCAN")

    if st.session_state.df is None:
        st.warning("Please upload and preprocess data first.")
    else:
        df = st.session_state.df.copy()
        total_points = len(df)

        # --- Run ST-DBSCAN ---
        start = time.perf_counter()

        st_df, st_clusters, st_avg_query_time = run_st_dbscan(df.copy()) 
        st_time = time.perf_counter() - start # Fix: Base time calculation
        st_latency_ms = st_time * 1000  # convert to ms
        st_noise_points = np.sum(st_df['cluster'] == -1)
        st_clustered_points = total_points - st_noise_points
        st_noise_ratio = st_noise_points / total_points

        # --- Run CD-DBSCAN ---
        start = time.perf_counter()

        cd_df, cd_clusters, cd_avg_query_time = run_cd_dbscan(df.copy())
        cd_time = time.perf_counter() - start # Fix: Base time calculation
        cd_latency_ms = cd_time * 1000  # convert to ms
        cd_noise_points = np.sum(cd_df['cluster'] == -1)
        cd_clustered_points = total_points - cd_noise_points
        cd_noise_ratio = cd_noise_points / total_points

        # --- 2.Performance Metrics ---
        st.write("### 📊Performance Metrics")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**ST-DBSCAN**")
            st.metric("Clusters Found", st_clusters)

        with col2:
            st.write("**CD-DBSCAN**")
            st.metric("Clusters Found", cd_clusters)

        st.info("The **total number of distinct clusters** identified by the algorithm.")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Clustered Points", f"{st_clustered_points:,}")

        with col2:
            st.metric("Clustered Points", f"{cd_clustered_points:,}")

        st.info("The **total number of points** assigned to clusters.")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Noise Points", f"{st_noise_points:,}")

        with col2:
            st.metric("Noise Points", f"{cd_noise_points:,}")

        st.info("The **total number of points** identified as noise.")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Noise Ratio", f"{st_noise_ratio:.1%}")

        with col2:
            st.metric("Noise Ratio", f"{cd_noise_ratio:.1%}")


        st.info("The **proportion of points** identified as noise.")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Execution Time (s)", f"{st_time:.2f}")

        with col2:
            st.metric("Execution Time (s)", f"{cd_time:.2f}")

        st.info("The **total time taken** to execute the algorithm.")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Latency (ms)", f"{st_latency_ms:.2f}")

        with col2:
            st.metric("Latency (ms)", f"{cd_latency_ms:.2f}")

        st.info("The **time taken to process each query**.")
        
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Query Responsiveness (ms/query)", f"{st_avg_query_time:.4f}")

        with col2:
            st.metric("Query Responsiveness (ms/query)", f"{cd_avg_query_time:.4f}")

        st.info("The **average time taken** to respond to each neighbor query.")

        st.write("### 📈 Metric Visualization Comparison")
        col_vis1, col_vis2 = st.columns(2)

        metrics_data = pd.DataFrame({
            'Algorithm': ['ST-DBSCAN', 'CD-DBSCAN'],
            'Execution Time (s)': [st_time, cd_time],
            'Clusters Found': [st_clusters, cd_clusters],
            'Clustered Points': [st_clustered_points, cd_clustered_points],
            'Noise Points': [st_noise_points, cd_noise_points],
            'Latency (ms)': [st_latency_ms, cd_latency_ms],
            'Query Responsiveness (ms/query)': [st_avg_query_time, cd_avg_query_time],
            'Noise Ratio': [st_noise_ratio, cd_noise_ratio] # Added for Plot 4
        })

        with col_vis1:
            df_melt_p1 = metrics_data.melt( # Renamed df_melt
                id_vars='Algorithm', 
                value_vars=['Execution Time (s)', 'Clusters Found'], 
                var_name='Metric Type', 
                value_name='Value'
            )
            fig_speed = px.bar(
                df_melt_p1,
                x='Algorithm',
                y='Value',
                color='Metric Type',
                barmode='group',
                title="Execution Speed & Cluster Count Comparison",
                color_discrete_map={'Execution Time (s)': 'red', 'Clusters Found': 'blue'}
            )
            fig_speed.update_layout(xaxis_title="", yaxis_title="Value", legend_title="Metric", height=400)
            st.plotly_chart(fig_speed, use_container_width=True)
            st.info(
                "**How Fast & How Many Groups?** 🏎️ This graph compares how quickly each method runs (**Execution Time**, the red bars) and how many groups, or **Clusters**, it manages to find (the blue bars). A **lower red bar** means a faster, more efficient algorithm. The **blue bars** show the total number of distinct groups the algorithm identified in the data."
            )

        with col_vis2:
            df_ratio = metrics_data.melt(
                id_vars='Algorithm',
                value_vars=['Clustered Points', 'Noise Points'],
                var_name='Point Type',
                value_name='Count'
            )
            fig_ratio = px.bar(
                df_ratio,
                x="Algorithm",
                y="Count",
                color="Point Type",
                title="Clustered vs. Noise Points",
                text_auto=True, 
                color_discrete_map={'Clustered Points': 'green', 'Noise Points': 'yellow'}
            )
            fig_ratio.update_layout(xaxis_title="", yaxis_title="Number of Points", height=400)
            st.plotly_chart(fig_ratio, use_container_width=True)
            st.info(
                "**What's Signal and What's Noise?** 💡 This chart shows how well each method uses the data. **Clustered Points** (green) are the data points that belong to a group, which is the 'signal' we want. **Noise Points** (orange) are points that didn't fit into any group—they're the 'clutter' or outliers. The goal is often to have a high number of **Clustered Points** (green) to ensure the algorithm is grouping most of the data."
            )

        with col_vis1:
            df_melt_p3 = metrics_data.melt(
                id_vars='Algorithm',
                value_vars=['Query Responsiveness (ms/query)'],
                var_name='Metric Type',
                value_name='Value' 
            )
            
            fig_query_responsiveness = px.bar(
                df_melt_p3,
                x='Algorithm',
                y='Value', 
                title="Query Responsiveness Comparison",
                color='Algorithm',
                color_discrete_map={'ST-DBSCAN': 'red', 'CD-DBSCAN': 'blue'}
            )
            
            fig_query_responsiveness.update_layout(
                xaxis_title="",
                yaxis_title="Query Responsiveness (ms/query)",  
                legend_title="Algorithm",
                height=400
            )
            
            st.plotly_chart(fig_query_responsiveness, use_container_width=True)
            st.info(
            "**How Quick is the Answer?** ⏱️ This metric, **Query Responsiveness**, is a measure of speed *after* the initial grouping is done. It tells you how long (in milliseconds) the system takes to answer a simple question, like 'What group does this new data point belong to?' **Shorter bars** mean faster answers and a more responsive system, which is great for real-time applications."
            )

        with col_vis2:
            fig_noise = px.bar(
                metrics_data,
                x='Algorithm',
                y='Noise Ratio',
                title="Noise Ratio Comparison",
                text_auto='.1%', 
                color='Algorithm',
                color_discrete_map={'ST-DBSCAN': 'red', 'CD-DBSCAN': 'blue'}
            )   

            fig_noise.update_layout(xaxis_title="", yaxis_title="Noise Ratio", legend_title="Algorithm", height=400)
            st.plotly_chart(fig_noise, use_container_width=True)
            st.info(
                "**The Percentage of Outliers** 📉 The **Noise Ratio** is simply the percentage of all your data that the algorithm decided was 'Noise' (clutter/outliers) and didn't put into any cluster. A **lower percentage** (smaller bar) is generally better, as it indicates the algorithm was able to organize a larger portion of the total data into meaningful groups."
            )

        st.write("### 🗺️ Geospatial Cluster Visualization Comparison")

        st_df_plot_filtered = st_df[(st_df['LAT'] != 0) & (st_df['LON'] != 0) & (st_df['cluster'] != -1)]
        cd_df_plot_filtered = cd_df[(cd_df['LAT'] != 0) & (cd_df['LON'] != 0) & (cd_df['cluster'] != -1)]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔵 ST-DBSCAN Algorithm")
            if not st_df_plot_filtered.empty:
                fig_st = px.density_mapbox(
                    st_df_plot_filtered.sample(min(10000, len(st_df_plot_filtered))),
                    lat='LAT',
                    lon='LON',
                    z='cluster',
                    radius=12,
                    center=dict(lat=st_df_plot_filtered['LAT'].mean(), lon=st_df_plot_filtered['LON'].mean()),
                    zoom=10,
                    mapbox_style="open-street-map",
                    title=f"ST-DBSCAN: {st_clusters} Clusters | {len(st_df_plot_filtered):,} Points",
                    labels={'cluster': 'Cluster ID'},
                    color_continuous_scale='Blues',
                    hover_data={'cluster': True, 'LAT': ':.4f', 'LON': ':.4f'}
                )
                fig_st.update_layout(height=500, margin={"r":0,"t":50,"l":0,"b":0})
                st.plotly_chart(fig_st, use_container_width=True)
            else:
                st.warning("⚠️ No clustered points to display (all noise)")

        with col2:
            st.markdown("#### 🔴 CD-DBSCAN Algorithm")
            if not cd_df_plot_filtered.empty:
                fig_cd = px.density_mapbox(
                    cd_df_plot_filtered.sample(min(10000, len(cd_df_plot_filtered))),
                    lat='LAT',
                    lon='LON',
                    z='cluster',
                    radius=12,
                    center=dict(lat=cd_df_plot_filtered['LAT'].mean(), lon=cd_df_plot_filtered['LON'].mean()),
                    zoom=10,
                    mapbox_style="open-street-map",
                    title=f"CD-DBSCAN: {cd_clusters} Clusters | {len(cd_df_plot_filtered):,} Points",
                    labels={'cluster': 'Cluster ID'},
                    color_continuous_scale='Reds',
                    hover_data={'cluster': True, 'LAT': ':.4f', 'LON': ':.4f'}
                )
                fig_cd.update_layout(height=500, margin={"r":0,"t":50,"l":0,"b":0})
                st.plotly_chart(fig_cd, use_container_width=True)
            else:
                st.warning("⚠️ No clustered points to display (all noise)")

        st.info(
            "### 🗺️ Interactive Choropleth Analysis\n\n"
            f"**ST-DBSCAN Performance:**\n"
            f"- Identified **{st_clusters} distinct clusters** across **{len(st_df_plot_filtered):,} locations**\n"
            f"- Color intensity (blue scale) represents cluster density and geographic spread\n"
            f"- Characteristic pattern: **Broader spatial coverage** with moderate density concentrations\n\n"
            f"**CD-DBSCAN Performance:**\n"
            f"- Identified **{cd_clusters} distinct clusters** across **{len(cd_df_plot_filtered):,} locations**\n"
            f"- Color intensity (red scale) represents cluster density and geographic spread\n"
            f"- Characteristic pattern: **Tighter spatial clustering** with high-density hotspots\n\n"
            "**Key Differences:**\n"
            "- **ST-DBSCAN (Blues):** Creates more **expansive cluster boundaries**, capturing loosely connected spatial patterns ideal for identifying broad crime zones\n"
            "- **CD-DBSCAN (Reds):** Forms **compact, high-density clusters**, pinpointing critical hotspots and concentrated crime activity centers\n\n"
            "💡 *Hover over any area on the maps to see exact cluster IDs and coordinates*"
        )

        if st.button("Proceed to Step 7: Download Output 📄", on_click=go_to_step_7):
            pass

elif menu == "Download Output":
    st.subheader("💾 Output & Save Clustered Data")
    
    if st.session_state.df is None:
        st.warning("Please upload and run clustering first to see the data.")
    else:
        df = st.session_state.df
        
        if 'cluster' not in df.columns:
            st.warning("Data is uploaded but clustering has not been run. Please go to '2. Run Clustering'.")
        else:
            st.write(f"Displaying the first {min(100, len(df))} rows of the clustered data.")
            st.dataframe(df.head(100))
            
            st.markdown("---")
            
            # Create the save button
            if st.button("📁 Save to Output History Folder"):
                
                # Call the new saving function
                with st.spinner("Saving file to system folder..."):
                    save_result = save_clustered_csv_to_folder(df)

                if save_result['status'] == "success":
                    # Success message with the path
                    st.success(f"✅ **Results successfully saved!** You can find your output history here:")
                    st.code(save_result['path'])
                else:
                    # Error message
                    st.error(f"❌ **Failed to save file!** Error: {save_result['message']}")
            
            st.info("This is the final step. Your data is ready to be saved to your local history folder.")
