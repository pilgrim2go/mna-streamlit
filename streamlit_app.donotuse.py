import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date

# Page configuration
st.set_page_config(
    page_title="Blockchain Log Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Blockchain Log Analyzer - Visualization Dashboard")
st.markdown("Interactive analysis and visualization of blockchain operation logs")

# Sidebar
st.sidebar.header("🔧 Data Source Configuration")

# Default to S3 with chromaway-tmp bucket and parsed-logs prefix
data_source = st.sidebar.selectbox(
    "Choose data source",
    ["S3 Bucket", "Local Folder", "Upload File", "Upload Zip"],
    index=0
)

date_range = st.sidebar.date_input(
    "Date range (optional)",
    value=(date.today(), date.today()),
    min_value=date(2020, 1, 1),
    max_value=date(2100, 1, 1)
)

# Add filters section
st.sidebar.header("🔍 Data Filters")

# Operation filter
operation_filter = st.sidebar.multiselect(
    "Filter by Operation",
    options=st.sidebar.session_state.get('operation_options', []),
    default=[],
    help="Select specific operations to include in analysis"
)

# Unique ID filter
unique_id_filter = st.sidebar.multiselect(
    "Filter by Unique ID",
    options=st.sidebar.session_state.get('unique_id_options', []),
    default=[],
    help="Select specific unique IDs to include in analysis"
)

# Duration range filter
st.sidebar.subheader("⏱️ Duration Filter")
duration_min = st.sidebar.number_input(
    "Min Duration (ms)",
    min_value=0.0,
    value=0.0,
    step=1.0,
    help="Minimum duration threshold"
)

duration_max = st.sidebar.number_input(
    "Max Duration (ms)",
    min_value=0.0,
    value=float('inf') if duration_min == 0.0 else duration_min + 1000.0,
    step=1.0,
    help="Maximum duration threshold"
)

df = None

# --- S3 Data Source ---
if data_source == "S3 Bucket":
    st.sidebar.subheader("☁️ S3 Settings")
    s3_bucket = st.sidebar.text_input("S3 Bucket", value="chromaway-tmp")
    s3_prefix = st.sidebar.text_input("S3 Prefix", value="parsed-logs")
    
    if st.sidebar.button("🚀 Load from S3", type="primary"):
        with st.spinner("Loading data from S3..."):
            df = load_from_s3(s3_bucket, s3_prefix, date_range)

# --- Local Folder Data Source ---
elif data_source == "Local Folder":
    st.sidebar.subheader("📁 Local Folder")
    folder_path = st.sidebar.text_input("Folder Path", value="./_temp/parsed")
    
    if st.sidebar.button("📂 Load from Folder", type="primary"):
        with st.spinner("Loading data from local folder..."):
            df = load_from_local_folder(folder_path, date_range)

# --- Single File Upload Data Source ---
elif data_source == "Upload File":
    st.sidebar.subheader("📤 Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file and st.sidebar.button("📥 Load Uploaded File", type="primary"):
        with st.spinner("Loading uploaded file..."):
            df = load_from_uploaded_file(uploaded_file, date_range)

# --- Zip Upload Data Source ---
elif data_source == "Upload Zip":
    st.sidebar.subheader("📦 Upload Zip Folder")
    uploaded_zip = st.sidebar.file_uploader("Choose a zip file", type=["zip"])
    
    if uploaded_zip and st.sidebar.button("📦 Load Uploaded Zip", type="primary"):
        with st.spinner("Loading from zip file..."):
            df = load_from_zip(uploaded_zip, date_range)

# --- Show Data and Analytics ---
if df is not None and not df.empty:
    st.success(f"✅ Successfully loaded {len(df):,} records from {data_source}")
    
    # Apply filters
    original_df = df.copy()
    
    # Update filter options based on loaded data
    if 'operation' in df.columns:
        all_operations = sorted(df['operation'].unique().tolist())
        st.sidebar.session_state['operation_options'] = all_operations
    
    if 'unique_id' in df.columns:
        all_unique_ids = sorted(df['unique_id'].unique().tolist())
        st.sidebar.session_state['unique_id_options'] = all_unique_ids
    
    # Apply operation filter
    if operation_filter and 'operation' in df.columns:
        df = df[df['operation'].isin(operation_filter)]
        st.info(f"🔍 Filtered to {len(operation_filter)} operations: {', '.join(operation_filter)}")
    
    # Apply unique ID filter
    if unique_id_filter and 'unique_id' in df.columns:
        df = df[df['unique_id'].isin(unique_id_filter)]
        st.info(f"🔍 Filtered to {len(unique_id_filter)} unique IDs: {', '.join(unique_id_filter)}")
    
    # Apply duration filter
    if 'duration_ms' in df.columns:
        if duration_min > 0:
            df = df[df['duration_ms'] >= duration_min]
        if duration_max < float('inf'):
            df = df[df['duration_ms'] <= duration_max]
            
        if duration_min > 0 or duration_max < float('inf'):
            st.info(f"⏱️ Filtered duration range: {duration_min:.1f} - {duration_max:.1f} ms")
    
    # Show filtered data count
    if len(df) != len(original_df):
        st.warning(f"📊 Showing {len(df):,} records after filtering (from {len(original_df):,} total)")
    
    # Display all visualizations
    st.header("📈 Key Metrics")
    display_metrics(df)
    
    st.header("⏱️ Duration Analysis")
    plot_duration_analysis(df)
    
    st.header("📊 Time Series Analysis")
    plot_time_series_analysis(df)
    
    st.header("🔄 Operation Analysis")
    plot_operation_analysis(df)
    
    st.header("🚨 Anomaly Detection")
    plot_anomaly_detection(df)

    st.header("🔍 Unique ID Analysis")
    plot_unique_id_analysis(df)
    
    st.header("📋 Detailed Data & Export")
    display_detailed_data(df)
    
elif df is not None and df.empty:
    st.warning("⚠️ No data found for the selected criteria. Try adjusting your filters or date range.")
else:
    st.info("👈 Please select a data source and load data to begin analysis.") 