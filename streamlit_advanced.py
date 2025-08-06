import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import zipfile
import glob
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import awswrangler as wr
    AWS_WRANGLER_AVAILABLE = True
except ImportError:
    AWS_WRANGLER_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Advanced Blockchain Log Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Robust date filtering ---
def robust_date_filter(df, date_range):
    if 'date' not in df.columns or not date_range:
        return df
    start_date, end_date = date_range
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
    return df[mask]

# --- Data Filtering Function ---
def apply_filters(df, operation_filter=None, unique_id_filter=None, duration_min=None, duration_max=None, date_range=None):
    """
    Apply filters to the dataframe without modifying the original
    """
    if df is None or df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Apply date filter
    if date_range:
        filtered_df = robust_date_filter(filtered_df, date_range)
    
    # Apply operation filter
    if operation_filter and 'operation' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['operation'].isin(operation_filter)]
    
    # Apply unique ID filter
    if unique_id_filter and 'unique_id' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['unique_id'].isin(unique_id_filter)]
    
    # Apply duration filter
    if 'duration_ms' in filtered_df.columns:
        if duration_min is not None and duration_min > 0:
            filtered_df = filtered_df[filtered_df['duration_ms'] >= duration_min]
        if duration_max is not None and duration_max < float('inf'):
            filtered_df = filtered_df[filtered_df['duration_ms'] <= duration_max]
    
    return filtered_df

# --- Data Loading Status Management ---
def get_data_loading_status():
    """Get the current data loading status and source info"""
    return {
        'loaded': st.session_state.get('data_loaded', False),
        'source': st.session_state.get('data_source', ''),
        'bucket': st.session_state.get('s3_bucket', ''),
        'prefix': st.session_state.get('s3_prefix', ''),
        'folder_path': st.session_state.get('folder_path', ''),
        'file_name': st.session_state.get('file_name', ''),
        'record_count': st.session_state.get('record_count', 0),
        'last_loaded': st.session_state.get('last_loaded', None)
    }

def set_data_loading_status(source, **kwargs):
    """Set the data loading status"""
    st.session_state['data_loaded'] = True
    st.session_state['data_source'] = source
    st.session_state['last_loaded'] = datetime.now()
    
    for key, value in kwargs.items():
        st.session_state[key] = value

def clear_data_cache():
    """Clear the cached data and reset status"""
    keys_to_clear = [
        'cached_data', 'data_loaded', 'data_source', 's3_bucket', 's3_prefix',
        'folder_path', 'file_name', 'record_count', 'last_loaded',
        'operation_options', 'unique_id_options', 'filter_history'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def add_filter_history(filter_config):
    """Add current filter configuration to history"""
    if 'filter_history' not in st.session_state:
        st.session_state['filter_history'] = []
    
    timestamp = datetime.now().strftime('%H:%M:%S')
    filter_entry = {
        'timestamp': timestamp,
        'config': filter_config,
        'record_count': len(filter_config.get('data', []))
    }
    
    # Keep only last 10 filter configurations
    st.session_state['filter_history'] = st.session_state['filter_history'][-9:] + [filter_entry]

# --- S3 Data Loading ---
def load_from_s3(s3_bucket, s3_prefix):
    try:
        s3_path = f"s3://{s3_bucket}/{s3_prefix}/"
        with st.spinner("Loading data from S3..."):
            # Try authenticated access first, then fallback to anonymous
            try:
                # Use AWS Data Wrangler with authenticated access
                import awswrangler as wr
                df = wr.s3.read_csv(path=s3_path, dataset=False)
                
                # If dataset=False doesn't work, try with dataset=True
                if df is None or df.empty:
                    df = wr.s3.read_csv(path=s3_path, dataset=True, ignore_partition_by=True)
                    
            except Exception as auth_error:
                st.info("Authenticated S3 access failed, trying anonymous access...")
                
                # Fallback to anonymous access using s3fs
                import s3fs
                
                # Create filesystem object for anonymous S3 access
                fs = s3fs.S3FileSystem(anon=True)
                
                # List all CSV files in the S3 path
                csv_files = fs.glob(f"{s3_bucket}/{s3_prefix}/**/*.csv")
                
                if not csv_files:
                    st.error("No CSV files found in S3 bucket")
                    return None
                
                # Load all CSV files
                dfs = []
                for csv_file in csv_files:
                    try:
                        # Read CSV using pandas with s3fs
                        df_chunk = pd.read_csv(f"s3://{csv_file}", storage_options={'anon': True})
                        dfs.append(df_chunk)
                    except Exception as e:
                        st.warning(f"Failed to load {csv_file}: {e}")
                
                if not dfs:
                    st.error("No data could be loaded from S3")
                    return None
                
                # Combine all dataframes
                df = pd.concat(dfs, ignore_index=True)
        
        if df is None or df.empty:
            st.error("No data found in S3 bucket")
            return None
            
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'hour' in df.columns:
            df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(-1).astype(int)
        if 'start_time' in df.columns:
            df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        if 'end_time' in df.columns:
            df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
        if 'duration_ms' in df.columns:
            df['duration_ms'] = pd.to_numeric(df['duration_ms'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading from S3: {e}")
        return None

# --- Local Folder Data Loading ---
def load_from_local_folder(folder_path):
    all_csvs = glob.glob(os.path.join(folder_path, '**', '*.csv'), recursive=True)
    dfs = []
    for csv_path in all_csvs:
        try:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        except Exception as e:
            st.warning(f"Failed to load {csv_path}: {e}")
    if not dfs:
        st.error("No CSV files found in folder.")
        return None
    df = pd.concat(dfs, ignore_index=True)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'hour' in df.columns:
        df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(-1).astype(int)
    if 'start_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    if 'end_time' in df.columns:
        df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
    if 'duration_ms' in df.columns:
        df['duration_ms'] = pd.to_numeric(df['duration_ms'], errors='coerce')
    return df

# --- Zip Upload Data Loading ---
def load_from_zip(uploaded_zip):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(uploaded_zip, 'r') as z:
            z.extractall(tmpdir)
        return load_from_local_folder(tmpdir)

# --- Single File Upload Data Loading ---
def load_from_uploaded_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'hour' in df.columns:
            df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(-1).astype(int)
        if 'start_time' in df.columns:
            df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        if 'end_time' in df.columns:
            df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
        if 'duration_ms' in df.columns:
            df['duration_ms'] = pd.to_numeric(df['duration_ms'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Failed to load uploaded file: {e}")
        return None

# --- Visualization Functions ---
def display_metrics(df):
    """Display key metrics at the top"""
    if df.empty:
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Operations", len(df))
    
    with col2:
        avg_duration = df['duration_ms'].mean() if 'duration_ms' in df.columns else 0
        st.metric("Avg Duration", f"{avg_duration:.1f} ms")
    
    with col3:
        unique_operations = df['operation'].nunique() if 'operation' in df.columns else 0
        st.metric("Unique Operations", unique_operations)
    
    with col4:
        unique_ids = df['unique_id'].nunique() if 'unique_id' in df.columns else 0
        st.metric("Unique IDs", unique_ids)
    
    with col5:
        date_range = f"{df['date'].min().date()} to {df['date'].max().date()}" if 'date' in df.columns else "N/A"
        st.metric("Date Range", date_range)

def plot_duration_analysis(df):
    """Plot duration analysis charts"""
    if df.empty or 'duration_ms' not in df.columns:
        st.info("No duration data available for analysis")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Duration distribution by operation
        fig = px.box(df, x='operation', y='duration_ms', 
                    title='Duration Distribution by Operation',
                    labels={'duration_ms': 'Duration (ms)', 'operation': 'Operation'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average duration by operation
        avg_duration = df.groupby('operation')['duration_ms'].mean().reset_index()
        fig = px.bar(avg_duration, x='duration_ms', y='operation', orientation='h',
                    title='Average Duration by Operation',
                    labels={'duration_ms': 'Average Duration (ms)', 'operation': 'Operation'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def plot_time_series_analysis(df):
    """Plot time series analysis"""
    if df.empty or 'start_time' not in df.columns:
        st.info("No time series data available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Operations per hour
        df_hourly = df.set_index('start_time').resample('1H').size().reset_index()
        df_hourly.columns = ['Time', 'Operations']
        fig = px.line(df_hourly, x='Time', y='Operations',
                     title='Operations per Hour',
                     labels={'Operations': 'Number of Operations', 'Time': 'Time'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average duration over time
        df_duration = df.set_index('start_time').resample('1H')['duration_ms'].mean().reset_index()
        fig = px.line(df_duration, x='start_time', y='duration_ms',
                     title='Average Duration Over Time',
                     labels={'duration_ms': 'Average Duration (ms)', 'start_time': 'Time'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def plot_operation_analysis(df):
    """Plot operation frequency and performance analysis"""
    if df.empty or 'operation' not in df.columns:
        st.info("No operation data available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Operation frequency
        op_freq = df['operation'].value_counts().reset_index()
        op_freq.columns = ['Operation', 'Count']
        fig = px.pie(op_freq, values='Count', names='Operation',
                    title='Operation Frequency Distribution')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top operations by average duration
        if 'duration_ms' in df.columns:
            top_ops = df.groupby('operation')['duration_ms'].agg(['mean', 'count']).reset_index()
            top_ops = top_ops.sort_values('mean', ascending=False).head(10)
            fig = px.bar(top_ops, x='operation', y='mean',
                        title='Top 10 Operations by Average Duration',
                        labels={'mean': 'Average Duration (ms)', 'operation': 'Operation'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def plot_anomaly_detection(df):
    """Plot anomaly detection for slow operations"""
    if df.empty or 'duration_ms' not in df.columns:
        st.info("No duration data available for anomaly detection")
        return
    
    # Calculate outliers using IQR method
    Q1 = df['duration_ms'].quantile(0.25)
    Q3 = df['duration_ms'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    
    outliers = df[df['duration_ms'] > outlier_threshold]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Outlier distribution
        fig = px.histogram(df, x='duration_ms', nbins=50,
                          title='Duration Distribution with Outliers',
                          labels={'duration_ms': 'Duration (ms)', 'count': 'Frequency'})
        fig.add_vline(x=outlier_threshold, line_dash="dash", line_color="red",
                     annotation_text=f"Outlier Threshold: {outlier_threshold:.1f}ms")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top outliers
        if not outliers.empty:
            top_outliers = outliers.sort_values('duration_ms', ascending=False).head(10)
            fig = px.bar(top_outliers, x='operation', y='duration_ms',
                        title='Top 10 Outlier Operations',
                        labels={'duration_ms': 'Duration (ms)', 'operation': 'Operation'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No outliers detected")

def plot_unique_id_analysis(df):
    """Plot unique_id analysis charts"""
    if df.empty or 'unique_id' not in df.columns:
        st.info("No unique_id data available for analysis")
        return
    
    # Get top 20 most active unique IDs
    id_counts = df['unique_id'].value_counts().head(20)
    
    col1, col2 = st.columns(2)
    
 
  
    # Additional unique_id analysis
    col3, col4 = st.columns(2)
    
    with col3:
        # Average duration by unique_id
        if 'duration_ms' in df.columns:
            avg_duration_by_id = df.groupby('unique_id')['duration_ms'].agg(['mean', 'count']).reset_index()
            avg_duration_by_id.columns = ['Unique ID', 'Avg Duration', 'Operation Count']
            # Filter to IDs with at least 5 operations
            avg_duration_by_id = avg_duration_by_id[avg_duration_by_id['Operation Count'] >= 5]
            avg_duration_by_id = avg_duration_by_id.sort_values('Avg Duration', ascending=False).head(15)
            
            fig = px.scatter(
                avg_duration_by_id,
                x='Operation Count',
                y='Avg Duration',
                size='Operation Count',
                hover_data=['Unique ID'],
                title='Average Duration vs Operation Count by Unique ID',
                labels={'Avg Duration': 'Average Duration (ms)', 'Operation Count': 'Number of Operations'},
                color='Avg Duration',
                color_continuous_scale='reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Unique ID activity over time
        if 'start_time' in df.columns:
            df_time = df.copy()
            df_time['hour'] = df_time['start_time'].dt.hour
            hourly_activity = df_time.groupby('hour')['unique_id'].nunique().reset_index()
            hourly_activity.columns = ['Hour', 'Active Unique IDs']
            
            fig = px.line(
                hourly_activity,
                x='Hour',
                y='Active Unique IDs',
                title='Active Unique IDs by Hour',
                labels={'Active Unique IDs': 'Number of Active Unique IDs'},
                markers=True
            )
            fig.update_layout(height=400)
            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=8)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("📊 Unique ID Summary Statistics")
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        total_unique_ids = df['unique_id'].nunique()
        st.metric("Total Unique IDs", f"{total_unique_ids:,}")
    
    with col6:
        avg_operations_per_id = df.groupby('unique_id').size().mean()
        st.metric("Avg Operations per ID", f"{avg_operations_per_id:.1f}")
    
    with col7:
        if 'duration_ms' in df.columns:
            avg_duration_per_id = df.groupby('unique_id')['duration_ms'].mean().mean()
            st.metric("Avg Duration per ID", f"{avg_duration_per_id:.1f} ms")
    
    with col8:
        most_active_id = df['unique_id'].value_counts().index[0]
        most_active_count = df['unique_id'].value_counts().iloc[0]
        st.metric("Most Active ID", f"{most_active_id} ({most_active_count} ops)")

def display_detailed_data(df):
    """Display detailed data table with filters"""
    st.subheader("📋 Detailed Data")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'operation' in df.columns:
            operations = ['All'] + sorted(df['operation'].unique().tolist())
            selected_op = st.selectbox("Filter by Operation", operations)
            if selected_op != 'All':
                df = df[df['operation'] == selected_op]
    
    with col2:
        if 'duration_ms' in df.columns:
            min_duration = st.number_input("Min Duration (ms)", 
                                         value=float(df['duration_ms'].min()), 
                                         min_value=0.0)
            df = df[df['duration_ms'] >= min_duration]
    
    with col3:
        if 'duration_ms' in df.columns:
            max_duration = st.number_input("Max Duration (ms)", 
                                         value=float(df['duration_ms'].max()), 
                                         min_value=0.0)
            df = df[df['duration_ms'] <= max_duration]
    
    # Display data
    st.dataframe(df, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name=f"filtered_log_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# --- Main Streamlit App ---
def main():
    st.title("📊 Advanced Blockchain Log Analyzer")
    st.markdown("Comprehensive analysis and visualization of blockchain operation logs")
    
    # Initialize session state for data caching
    if 'cached_data' not in st.session_state:
        st.session_state['cached_data'] = None
    
    # Sidebar
    st.sidebar.header("🔧 Data Source Configuration")
    
    # Data loading status display
    status = get_data_loading_status()
    if status['loaded']:
        st.sidebar.success(f"✅ Data loaded from {status['source']}")
        st.sidebar.info(f"📊 {status['record_count']:,} records cached")
        if status['last_loaded']:
            st.sidebar.caption(f"Last loaded: {status['last_loaded'].strftime('%H:%M:%S')}")
        
        # Clear cache button
        if st.sidebar.button("🗑️ Clear Cache", type="secondary"):
            clear_data_cache()
            st.rerun()
    
    # Default to S3 with chromaway-tmp bucket and parsed-logs prefix
    data_source = st.sidebar.selectbox(
        "Choose data source",
        ["S3 Bucket", "Local Folder", "Upload File", "Upload Zip"],
        index=0
    )
    
    # Date range filter (now applied dynamically)
    date_range = st.sidebar.date_input(
        "Date range (optional)",
        value=(date.today(), date.today()),
        min_value=date(2020, 1, 1),
        max_value=date(2100, 1, 1)
    )
    
    # Data loading section
    df = None
    data_loaded_this_session = False
    
    # --- S3 Data Source ---
    if data_source == "S3 Bucket":
        st.sidebar.subheader("☁️ S3 Settings")
        s3_bucket = st.sidebar.text_input("S3 Bucket", value="chromaway-tmp")
        s3_prefix = st.sidebar.text_input("S3 Prefix", value="parsed-logs")
        
        if st.sidebar.button("🚀 Load from S3", type="primary"):
            with st.spinner("Loading data from S3..."):
                df = load_from_s3(s3_bucket, s3_prefix)
                if df is not None and not df.empty:
                    st.session_state['cached_data'] = df
                    set_data_loading_status(
                        "S3 Bucket",
                        s3_bucket=s3_bucket,
                        s3_prefix=s3_prefix,
                        record_count=len(df)
                    )
                    data_loaded_this_session = True
    
    # --- Local Folder Data Source ---
    elif data_source == "Local Folder":
        st.sidebar.subheader("📁 Local Folder")
        folder_path = st.sidebar.text_input("Folder Path", value="./_temp/parsed")
        
        if st.sidebar.button("📂 Load from Folder", type="primary"):
            with st.spinner("Loading data from local folder..."):
                df = load_from_local_folder(folder_path)
                if df is not None and not df.empty:
                    st.session_state['cached_data'] = df
                    set_data_loading_status(
                        "Local Folder",
                        folder_path=folder_path,
                        record_count=len(df)
                    )
                    data_loaded_this_session = True
    
    # --- Single File Upload Data Source ---
    elif data_source == "Upload File":
        st.sidebar.subheader("📤 Upload CSV File")
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file and st.sidebar.button("📥 Load Uploaded File", type="primary"):
            with st.spinner("Loading uploaded file..."):
                df = load_from_uploaded_file(uploaded_file)
                if df is not None and not df.empty:
                    st.session_state['cached_data'] = df
                    set_data_loading_status(
                        "Upload File",
                        file_name=uploaded_file.name,
                        record_count=len(df)
                    )
                    data_loaded_this_session = True
    
    # --- Zip Upload Data Source ---
    elif data_source == "Upload Zip":
        st.sidebar.subheader("📦 Upload Zip Folder")
        uploaded_zip = st.sidebar.file_uploader("Choose a zip file", type=["zip"])
        
        if uploaded_zip and st.sidebar.button("📦 Load Uploaded Zip", type="primary"):
            with st.spinner("Loading from zip file..."):
                df = load_from_zip(uploaded_zip)
                if df is not None and not df.empty:
                    st.session_state['cached_data'] = df
                    set_data_loading_status(
                        "Upload Zip",
                        file_name=uploaded_zip.name,
                        record_count=len(df)
                    )
                    data_loaded_this_session = True
    
    # Use cached data if available and no new data was loaded
    if df is None and st.session_state['cached_data'] is not None:
        df = st.session_state['cached_data']
    
    # Update filter options based on available data
    if df is not None and not df.empty:
        if 'operation' in df.columns:
            all_operations = sorted(df['operation'].unique().tolist())
            st.session_state['operation_options'] = all_operations
        
        if 'unique_id' in df.columns:
            all_unique_ids = sorted(df['unique_id'].unique().tolist())
            st.session_state['unique_id_options'] = all_unique_ids
    
    # Add filters section
    st.sidebar.header("🔍 Data Filters")
    
    # Show filter status and quick actions
    if df is not None and not df.empty:
        st.sidebar.info(f"📊 {len(df):,} total records available")
        
        # Quick filter actions
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Reset All Filters", type="secondary"):
                st.rerun()
        with col2:
            if st.button("📈 Show Top 10", type="secondary"):
                # This will be handled in the filtering logic
                pass
        
        # Show filter history
        if st.session_state.get('filter_history'):
            st.sidebar.subheader("📋 Recent Filters")
            for i, entry in enumerate(reversed(st.session_state['filter_history'][-3:])):  # Show last 3
                with st.sidebar.expander(f"{entry['timestamp']} - {entry['record_count']} records"):
                    config = entry['config']
                    if config.get('operation_filter'):
                        st.caption(f"Ops: {len(config['operation_filter'])} selected")
                    if config.get('unique_id_filter'):
                        st.caption(f"IDs: {len(config['unique_id_filter'])} selected")
                    if config.get('duration_min', 0) > 0 or config.get('duration_max', float('inf')) < float('inf'):
                        st.caption(f"Duration: {config['duration_min']:.0f}-{config['duration_max']:.0f}ms")
    
    # Operation filter with search
    if df is not None and not df.empty and 'operation' in df.columns:
        operation_filter = st.sidebar.multiselect(
            "Filter by Operation",
            options=st.session_state.get('operation_options', []),
            default=[],
            help="Select specific operations to include in analysis"
        )
        
        # Quick operation filters
        if st.session_state.get('operation_options'):
            st.sidebar.caption("Quick filters:")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("🐌 Slow Ops (>1s)"):
                    # This will be handled in the filtering logic
                    pass
            with col2:
                if st.button("⚡ Fast Ops (<100ms)"):
                    # This will be handled in the filtering logic
                    pass
    else:
        operation_filter = []
    
    # Unique ID filter with search
    if df is not None and not df.empty and 'unique_id' in df.columns:
        unique_id_filter = st.sidebar.multiselect(
            "Filter by Unique ID",
            options=st.session_state.get('unique_id_options', []),
            default=[],
            help="Select specific unique IDs to include in analysis"
        )
    else:
        unique_id_filter = []
    
    # Duration range filter with presets
    st.sidebar.subheader("⏱️ Duration Filter")
    
    # Duration presets
    duration_preset = st.sidebar.selectbox(
        "Duration Presets",
        ["Custom", "All", "Fast (<100ms)", "Medium (100ms-1s)", "Slow (>1s)", "Very Slow (>5s)"],
        help="Quick duration range presets"
    )
    
    # Set duration range based on preset
    if duration_preset == "All":
        duration_min = 0.0
        duration_max = float('inf')
    elif duration_preset == "Fast (<100ms)":
        duration_min = 0.0
        duration_max = 100.0
    elif duration_preset == "Medium (100ms-1s)":
        duration_min = 100.0
        duration_max = 1000.0
    elif duration_preset == "Slow (>1s)":
        duration_min = 1000.0
        duration_max = float('inf')
    elif duration_preset == "Very Slow (>5s)":
        duration_min = 5000.0
        duration_max = float('inf')
    else:  # Custom
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
            value=10000.0 if duration_min == 0.0 else duration_min + 1000.0,
            step=1.0,
            help="Maximum duration threshold"
        )
    
    # --- Show Data and Analytics ---
    if df is not None and not df.empty:
        # Show success message only when data is first loaded
        if data_loaded_this_session:
            st.success(f"✅ Successfully loaded {len(df):,} records from {data_source}")
        
        # Apply filters dynamically with progress indicator
        original_df = df.copy()
        
        with st.spinner("Applying filters..."):
            filtered_df = apply_filters(
                df, 
                operation_filter=operation_filter if operation_filter else None,
                unique_id_filter=unique_id_filter if unique_id_filter else None,
                duration_min=duration_min if duration_min > 0 else None,
                duration_max=duration_max if duration_max < float('inf') else None,
                date_range=date_range if date_range[0] != date_range[1] else None
            )
            
            # Track filter history
            filter_config = {
                'operation_filter': operation_filter,
                'unique_id_filter': unique_id_filter,
                'duration_min': duration_min,
                'duration_max': duration_max,
                'date_range': date_range,
                'data': filtered_df
            }
            add_filter_history(filter_config)
        
        # Show filter status and performance
        filters_applied = []
        if operation_filter:
            filters_applied.append(f"Operations: {len(operation_filter)} selected")
        if unique_id_filter:
            filters_applied.append(f"Unique IDs: {len(unique_id_filter)} selected")
        if duration_min > 0 or duration_max < float('inf'):
            filters_applied.append(f"Duration: {duration_min:.1f} - {duration_max:.1f} ms")
        if date_range and date_range[0] != date_range[1]:
            filters_applied.append(f"Date: {date_range[0]} to {date_range[1]}")
        
        if filters_applied:
            st.info(f"🔍 Active filters: {' | '.join(filters_applied)}")
        
        # Show filtered data count and performance metrics
        if len(filtered_df) != len(original_df):
            reduction_percent = ((len(original_df) - len(filtered_df)) / len(original_df)) * 100
            st.warning(f"📊 Showing {len(filtered_df):,} records after filtering (from {len(original_df):,} total, {reduction_percent:.1f}% reduction)")
        else:
            st.success(f"📊 Showing all {len(filtered_df):,} records")
        
        # Show quick stats about filtered data
        if len(filtered_df) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if 'duration_ms' in filtered_df.columns:
                    avg_duration = filtered_df['duration_ms'].mean()
                    st.metric("Avg Duration", f"{avg_duration:.1f} ms")
            with col2:
                if 'operation' in filtered_df.columns:
                    unique_ops = filtered_df['operation'].nunique()
                    st.metric("Unique Operations", unique_ops)
            with col3:
                if 'unique_id' in filtered_df.columns:
                    unique_ids = filtered_df['unique_id'].nunique()
                    st.metric("Unique IDs", unique_ids)
            with col4:
                if 'date' in filtered_df.columns:
                    date_range_str = f"{filtered_df['date'].min().date()} to {filtered_df['date'].max().date()}"
                    st.metric("Date Range", date_range_str)
        
        # Display all visualizations with filtered data
        st.header("📈 Key Metrics")
        display_metrics(filtered_df)
        
        st.header("⏱️ Duration Analysis")
        plot_duration_analysis(filtered_df)
        
        st.header("📊 Time Series Analysis")
        plot_time_series_analysis(filtered_df)
        
        st.header("🔄 Operation Analysis")
        plot_operation_analysis(filtered_df)
        
        st.header("🚨 Anomaly Detection")
        plot_anomaly_detection(filtered_df)

        st.header("🔍 Unique ID Analysis")
        plot_unique_id_analysis(filtered_df)
        
        st.header("📋 Detailed Data & Export")
        display_detailed_data(filtered_df)
        
    elif df is not None and df.empty:
        st.warning("⚠️ No data found for the selected criteria. Try adjusting your filters or date range.")
    else:
        st.info("👈 Please select a data source and load data to begin analysis.")

if __name__ == "__main__":
    main() 