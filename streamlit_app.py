import streamlit as st
import pandas as pd
import numpy as np
import glob
import os

st.set_page_config(page_title="Operation Log Analyzer", layout="wide")

st.title("Operation Log Analyzer - Visualization")

# File selector (default to parsed folder)
default_folder = os.path.join(os.path.dirname(__file__), '_temp', 'parsed')
folder_path = st.text_input("Parsed folder to load:", value=default_folder)

@st.cache_data
def load_all_parsed(folder):
    # Recursively find all CSVs in the folder
    csv_files = sorted(glob.glob(os.path.join(folder, '**', '*.csv'), recursive=True))
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, parse_dates=["start_time", "end_time"], dtype={"date": str, "hour": str})
            # If date/hour columns missing, try to extract from path
            if 'date' not in df.columns or 'hour' not in df.columns:
                m = None
                date, hour = None, None
                m = [x for x in f.split(os.sep) if x.startswith('date=')]
                if m:
                    date = m[0].split('=')[1]
                m = [x for x in f.split(os.sep) if x.startswith('hour=')]
                if m:
                    hour = m[0].split('=')[1]
                if 'date' not in df.columns:
                    df['date'] = date
                if 'hour' not in df.columns:
                    df['hour'] = hour
            dfs.append(df)
        except Exception as e:
            st.warning(f"Failed to load {f}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

# File uploader
uploaded_file = st.file_uploader("Or upload a parsed CSV file to visualize:", type=["csv"])

@st.cache_data
def load_uploaded_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["start_time", "end_time"], dtype={"date": str, "hour": str})
        # If date/hour columns missing, try to extract from filename
        if 'date' not in df.columns or 'hour' not in df.columns:
            date, hour = None, None
            if hasattr(uploaded_file, 'name'):
                fname = uploaded_file.name
                import re
                m = re.search(r'(\d{4})(\d{2})(\d{2})_(\d{2})', fname)
                if m:
                    date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                    hour = m.group(4)
            if 'date' not in df.columns:
                df['date'] = date
            if 'hour' not in df.columns:
                df['hour'] = hour
        return df
    except Exception as e:
        st.warning(f"Failed to load uploaded file: {e}")
        return pd.DataFrame()

try:
    if uploaded_file is not None:
        df = load_uploaded_csv(uploaded_file)
        if df.empty:
            st.warning("Uploaded file is empty or could not be loaded.")
            st.stop()
    else:
        df = load_all_parsed(folder_path)
        if df.empty:
            st.warning("No parsed CSVs found in the folder.")
            st.stop()
    # Sidebar filters
    st.sidebar.header("Filters")
    # Date filter
    date_options = ["All"] + sorted(df["date"].dropna().unique())
    selected_date = st.sidebar.selectbox("Date", date_options)
    filtered_df = df.copy()
    if selected_date != "All":
        filtered_df = filtered_df[filtered_df["date"] == selected_date]
    # Operation filter
    operation_options = ["All"] + sorted(filtered_df["operation"].unique())
    selected_operation = st.sidebar.selectbox("Operation", operation_options)
    if selected_operation != "All":
        filtered_df = filtered_df[filtered_df["operation"] == selected_operation]
    # Unique ID filter
    unique_id_options = ["All"] + sorted(filtered_df["unique_id"].astype(str).unique())
    selected_unique_id = st.sidebar.selectbox("Unique ID", unique_id_options)
    if selected_unique_id != "All":
        filtered_df = filtered_df[filtered_df["unique_id"].astype(str) == selected_unique_id]

    st.subheader("Summary Statistics")
    if not filtered_df.empty:
        stats = filtered_df.groupby("operation")["duration_ms"].agg(['count', 'min', 'max', 'mean', 'median', 'std'])
        st.dataframe(stats.style.format("{:.3f}"))
    else:
        st.info("No data for selected filters.")

    st.subheader("Detailed Data Table")
    st.dataframe(filtered_df)

    st.subheader("Duration Distribution")
    if not filtered_df.empty:
        # st.bar_chart(filtered_df["duration_ms"])
        st.box_chart = st.box_plot = None
        try:
            import altair as alt
            chart = alt.Chart(filtered_df).mark_boxplot().encode(
                y=alt.Y('duration_ms:Q', title='Duration (ms)'),
                x=alt.X('operation:N', title='Operation'),
                color='operation:N'
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            st.info("Install altair for boxplot visualization.") 

    # ---
    # 📊 1. Overall Duration Analysis
    st.subheader("Average Duration per Operation")
    if not filtered_df.empty:
        import altair as alt
        avg_duration = filtered_df.groupby("operation")["duration_ms"].mean().reset_index()
        chart = alt.Chart(avg_duration).mark_bar().encode(
            x=alt.X("duration_ms:Q", title="Avg Duration (ms)"),
            y=alt.Y("operation:N", sort='-x')
        ).properties(title="Average Duration per Operation")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No data for average duration chart.")

    # ---
    # 🔄 2. Top N Slowest Calls
    st.subheader("Top 10 Slowest Operations")
    if not filtered_df.empty:
        slowest = filtered_df.sort_values("duration_ms", ascending=False).head(10)
        st.write("🚨 Top 10 Slowest Operations", slowest)
    else:
        st.info("No data for slowest operations table.")

    # ---
    # 🧍 3. Per-Unique ID Operation Frequency
    st.subheader("Operation Frequency by Unique ID")
    if not filtered_df.empty:
        freq = filtered_df.groupby(["unique_id", "operation"]).size().reset_index(name="count")
        st.dataframe(freq.sort_values("count", ascending=False).head(20))
    else:
        st.info("No data for operation frequency table.")

    # ---
    # 🕓 4. Time Series of Requests
    st.subheader("Requests per Minute Over Time")
    if not filtered_df.empty:
        ts = filtered_df.set_index("start_time").resample("1T").count()["operation"].reset_index()
        ts = ts.rename(columns={"start_time": "Time", "operation": "Requests per Minute"})
        st.line_chart(ts.set_index("Time")["Requests per Minute"])
    else:
        st.info("No data for time series chart.")

    
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop() 