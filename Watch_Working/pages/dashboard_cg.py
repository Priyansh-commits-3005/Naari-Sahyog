# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Static data to simulate streaming
DATA_ARRAY = [
   [0, 85.47, 37.31, 12.8247, 80.0458, 4.2],
    [0, 85.47, 37.31, 12.8247, 80.0458, 4.3],
    [0, 85.47, 37.50, 12.8247, 80.0458, 4.1],
    [0, 61.54, 37.31, 12.8247, 80.0458, 2.7],
    [0, 61.54, 37.19, 12.8247, 80.0458, 2.6],
    [0, 61.54, 37.25, 12.8247, 80.0458, 2.9],
    [0, 61.54, 37.38, 12.8247, 80.0458, 3.0],
    [0, 61.54, 37.31, 12.8247, 80.0458, 3.1],
    [0, 61.54, 37.44, 12.8247, 80.0458, 3.2],
    [0, 61.54, 37.31, 12.8247, 80.0458, 3.0],
    [0, 85.47, 37.38, 12.8247, 80.0458, 4.3],
    [0, 85.47, 37.44, 12.8247, 80.0458, 4.4],
    [0, 85.47, 37.19, 12.8247, 80.0458, 4.0],
    [0, 85.47, 37.38, 12.8247, 80.0458, 4.5],
    [0, 85.47, 37.19, 12.8247, 80.0458, 4.1],
    [0, 85.47, 37.31, 12.8247, 80.0458, 4.2],
    [0, 85.47, 37.38, 12.8247, 80.0458, 4.4],
    [1, 105.62, 38.21, 12.8247, 80.0458, 6.1],
    [1, 110.75, 38.30, 12.8247, 80.0458, 6.4],
    [1, 115.20, 38.42, 12.8247, 80.0458, 6.9],
    [1, 120.88, 38.50, 12.8247, 80.0458, 7.3],
    [1, 125.45, 38.64, 12.8247, 80.0458, 7.7],
]

COLUMNS = ["SOS", "Heart Rate", "Temperature", "latitude", "longitude", "EDA"]


def get_next_data_point():
    """Return the next row in circular fashion as a DataFrame with timestamp index."""
    index = st.session_state.get("data_index", 0)
    row = DATA_ARRAY[index]
    st.session_state.data_index = (index + 1) % len(DATA_ARRAY)
    
    # Convert to a DataFrame with proper structure
    df = pd.DataFrame([row], columns=COLUMNS)
    df.index = [datetime.now()]
    return df



# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=COLUMNS)
if "stream" not in st.session_state:
    st.session_state.stream = False
if "data_index" not in st.session_state:
    st.session_state.data_index = 0


def toggle_streaming():
    st.session_state.stream = not st.session_state.stream


st.title("📊 Real-Time Sensor Dashboard")
st.sidebar.slider(
    "Update interval (seconds)", 0.5, 5.0, value=1.0, step=0.5, key="run_every"
)
st.sidebar.button(
    "Start streaming", disabled=st.session_state.stream, on_click=toggle_streaming
)
st.sidebar.button(
    "Stop streaming", disabled=not st.session_state.stream, on_click=toggle_streaming
)

run_every = st.session_state.run_every if st.session_state.stream else None


@st.fragment(run_every=run_every)
def show_latest_data():
    new_data = get_next_data_point()
    st.session_state.data = pd.concat([st.session_state.data, new_data])
    st.session_state.data = st.session_state.data[-100:]
    if st.session_state.data["SOS"].iloc[-1] == 1:
        with st.container():
            st.markdown(
                """
                <div style="padding: 1rem; background-color: #ff4d4d; border-radius: 10px; text-align: center;">
                    <h2 style="color: white; margin: 0;">🚨 STRESS ALERT: SOS Triggered!</h2>
                    <p style="color: white;">Immediate attention may be required.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.subheader("Heart Rate")
    st.line_chart(st.session_state.data[["Heart Rate"]] )
    st.subheader("Temperature")
    st.line_chart(st.session_state.data[["Temperature"]])
    st.subheader("Electrodermal Activity (EDA)")
    st.line_chart(st.session_state.data[["EDA"]])
    st.subheader("SOS Signal")
    

@st.fragment(run_every="10000s")
def show_map_data():
    st.write("Map Data")
    st.map(st.session_state.data[["latitude", "longitude"]])

show_latest_data()
show_map_data()
