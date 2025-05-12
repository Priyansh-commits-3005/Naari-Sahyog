import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import serial
import serial.tools.list_ports
import time
import joblib

# Dashboard title and configuration
st.set_page_config(page_title="Naari Sahyog - Real-Time Sensor Dashboard", layout="wide")
st.title("📊 Real-Time Sensor Dashboard")

# Initialize session state for data storage and control flags
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["SOS", "Heart Rate", "Temperature", "latitude", "longitude", "EDA"])
    # Add timestamps as the index
    st.session_state.data.index = pd.DatetimeIndex([])
if "stream" not in st.session_state:
    st.session_state.stream = False
if "serial_connected" not in st.session_state:
    st.session_state.serial_connected = False
if "serial_obj" not in st.session_state:
    st.session_state.serial_obj = None
if "error_message" not in st.session_state:
    st.session_state.error_message = ""
if "data_buffer" not in st.session_state:
    st.session_state.data_buffer = None

# EDA hardcoded data - kept separate as requested
EDA_VALUES = [4.2, 4.3, 4.1, 2.7, 2.6, 2.9, 3.0, 3.1, 3.2, 3.0, 4.3, 4.4, 4.0]
if "eda_index" not in st.session_state:
    st.session_state.eda_index = 0

class DataBuffer:
    def __init__(self):
        self.buffer = ""
        
    def add_data(self, new_data):
        try:
            self.buffer += new_data.decode('utf-8')
        except Exception:
            return None
            
    def get_complete_message(self):
        try:
            messages = []
            while '[' in self.buffer and ']' in self.buffer:
                start = self.buffer.find('[')
                end = self.buffer.find(']') + 1
                
                if start > end:
                    self.buffer = self.buffer[start:]
                    continue
                    
                complete_message = self.buffer[start:end]
                messages.append(complete_message)
                self.buffer = self.buffer[end:]
                
            return messages
        except Exception:
            self.buffer = ""
            return None

def parse_data(message):
    try:
        if "Triple tap detected" in message:
            return "SOS_ACTIVATED"
            
        if message.startswith('[') and message.endswith(']'):
            data_str = message.strip('[]')
            values = [x.strip().strip(',') for x in data_str.split(',')]
            values = [v for v in values if v]
            
            if len(values) == 5:
                return {
                    'sos_signal': int(float(values[0])),
                    'heart_rate': float(values[1]),
                    'temperature': float(values[2]),
                    'latitude': float(values[3]),
                    'longitude': float(values[4].strip('\r\n'))
                }
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return None
    return None

def get_next_eda():
    """Return the next hardcoded EDA value in circular fashion."""
    index = st.session_state.eda_index
    value = EDA_VALUES[index]
    st.session_state.eda_index = (index + 1) % len(EDA_VALUES)
    return value

def get_available_ports():
    """Get list of available serial ports."""
    return [p.device for p in serial.tools.list_ports.comports()]

def connect_serial():
    """Connect to the serial port."""
    try:
        com_port = st.session_state.com_port
        baud_rate = st.session_state.baud_rate
        
        st.session_state.serial_obj = serial.Serial(com_port, baud_rate, timeout=1)
        st.session_state.serial_connected = True
        st.session_state.data_buffer = DataBuffer()
        st.session_state.error_message = ""
        st.session_state.stream = True  # Automatically start streaming
        
    except Exception as e:
        st.session_state.error_message = f"Error connecting to serial port: {str(e)}"
        st.session_state.serial_connected = False

def disconnect_serial():
    """Disconnect from the serial port."""
    try:
        if st.session_state.serial_obj and st.session_state.serial_obj.is_open:
            st.session_state.serial_obj.close()
        st.session_state.serial_connected = False
        st.session_state.stream = False
        st.session_state.serial_obj = None
    except Exception as e:
        st.session_state.error_message = f"Error disconnecting: {str(e)}"

def toggle_streaming():
    """Toggle data streaming on/off."""
    if not st.session_state.serial_connected and st.session_state.stream:
        st.session_state.error_message = "Cannot stream without a serial connection"
        st.session_state.stream = False
        return
    
    st.session_state.stream = not st.session_state.stream

# Sidebar for controls
st.sidebar.header("Connection Settings")

# Serial port selection
available_ports = get_available_ports()
default_port = "COM12" if "COM12" in available_ports else (available_ports[0] if available_ports else "")
selected_port = st.sidebar.selectbox("Select COM Port", 
                                  options=available_ports,
                                  index=available_ports.index(default_port) if default_port in available_ports and available_ports else 0,
                                  key="com_port")

# Baud rate selection
st.sidebar.selectbox("Baud Rate", 
                  options=[9600, 19200, 38400, 57600, 115200], 
                  index=4,  # Default to 115200
                  key="baud_rate")

# Connect/Disconnect buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    st.button("Connect", 
             on_click=connect_serial, 
             disabled=st.session_state.serial_connected)
with col2:
    st.button("Disconnect", 
             on_click=disconnect_serial, 
             disabled=not st.session_state.serial_connected)

# Connection status indicator
if st.session_state.serial_connected:
    st.sidebar.success("✅ Connected")
else:
    st.sidebar.error("❌ Disconnected")

# Error message display
if st.session_state.error_message:
    st.sidebar.error(st.session_state.error_message)

# Streaming controls
st.sidebar.header("Data Streaming")
st.sidebar.slider(
    "Update interval (seconds)", 0.5, 5.0, value=1.0, step=0.5, key="run_every"
)

col3, col4 = st.sidebar.columns(2)
with col3:
    st.button(
        "Start streaming", 
        disabled=st.session_state.stream or not st.session_state.serial_connected, 
        on_click=toggle_streaming
    )
with col4:
    st.button(
        "Stop streaming", 
        disabled=not st.session_state.stream, 
        on_click=toggle_streaming
    )

# Run button that connects and begins streaming immediately
if st.sidebar.button("Run Dashboard", type="primary"):
    if not st.session_state.serial_connected:
        connect_serial()
    if not st.session_state.stream and st.session_state.serial_connected:
        st.session_state.stream = True

# Debug options
if st.sidebar.checkbox("Show debug info", value=False):
    with st.sidebar.expander("Debug Information"):
        st.write(f"Connected: {st.session_state.serial_connected}")
        st.write(f"Streaming: {st.session_state.stream}")
        if not st.session_state.data.empty:
            st.write("Latest data point:")
            st.write(st.session_state.data.iloc[-1])

run_every = st.session_state.run_every if st.session_state.stream else None

# Data display fragment
@st.fragment(run_every=run_every)
def show_latest_data():
    # Process data from serial if connected and streaming
    if st.session_state.stream and st.session_state.serial_connected and st.session_state.serial_obj:
        try:
            if st.session_state.serial_obj.in_waiting:
                raw_data = st.session_state.serial_obj.read(st.session_state.serial_obj.in_waiting)
                st.session_state.data_buffer.add_data(raw_data)
                messages = st.session_state.data_buffer.get_complete_message()
                
                if messages:
                    for message in messages:
                        result = parse_data(message)
                        
                        if result == "SOS_ACTIVATED":
                            # Handle SOS activation
                            new_data = pd.DataFrame([[1, 0, 0, 0, 0, 0]], 
                                                  columns=["SOS", "Heart Rate", "Temperature", "latitude", "longitude", "EDA"])
                            new_data.index = [datetime.now()]
                            st.session_state.data = pd.concat([st.session_state.data, new_data])
                        
                        elif isinstance(result, dict):
                            # Create a new data point
                            new_data = pd.DataFrame([[
                                result['sos_signal'], 
                                result['heart_rate'], 
                                result['temperature'],
                                result['latitude'],
                                result['longitude'],
                                get_next_eda()  # Add hardcoded EDA value
                            ]], columns=["SOS", "Heart Rate", "Temperature", "latitude", "longitude", "EDA"])
                            
                            new_data.index = [datetime.now()]
                            st.session_state.data = pd.concat([st.session_state.data, new_data])
                            st.session_state.data = st.session_state.data[-100:]  # Keep only last 100 points
        
        except Exception as e:
            st.session_state.error_message = f"Error reading serial data: {str(e)}"
    
    # Display SOS alert if triggered
    if not st.session_state.data.empty and "SOS" in st.session_state.data.columns and st.session_state.data["SOS"].iloc[-1] == 1:
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
    
    # Create a 2-column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Heart Rate")
        if not st.session_state.data.empty and "Heart Rate" in st.session_state.data.columns:
            valid_hr_data = st.session_state.data[st.session_state.data["Heart Rate"] > 0]
            if not valid_hr_data.empty:
                st.line_chart(valid_hr_data[["Heart Rate"]])
            else:
                st.info("No valid heart rate data received yet")
        else:
            st.info("No heart rate data received yet")
    
    with col2:
        st.subheader("Temperature")
        if not st.session_state.data.empty and "Temperature" in st.session_state.data.columns:
            valid_temp_data = st.session_state.data[st.session_state.data["Temperature"] > 0]
            if not valid_temp_data.empty:
                st.line_chart(valid_temp_data[["Temperature"]])
            else:
                st.info("No valid temperature data received yet")
        else:
            st.info("No temperature data received yet")
    
    # EDA in a separate section
    st.subheader("Electrodermal Activity (EDA) - Hardcoded")
    if not st.session_state.data.empty and "EDA" in st.session_state.data.columns:
        st.line_chart(st.session_state.data[["EDA"]])
    else:
        st.info("No EDA data generated yet")

# Map display fragment - updated less frequently
@st.fragment(run_every=5.0)  # Update every 5 seconds
def show_map_data():
    st.subheader("Location Data")
    if not st.session_state.data.empty and "latitude" in st.session_state.data.columns:
        # Filter for valid GPS coordinates
        valid_gps = st.session_state.data[(st.session_state.data["latitude"] != 0) & 
                                         (st.session_state.data["longitude"] != 0)]
        
        if not valid_gps.empty:
            # Get the most recent location
            latest_data = valid_gps[["latitude", "longitude"]].iloc[-1:]
            st.map(latest_data)
        else:
            st.info("No valid location data received yet")
    else:
        st.info("No location data received yet")

# Display the data
show_latest_data()
show_map_data()

# Cleanup when the app exits
def cleanup():
    disconnect_serial()

# Register the cleanup handler
import atexit
atexit.register(cleanup)