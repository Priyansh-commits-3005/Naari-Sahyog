import streamlit as st
import serial
import time
import pandas as pd
import joblib
from twilio.rest import Client
import profiles
# Dictionary to store Twilio profiles
TWILIO_PROFILES = profiles.TWILIO_PROFILES

def initialize_serial():
    try:
        ser = serial.Serial('COM12', 115200, timeout=0)
        return ser
    except Exception as e:
        st.error(f"Error connecting to serial port: {e}")
        return None

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
    except Exception:
        return None
    return None

def process_result(result, use_real_data):
    def temperature_in_fahrenheit(celsius):
        return celsius * 9 / 5 + 32

    if use_real_data:
        temperature_f = temperature_in_fahrenheit(result['temperature'])
        heart_rate = result['heart_rate']
        blood_oxygen = result['blood_oxygen']#.
    else:
        temperature_f = 98
        heart_rate = 60
        blood_oxygen = 95
    
    processed_data = {
        "body_temperature": temperature_f,
        "blood_oxygen": blood_oxygen,#.
        "heart_rate": heart_rate
    }
    return processed_data

def check_data_fraudulency(result, use_real_data):
    processed_data = process_result(result, use_real_data)
    data_to_be_predicted = pd.DataFrame([processed_data])
    model = joblib.load("C:/codes/hackathon/SIH/V1.0/aprit-github/frontend/SIH-new/arduino-code/SVM.pkl")
    # model = joblib.load("C:/codes/Projects/Naari-Sahyog/model/CNN_Model_Stress_v1.h5")
    predicted_stress_level = model.predict(data_to_be_predicted)[0]
    return predicted_stress_level >= 3

def send_twilio_message(result, twilio_profile):
    account_sid = twilio_profile["account_sid"]
    auth_token = twilio_profile["auth_token"]
    twilio_from = twilio_profile["twilio_from"]
    client = Client(account_sid, auth_token)
    message_text = (
        f"Your emergency contact {twilio_from} needs immediate help at the location "
        f"coordinates: https://www.google.com/maps?q=12.8235,80.0424 \n"
        f"Their basic statistics: Heart Rate - {result['heart_rate']} | "
        f"Body Temperature: {result['temperature']}.\n"
        "The local authorities are on their way. Kindly contact them to ensure safety.\n"
        "- Naari Sahyog"
    )
    try:
        print(message_text)
        message = client.messages.create(
            body=message_text,
            from_=twilio_from,
            # to="+919650571007",
            # to="+919837125560",
            to='+917725041995',
              # Hardcoded recipient, can be made dynamic
        )
        st.success(f"Twilio Message Sent Successfully. SID: {message.sid}")
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")

def main():
    st.title("SOS Monitoring System Configuration")
    
    # Section 1: Select Twilio Profile
    st.header("Twilio Profile Selection")
    selected_profile = st.selectbox("Choose a Twilio Profile:", options=list(TWILIO_PROFILES.keys()))
    twilio_profile = TWILIO_PROFILES[selected_profile]

    # Section 2: Data Choices
    st.header("Data Configuration")
    use_real_data = st.radio(
        "Select Data Source for Heart Rate and Temperature:",
        ("Use Real Data", "Use Fraudulent Data"),
        index=0
    ) == "Use Real Data"
    
    # Section 3: Twilio Message Toggle
    st.header("Twilio Messaging")
    send_message_flag = st.checkbox("Enable Twilio Messaging", value=True)

    # Run Button
    if st.button("Run"):
        print("Initializing SOS Monitoring System...")
        ser = initialize_serial()
        if ser:
            data_buffer = DataBuffer()
            sos_active = False
            
            try:
                while True:
                    if ser.in_waiting:
                        raw_data = ser.read(ser.in_waiting)
                        data_buffer.add_data(raw_data)
                        messages = data_buffer.get_complete_message()
                        
                        if messages:
                            for message in messages:
                                result = parse_data(message)
                                
                                if result == "SOS_ACTIVATED":
                                    sos_active = True
                                    print("Triple tap detected: SOS activated.")
                                elif isinstance(result, dict):
                                    sos_status = result.get('sos_signal')
                                    if sos_status == 1:
                                        sos_active = True
                                        st.write("SOS is ACTIVE")
                                    elif sos_status == 0:
                                        sos_active = False
                                        print("SOS is INACTIVE")
                                    
                                    if sos_active:
                                        false_positive = check_data_fraudulency(result, use_real_data)
                                        if not false_positive and send_message_flag:
                                            send_twilio_message(result, twilio_profile)
                                            return 
                                        else:
                                            st.warning("Data is fraudulent or messaging disabled.")
                                        sos_active = False
                    time.sleep(0.001)
            except KeyboardInterrupt:
                print("Program terminated by user.")
            finally:
                ser.close()
                print("Serial connection closed.")


if __name__ == "__main__":
    main()
